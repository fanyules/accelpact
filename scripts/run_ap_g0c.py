#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from accelpact import (  # noqa: E402
    LifecycleOracle,
    LitmusResult,
    ResourceKind,
    append_result,
)

PROTOCOL_ID = "AP-G0C"
CONFIG_SCHEMA = "accelpact.ap_g0c.config.v1"
RANK_STATUS_SCHEMA = "accelpact.ap_g0c.rank_status.v1"
DEFAULT_PROCESS_GROUP = object()

PASS_CLASSIFICATIONS = {
    "valid_pass",
    "negative_control_detected",
    "capability_pass",
    "expected_timeout_recovered",
    "capability_unavailable",
}
EXIT_CODE_BY_CLASSIFICATION = {
    "valid_pass": 0,
    "negative_control_detected": 0,
    "capability_pass": 0,
    "expected_timeout_recovered": 0,
    "protocol_violation": 2,
    "negative_control_missed": 3,
    "harness_error": 4,
    "capability_unavailable": 5,
    "reinitialization_capability_failure": 5,
    "capability_timeout": 5,
    "inconclusive": 5,
}


class RunnerContractError(RuntimeError):
    pass


class MarkerError(RunnerContractError):
    pass


class StaleGenerationError(RunnerContractError):
    pass


class EpochFailure(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


@dataclass
class RankContext:
    config: dict[str, Any]
    platform: str
    platform_config: dict[str, Any]
    litmus_id: str
    run_id: str
    repetition: int
    source_revision: str
    raw_rank_log_reference: str
    output_dir: Path
    rank: int
    local_rank: int
    world_size: int
    torch: Any
    dist: Any
    device_api: Any
    device: Any
    schedule_digest: str
    backend_dispatch_count: int = 0
    marker_cache: dict[str, dict[int, dict[str, Any]]] = field(default_factory=dict)
    active_device_group: Any | None = None
    device_group_destroyed: bool = False
    recreate_call_started: bool = False
    replacement_group_created: bool = False

    @property
    def backend(self) -> str:
        return str(self.platform_config["device_group_backend"])

    @property
    def rank_output(self) -> Path:
        return self.output_dir / "ranks" / f"rank-{self.rank:04d}.jsonl"


@dataclass
class GenerationEvidence:
    generation: int
    oracle: LifecycleOracle
    epoch_start: int
    epoch_end: int
    dispatch_start: int
    dispatch_end: int | None = None
    ledger: list[dict[str, Any]] = field(default_factory=list)
    timestamps: dict[str, int | None] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunOutcome:
    rows: list[LitmusResult]
    classification: str
    active_device_group: Any | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AccelPact AP-G0C TP2 litmus")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "ap_g0c.json",
    )
    parser.add_argument("--platform", choices=("a100", "910b"), required=True)
    parser.add_argument("--litmus", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--raw-rank-log-reference")
    args = parser.parse_args()
    if args.repetition <= 0:
        parser.error("--repetition must be positive")
    if not args.run_id.strip():
        parser.error("--run-id must be non-empty")
    if not args.source_revision.strip():
        parser.error("--source-revision must be non-empty")
    return args


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CONFIG_SCHEMA:
        raise RunnerContractError("unexpected AP-G0C config schema")
    if payload.get("world_size") != 2:
        raise RunnerContractError("AP-G0C requires world_size=2")
    if payload.get("dtype") != "int64" or payload.get("tensor_shape") != [4]:
        raise RunnerContractError("AP-G0C requires int64 tensors with shape [4]")
    if payload.get("reduce_op") != "sum":
        raise RunnerContractError("AP-G0C requires SUM")
    litmus = payload.get("litmus_order")
    if not isinstance(litmus, list) or not all(
        isinstance(item, str) for item in litmus
    ):
        raise RunnerContractError("litmus_order must be a string list")
    return payload


def _required_env_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RunnerContractError(f"torchrun environment variable {name} is missing")
    try:
        return int(value)
    except ValueError as error:
        raise RunnerContractError(f"{name} must be an integer") from error


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _schedule_digest(
    config: dict[str, Any],
    *,
    platform: str,
    litmus_id: str,
    run_id: str,
    repetition: int,
    source_revision: str,
) -> str:
    return _canonical_digest(
        {
            "config": config,
            "litmus_id": litmus_id,
            "platform": platform,
            "protocol_id": PROTOCOL_ID,
            "repetition": repetition,
            "run_id": run_id,
            "source_revision": source_revision,
        }
    )


def _resolve_runtime(platform_config: dict[str, Any]) -> tuple[Any, Any, Any]:
    torch = importlib.import_module("torch")
    if platform_config["device_type"] == "npu":
        importlib.import_module("torch_npu")
    dist = importlib.import_module("torch.distributed")
    device_api = getattr(torch, str(platform_config["device_type"]), None)
    if device_api is None:
        raise RunnerContractError(
            f"torch.{platform_config['device_type']} is unavailable"
        )
    return torch, dist, device_api


def build_context(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    runtime: tuple[Any, Any, Any] | None = None,
) -> RankContext:
    if args.litmus not in config["litmus_order"]:
        raise RunnerContractError(f"unknown AP-G0C litmus: {args.litmus}")
    platform_config = config["platforms"].get(args.platform)
    if not isinstance(platform_config, dict):
        raise RunnerContractError(f"platform {args.platform} is not configured")

    rank = _required_env_int("RANK")
    local_rank = _required_env_int("LOCAL_RANK")
    world_size = _required_env_int("WORLD_SIZE")
    if world_size != config["world_size"]:
        raise RunnerContractError(
            f"WORLD_SIZE={world_size} does not match frozen world_size=2"
        )
    if rank not in range(world_size) or local_rank not in range(world_size):
        raise RunnerContractError("rank and local rank must both be in [0, world_size)")
    if rank != local_rank:
        raise RunnerContractError("single-host AP-G0C requires RANK == LOCAL_RANK")

    for name, expected in platform_config.get("launch_environment", {}).items():
        if os.environ.get(name) != expected:
            raise RunnerContractError(f"{name} must equal {expected!r} before launch")

    torch, dist, device_api = runtime or _resolve_runtime(platform_config)
    device_api.set_device(local_rank)
    device = torch.device(f"{platform_config['device_type']}:{local_rank}")
    raw_log = args.raw_rank_log_reference or f"rank-{rank:04d}.log"
    context = RankContext(
        config=config,
        platform=args.platform,
        platform_config=platform_config,
        litmus_id=args.litmus,
        run_id=args.run_id,
        repetition=args.repetition,
        source_revision=args.source_revision,
        raw_rank_log_reference=raw_log,
        output_dir=args.output_dir,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        torch=torch,
        dist=dist,
        device_api=device_api,
        device=device,
        schedule_digest=_schedule_digest(
            config,
            platform=args.platform,
            litmus_id=args.litmus,
            run_id=args.run_id,
            repetition=args.repetition,
            source_revision=args.source_revision,
        ),
    )
    if context.rank_output.exists():
        raise FileExistsError(f"refusing to overwrite {context.rank_output}")
    return context


def _initialize_control_group(context: RankContext) -> None:
    context.dist.init_process_group(
        backend=context.platform_config["control_group_backend"],
        timeout=timedelta(seconds=int(context.config["control_group_timeout_seconds"])),
    )


def _create_device_group(context: RankContext) -> Any:
    return context.dist.new_group(
        ranks=list(range(context.world_size)),
        backend=context.backend,
        timeout=timedelta(seconds=int(context.config["device_group_timeout_seconds"])),
    )


def _is_explicitly_unavailable(error: BaseException) -> bool:
    if isinstance(error, (ImportError, ModuleNotFoundError, NotImplementedError)):
        return True
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "backend is unavailable",
            "does not support",
            "not available",
            "not implemented",
            "unknown backend",
            "unsupported backend",
        )
    )


def _marker_payload(
    context: RankContext, phase: str, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "litmus_id": context.litmus_id,
        "phase": phase,
        "protocol_id": PROTOCOL_ID,
        "published_at_ns": time.time_ns(),
        "rank": context.rank,
        "run_id": context.run_id,
        "schedule_digest": context.schedule_digest,
        "world_size": context.world_size,
        **(extra or {}),
    }


def _marker_path(context: RankContext, phase: str, rank: int) -> Path:
    return context.output_dir / "control" / phase / f"rank-{rank:04d}.json"


def publish_marker(
    context: RankContext, phase: str, extra: dict[str, Any] | None = None
) -> Path:
    final_path = _marker_path(context, phase, context.rank)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.parent / f".rank-{context.rank:04d}.{os.getpid()}.tmp"
    payload = _marker_payload(context, phase, extra)
    try:
        with temp_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    payload,
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, final_path)
        temp_path.unlink()
    except FileExistsError as error:
        raise MarkerError(f"marker publication would overwrite {final_path}") from error
    except OSError as error:
        raise MarkerError(f"could not publish marker {final_path}: {error}") from error
    return final_path


def _read_marker(context: RankContext, phase: str, rank: int) -> dict[str, Any]:
    path = _marker_path(context, phase, rank)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MarkerError(f"invalid marker {path}: {error}") from error
    expected = {
        "litmus_id": context.litmus_id,
        "phase": phase,
        "protocol_id": PROTOCOL_ID,
        "rank": rank,
        "run_id": context.run_id,
        "schedule_digest": context.schedule_digest,
        "world_size": context.world_size,
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise MarkerError(f"stale or mismatched marker {path}")
    return payload


def wait_for_markers(
    context: RankContext,
    phase: str,
    *,
    ranks: list[int] | None = None,
) -> dict[int, dict[str, Any]]:
    expected_ranks = ranks or list(range(context.world_size))
    deadline = time.monotonic() + float(context.config["oob_phase_timeout_seconds"])
    rows: dict[int, dict[str, Any]] = {}
    while len(rows) != len(expected_ranks):
        for rank in expected_ranks:
            if rank in rows:
                continue
            path = _marker_path(context, phase, rank)
            if path.exists():
                rows[rank] = _read_marker(context, phase, rank)
        if len(rows) == len(expected_ranks):
            break
        if time.monotonic() >= deadline:
            missing = sorted(set(expected_ranks) - set(rows))
            raise MarkerError(
                f"marker phase {phase} timed out; missing ranks {missing}"
            )
        time.sleep(0.01)
    context.marker_cache[phase] = rows
    return rows


def publish_and_wait(
    context: RankContext, phase: str, extra: dict[str, Any] | None = None
) -> dict[int, dict[str, Any]]:
    publish_marker(context, phase, extra)
    return wait_for_markers(context, phase)


def _payload(generation: int, epoch: int, rank: int) -> list[int]:
    return [generation, epoch, rank, 1]


def _expected_payload(generation: int, epoch: int) -> list[int]:
    return [2 * generation, 2 * epoch, 1, 2]


def _tensor_values(tensor: Any) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().tolist()]


def _run_epoch(
    context: RankContext,
    group: Any,
    oracle: LifecycleOracle,
    *,
    generation: int,
    epoch: int,
    reopen: bool,
) -> dict[str, Any]:
    initial = _payload(generation, epoch, context.rank)
    expected = _expected_payload(generation, epoch)
    tensor = context.torch.tensor(
        initial,
        dtype=context.torch.int64,
        device=context.device,
    )
    oracle.transition("enqueued")
    enqueue_started_at_ns = time.time_ns()
    work = None
    observed: list[int] | None = None
    try:
        context.backend_dispatch_count += 1
        work = context.dist.all_reduce(
            tensor,
            op=context.dist.ReduceOp.SUM,
            group=group,
            async_op=True,
        )
        wait_result = work.wait(
            timedelta(seconds=int(context.config["work_wait_timeout_seconds"]))
        )
        if wait_result is False:
            raise TimeoutError("Work.wait returned false")
        observed = _tensor_values(tensor)
    except Exception as error:  # noqa: BLE001 - retain backend failure evidence
        raise EpochFailure(
            f"{type(error).__name__}: {error}",
            {
                "enqueue_started_at_ns": enqueue_started_at_ns,
                "epoch": epoch,
                "error_type": type(error).__name__,
                "generation": generation,
                "work_handle_present": work is not None,
            },
        ) from error

    completion_at_ns = time.time_ns()
    oracle.transition("completed")
    entry = {
        "completion_at_ns": completion_at_ns,
        "completion_count": 1,
        "enqueue_started_at_ns": enqueue_started_at_ns,
        "epoch": epoch,
        "expected_payload_digest": _canonical_digest(expected),
        "generation": generation,
        "observed_payload_digest": _canonical_digest(observed),
        "payload_matches": observed == expected,
        "work_handle_present": work is not None,
    }
    if observed != expected:
        raise EpochFailure(
            f"generation {generation} epoch {epoch} returned {observed}, "
            f"expected {expected}",
            {**entry, "error_type": "PayloadMismatch"},
        )
    oracle.transition("reusable_same_generation")
    if reopen:
        oracle.transition("epoch_open")
    return entry


def _run_epochs(
    context: RankContext,
    group: Any,
    evidence: GenerationEvidence,
    *,
    count: int,
    reopen_final: bool = False,
) -> None:
    try:
        for offset in range(count):
            epoch = evidence.epoch_start + offset
            try:
                entry = _run_epoch(
                    context,
                    group,
                    evidence.oracle,
                    generation=evidence.generation,
                    epoch=epoch,
                    reopen=offset < count - 1 or reopen_final,
                )
            except EpochFailure:
                evidence.epoch_end = epoch
                raise
            evidence.ledger.append(entry)
            evidence.epoch_end = epoch
    finally:
        evidence.dispatch_end = context.backend_dispatch_count


def _classification_error(
    classification: str, error: BaseException | str | None
) -> tuple[bool, str | None, str | None]:
    passed = classification in PASS_CLASSIFICATIONS
    if passed:
        normalized = type(error).__name__ if isinstance(error, BaseException) else None
        return True, None, normalized
    if error is None:
        message = f"{classification}: see observations"
        return False, message, None
    if isinstance(error, BaseException):
        return False, f"{type(error).__name__}: {error}", type(error).__name__
    return False, error, None


def _make_result(
    context: RankContext,
    evidence: GenerationEvidence,
    classification: str,
    *,
    error: BaseException | str | None = None,
    extra: dict[str, Any] | None = None,
) -> LitmusResult:
    passed, result_error, normalized_error_type = _classification_error(
        classification, error
    )
    completion_count = sum(
        int(row.get("completion_count", 0)) for row in evidence.ledger
    )
    work_handle_present = any(
        bool(row.get("work_handle_present")) for row in evidence.ledger
    )
    expected_digests = [row["expected_payload_digest"] for row in evidence.ledger]
    observed_digests = [row["observed_payload_digest"] for row in evidence.ledger]
    partial_fault_g0 = (
        context.litmus_id == "collective_partial_epoch_timeout_recreate"
        and evidence.generation == 0
    )
    local_enqueue_role = (
        "submit_fault"
        if partial_fault_g0 and context.rank in context.config["fault_submit_ranks"]
        else "skip_fault"
        if partial_fault_g0
        else "submit"
    )
    planned_submit_ranks = (
        context.config["fault_submit_ranks"]
        if partial_fault_g0
        else list(range(context.world_size))
    )
    dispatch_end = (
        evidence.dispatch_end
        if evidence.dispatch_end is not None
        else context.backend_dispatch_count
    )
    observed_error = (
        f"{type(error).__name__}: {error}"
        if isinstance(error, BaseException)
        else error
    )
    observations = {
        "backend_dispatch_count": (dispatch_end - evidence.dispatch_start),
        "classification": classification,
        "collective_backend": context.backend,
        "completion_count": completion_count,
        "current_generation": evidence.generation,
        "destroy_marker_bitmap": [
            rank in context.marker_cache.get("destroyed", {})
            for rank in range(context.world_size)
        ],
        "dtype": context.config["dtype"],
        "epoch_end": evidence.epoch_end,
        "epoch_ledger": evidence.ledger,
        "epoch_start": evidence.epoch_start,
        "expected_payload_digest": _canonical_digest(expected_digests),
        "generation": evidence.generation,
        "litmus_id": context.litmus_id,
        "local_enqueue_role": local_enqueue_role,
        "local_rank": context.local_rank,
        "logical_device": str(context.device),
        "normalized_error_type": normalized_error_type,
        "observed_error": observed_error,
        "observed_payload_digest": _canonical_digest(observed_digests),
        "planned_submit_ranks": planned_submit_ranks,
        "protocol_id": PROTOCOL_ID,
        "rank": context.rank,
        "raw_rank_log_reference": context.raw_rank_log_reference,
        "reduce_op": context.config["reduce_op"],
        "repetition": context.repetition,
        "run_id": context.run_id,
        "schedule_digest": context.schedule_digest,
        "seed": context.config["seed"],
        "source_revision": context.source_revision,
        "tensor_shape": context.config["tensor_shape"],
        "work_handle_present": work_handle_present,
        "world_size": context.world_size,
        **evidence.timestamps,
        **evidence.extra,
        **(extra or {}),
    }
    if evidence.ledger:
        observations.setdefault(
            "enqueue_started_at_ns", evidence.ledger[0]["enqueue_started_at_ns"]
        )
        observations.setdefault(
            "completion_at_ns", evidence.ledger[-1]["completion_at_ns"]
        )
    return LitmusResult.from_oracle(
        evidence.oracle,
        litmus_id=context.litmus_id,
        backend=context.backend,
        passed=passed,
        observations=observations,
        error=result_error,
    )


def _new_evidence(
    context: RankContext,
    generation: int,
    epoch_start: int,
) -> GenerationEvidence:
    return GenerationEvidence(
        generation=generation,
        oracle=LifecycleOracle(
            ResourceKind.COLLECTIVE,
            f"comm-g{generation:03d}-rank{context.rank:04d}",
        ),
        epoch_start=epoch_start,
        epoch_end=epoch_start - 1,
        dispatch_start=context.backend_dispatch_count,
        dispatch_end=context.backend_dispatch_count,
    )


def _guard_generation(current_generation: int, request_generation: int) -> None:
    if request_generation != current_generation:
        raise StaleGenerationError(
            f"request generation {request_generation} is not current "
            f"generation {current_generation}"
        )


def stale_generation_dispatch(context: RankContext, group: Any) -> RunOutcome:
    current = int(context.config["current_generation_for_negative_control"])
    retired = int(context.config["retired_generation_for_negative_control"])
    evidence = _new_evidence(context, current, 0)
    try:
        _guard_generation(current, retired)
    except StaleGenerationError:
        detected = True
    else:
        detected = False

    if not detected:
        result = _make_result(
            context,
            evidence,
            "negative_control_missed",
            error="host guard would dispatch the retired generation",
            extra={
                "current_generation": current,
                "request_generation": retired,
                "retired_generation": retired,
                "stale_request_backend_dispatch_count": 0,
                "stale_request_rejected": False,
                "would_dispatch": True,
            },
        )
        return RunOutcome([result], "negative_control_missed", group)

    try:
        _run_epochs(context, group, evidence, count=1)
    except EpochFailure as error:
        evidence.extra.update(error.details)
        result = _make_result(
            context,
            evidence,
            "protocol_violation",
            error=error,
            extra={
                "current_generation": current,
                "request_generation": retired,
                "retired_generation": retired,
                "stale_request_backend_dispatch_count": 0,
                "stale_request_rejected": True,
                "would_dispatch": False,
            },
        )
        return RunOutcome([result], "protocol_violation", group)
    result = _make_result(
        context,
        evidence,
        "negative_control_detected",
        extra={
            "current_generation": current,
            "request_generation": retired,
            "retired_generation": retired,
            "stale_request_backend_dispatch_count": 0,
            "stale_request_rejected": True,
            "would_dispatch": False,
        },
    )
    return RunOutcome([result], "negative_control_detected", group)


def collective_same_generation_reuse(context: RankContext, group: Any) -> RunOutcome:
    evidence = _new_evidence(context, 0, 0)
    try:
        _run_epochs(
            context,
            group,
            evidence,
            count=int(context.config["same_generation_epochs"]),
        )
    except EpochFailure as error:
        evidence.extra.update(error.details)
        result = _make_result(context, evidence, "protocol_violation", error=error)
        return RunOutcome([result], "protocol_violation", group)
    result = _make_result(context, evidence, "valid_pass")
    return RunOutcome([result], "valid_pass", group)


def _retire_and_recreate(
    context: RankContext,
    group: Any,
    evidence: GenerationEvidence,
    *,
    failed_unknown: bool,
) -> tuple[Any | None, str | None, BaseException | None, Any | None]:
    evidence.oracle.transition("aborting" if failed_unknown else "retiring")
    try:
        publish_and_wait(context, "ready_destroy")
    except MarkerError as error:
        return None, "harness_error", error, group

    evidence.timestamps["destroy_started_at_ns"] = time.time_ns()
    try:
        context.dist.destroy_process_group(group)
    except Exception as error:  # noqa: BLE001 - capability evidence
        return None, "reinitialization_capability_failure", error, group
    evidence.timestamps["destroy_completed_at_ns"] = time.time_ns()
    context.active_device_group = None
    context.device_group_destroyed = True
    evidence.oracle.transition("destroyed")

    try:
        publish_and_wait(context, "destroyed")
        publish_and_wait(context, "recreate_start")
    except MarkerError as error:
        return None, "harness_error", error, None

    evidence.timestamps["recreate_started_at_ns"] = time.time_ns()
    context.recreate_call_started = True
    try:
        replacement = _create_device_group(context)
    except Exception as error:  # noqa: BLE001 - classify capability boundary
        classification = (
            "capability_unavailable"
            if _is_explicitly_unavailable(error)
            else "reinitialization_capability_failure"
        )
        return None, classification, error, None
    context.active_device_group = replacement
    context.replacement_group_created = True
    evidence.timestamps["recreate_completed_at_ns"] = time.time_ns()
    evidence.oracle.transition("recreated")
    evidence.extra.update(
        {
            "distinct_replacement_object": replacement is not group,
            "old_object_reused": False,
        }
    )
    return replacement, None, None, replacement


def _cross_generation_failure(
    context: RankContext,
    g0: GenerationEvidence,
    classification: str,
    error: BaseException | str,
    active_group: Any | None,
) -> RunOutcome:
    result = _make_result(context, g0, classification, error=error)
    return RunOutcome([result], classification, active_group)


def collective_clean_destroy_recreate(context: RankContext, group: Any) -> RunOutcome:
    g0 = _new_evidence(context, 0, 0)
    try:
        _run_epochs(
            context,
            group,
            g0,
            count=int(context.config["clean_recreate_warmup_epochs"]),
        )
    except EpochFailure as error:
        g0.extra.update(error.details)
        result = _make_result(context, g0, "protocol_violation", error=error)
        return RunOutcome([result], "protocol_violation", group)

    replacement, classification, error, active_group = _retire_and_recreate(
        context, group, g0, failed_unknown=False
    )
    if replacement is None:
        assert classification is not None and error is not None
        return _cross_generation_failure(
            context, g0, classification, error, active_group
        )

    g1 = _new_evidence(context, 1, 0)
    g1.extra.update(
        {
            "distinct_replacement_object": replacement is not group,
            "old_object_reused": False,
        }
    )
    if replacement is group:
        message = "new_group returned the destroyed process-group object"
        rows = [
            _make_result(
                context, evidence, "reinitialization_capability_failure", error=message
            )
            for evidence in (g0, g1)
        ]
        return RunOutcome(rows, "reinitialization_capability_failure", replacement)
    try:
        _run_epochs(
            context,
            replacement,
            g1,
            count=int(context.config["recovery_epochs"]),
        )
    except EpochFailure as error:
        g1.extra.update(error.details)
        rows = [
            _make_result(
                context,
                evidence,
                "reinitialization_capability_failure",
                error=error,
            )
            for evidence in (g0, g1)
        ]
        return RunOutcome(rows, "reinitialization_capability_failure", replacement)

    rows = [
        _make_result(context, g0, "capability_pass"),
        _make_result(context, g1, "capability_pass"),
    ]
    return RunOutcome(rows, "capability_pass", replacement)


def _fault_tensor(context: RankContext, generation: int, epoch: int) -> Any:
    return context.torch.tensor(
        _payload(generation, epoch, context.rank),
        dtype=context.torch.int64,
        device=context.device,
    )


def _run_designated_fault(
    context: RankContext, group: Any, evidence: GenerationEvidence
) -> tuple[bool, dict[str, Any]]:
    epoch = int(context.config["fault_epoch"])
    evidence.oracle.transition("enqueued")
    if context.rank not in context.config["fault_submit_ranks"]:
        rank_zero = wait_for_markers(context, "fault_observed", ranks=[0])[0]
        fault_observed = rank_zero.get("fault_observed")
        if not isinstance(fault_observed, bool):
            raise MarkerError("rank-0 fault_observed marker lacks a boolean outcome")
        publish_and_wait(
            context,
            "fault_observed",
            {
                "fault_observed": fault_observed,
                "local_enqueued": False,
                "rank_0_error": rank_zero.get("error"),
                "rank_0_error_type": rank_zero.get("error_type"),
                "rank_0_work_handle_present": rank_zero.get("work_handle_present"),
            },
        )
        evidence.epoch_end = epoch
        evidence.dispatch_end = context.backend_dispatch_count
        if fault_observed:
            evidence.oracle.transition("failed_unknown")
        return fault_observed, rank_zero

    tensor = _fault_tensor(context, 0, epoch)
    started_at_ns = time.time_ns()
    work = None
    fault_error: BaseException | None = None
    try:
        context.backend_dispatch_count += 1
        work = context.dist.all_reduce(
            tensor,
            op=context.dist.ReduceOp.SUM,
            group=group,
            async_op=True,
        )
    except Exception as error:  # noqa: BLE001 - synchronous dispatch evidence
        fault_error = error

    if work is not None:
        try:
            wait_result = work.wait(
                timedelta(seconds=int(context.config["work_wait_timeout_seconds"]))
            )
            if wait_result is False:
                raise TimeoutError("Work.wait returned false")
        except Exception as error:  # noqa: BLE001 - expected wait-time signal
            fault_error = error

    fault_observed = work is not None and fault_error is not None
    if work is None and fault_error is None:
        fault_error = RunnerContractError("all_reduce returned no Work handle")

    details = {
        "error": str(fault_error) if fault_error is not None else None,
        "error_type": (type(fault_error).__name__ if fault_error is not None else None),
        "fault_observed": fault_observed,
        "local_enqueued": True,
        "wait_started_at_ns": started_at_ns,
        "work_handle_present": work is not None,
    }
    if fault_observed:
        details["timeout_at_ns"] = time.time_ns()
    publish_and_wait(context, "fault_observed", details)
    evidence.epoch_end = epoch
    evidence.dispatch_end = context.backend_dispatch_count
    if fault_observed:
        evidence.oracle.transition("failed_unknown")
        evidence.timestamps["timeout_at_ns"] = int(details["timeout_at_ns"])
    return fault_observed, details


def collective_partial_epoch_timeout_recreate(
    context: RankContext, group: Any
) -> RunOutcome:
    g0 = _new_evidence(context, 0, 0)
    try:
        _run_epochs(context, group, g0, count=1, reopen_final=True)
    except EpochFailure as error:
        g0.extra.update(error.details)
        result = _make_result(context, g0, "protocol_violation", error=error)
        return RunOutcome([result], "protocol_violation", group)

    try:
        publish_and_wait(context, "fault_ready")
        observed, details = _run_designated_fault(context, group, g0)
    except MarkerError as error:
        result = _make_result(context, g0, "harness_error", error=error)
        return RunOutcome([result], "harness_error", group)
    g0.extra.update(
        {
            "fault_details": details,
            "local_enqueued": context.rank in context.config["fault_submit_ranks"],
        }
    )
    if not observed:
        result = _make_result(
            context,
            g0,
            "inconclusive",
            error="designated rank-0 incomplete-epoch stimulus was not observed",
        )
        return RunOutcome([result], "inconclusive", group)

    replacement, classification, error, active_group = _retire_and_recreate(
        context, group, g0, failed_unknown=True
    )
    if replacement is None:
        assert classification is not None and error is not None
        return _cross_generation_failure(
            context, g0, classification, error, active_group
        )

    g1 = _new_evidence(context, 1, 0)
    replacement_fields = {
        "distinct_replacement_object": replacement is not group,
        "old_object_reused": False,
    }
    g1.extra.update(replacement_fields)
    if replacement is group:
        message = "new_group returned the destroyed process-group object"
        rows = [
            _make_result(
                context, evidence, "reinitialization_capability_failure", error=message
            )
            for evidence in (g0, g1)
        ]
        return RunOutcome(rows, "reinitialization_capability_failure", replacement)
    try:
        _run_epochs(
            context,
            replacement,
            g1,
            count=int(context.config["recovery_epochs"]),
        )
    except EpochFailure as error:
        g1.extra.update(error.details)
        rows = [
            _make_result(
                context,
                evidence,
                "reinitialization_capability_failure",
                error=error,
            )
            for evidence in (g0, g1)
        ]
        return RunOutcome(rows, "reinitialization_capability_failure", replacement)

    rows = [
        _make_result(context, g0, "expected_timeout_recovered"),
        _make_result(context, g1, "capability_pass"),
    ]
    return RunOutcome(rows, "expected_timeout_recovered", replacement)


LITMUS_FUNCTIONS = {
    "stale_generation_dispatch": stale_generation_dispatch,
    "collective_same_generation_reuse": collective_same_generation_reuse,
    "collective_clean_destroy_recreate": collective_clean_destroy_recreate,
    "collective_partial_epoch_timeout_recreate": (
        collective_partial_epoch_timeout_recreate
    ),
}


def _rank_status(
    context: RankContext,
    classification: str,
    error: BaseException | str | None = None,
) -> dict[str, Any]:
    if isinstance(error, BaseException):
        error_type = type(error).__name__
        error_text = str(error)
    else:
        error_type = None
        error_text = error
    return {
        "classification": classification,
        "error": error_text,
        "error_type": error_type,
        "litmus_id": context.litmus_id,
        "protocol_id": PROTOCOL_ID,
        "rank": context.rank,
        "run_id": context.run_id,
        "schedule_digest": context.schedule_digest,
        "schema": RANK_STATUS_SCHEMA,
        "world_size": context.world_size,
    }


def _print_status(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True))


def _safe_destroy(
    context: RankContext, group: Any = DEFAULT_PROCESS_GROUP
) -> BaseException | None:
    if group is None:
        return None
    try:
        if group is DEFAULT_PROCESS_GROUP:
            context.dist.destroy_process_group()
        else:
            context.dist.destroy_process_group(group)
    except Exception as error:  # noqa: BLE001 - cleanup is reported, never hidden
        return error
    return None


def run(
    args: argparse.Namespace,
    *,
    runtime: tuple[Any, Any, Any] | None = None,
) -> int:
    config = load_config(args.config)
    context = build_context(args, config, runtime=runtime)
    control_initialized = False
    device_group = None
    active_group = None
    try:
        _initialize_control_group(context)
        control_initialized = True
        try:
            device_group = _create_device_group(context)
        except Exception as error:  # noqa: BLE001 - zero-row run-level evidence
            classification = (
                "capability_unavailable"
                if _is_explicitly_unavailable(error)
                else "harness_error"
            )
            _print_status(_rank_status(context, classification, error))
            return EXIT_CODE_BY_CLASSIFICATION[classification]

        active_group = device_group
        context.active_device_group = device_group
        outcome = LITMUS_FUNCTIONS[context.litmus_id](context, device_group)
        active_group = outcome.active_device_group
        context.active_device_group = active_group
        for row in outcome.rows:
            append_result(context.rank_output, row)
        cleanup_error = _safe_destroy(context, active_group)
        active_group = None
        context.active_device_group = None
        if cleanup_error is not None:
            _print_status(_rank_status(context, "harness_error", cleanup_error))
            return EXIT_CODE_BY_CLASSIFICATION["harness_error"]
        _print_status(_rank_status(context, outcome.classification))
        return EXIT_CODE_BY_CLASSIFICATION[outcome.classification]
    except Exception as error:  # noqa: BLE001 - preserve a rank-level status
        _print_status(_rank_status(context, "harness_error", error))
        return EXIT_CODE_BY_CLASSIFICATION["harness_error"]
    finally:
        if context.active_device_group is not None:
            _safe_destroy(context, context.active_device_group)
            context.active_device_group = None
        if control_initialized:
            in_forbidden_gap = (
                context.device_group_destroyed and not context.replacement_group_created
            )
            if not in_forbidden_gap:
                _safe_destroy(context)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
