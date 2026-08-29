#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from accelpact import LitmusResult  # noqa: E402

WORLD_SIZE = 2
PROTOCOL_ID = "AP-G0C"
SUMMARY_SCHEMA = "accelpact.ap_g0c.adjudication.v1"

LITMUS = (
    "stale_generation_dispatch",
    "collective_same_generation_reuse",
    "collective_clean_destroy_recreate",
    "collective_partial_epoch_timeout_recreate",
)
RECREATE_LITMUS = frozenset(LITMUS[2:])

PASS_CLASSIFICATIONS = frozenset(
    {
        "valid_pass",
        "negative_control_detected",
        "capability_pass",
        "expected_timeout_recovered",
    }
)
CAPABILITY_CLASSIFICATIONS = frozenset(
    {
        "capability_unavailable",
        "reinitialization_capability_failure",
        "capability_timeout",
        "inconclusive",
    }
)
FAIL_CLASSIFICATIONS = frozenset(
    {
        "negative_control_missed",
        "reinitialization_capability_failure",
        "protocol_violation",
        "harness_error",
        "inconclusive",
    }
)
KNOWN_CLASSIFICATIONS = (
    PASS_CLASSIFICATIONS
    | CAPABILITY_CLASSIFICATIONS
    | {"negative_control_missed", "protocol_violation", "harness_error"}
)

MARKER_PHASES = {
    "collective_clean_destroy_recreate": (
        "ready_destroy",
        "destroyed",
        "recreate_start",
    ),
    "collective_partial_epoch_timeout_recreate": (
        "fault_ready",
        "fault_observed",
        "ready_destroy",
        "destroyed",
        "recreate_start",
    ),
}
KNOWN_TIMEOUT_PHASES = frozenset(
    {
        "initial_group_creation",
        "device_group_creation",
        "device_work",
        "warmup",
        "generation_1_creation",
        "generation_1_use",
        *MARKER_PHASES["collective_clean_destroy_recreate"],
        *MARKER_PHASES["collective_partial_epoch_timeout_recreate"],
    }
)

_MISSING = object()
_RANK_NAME = re.compile(r"rank[-_]?(\d+)", re.IGNORECASE)


class EvidenceError(ValueError):
    """The supplied external evidence cannot be parsed safely."""


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate one frozen AP-G0C world_size=2 run"
    )
    parser.add_argument(
        "--rank-jsonl",
        "--rank-result",
        action="append",
        default=[],
        metavar="RANK=PATH",
        help=(
            "per-rank JSONL as 0=PATH/1=PATH; a rank-0000 style filename may "
            "also be used"
        ),
    )
    parser.add_argument(
        "--marker",
        action="append",
        type=Path,
        default=[],
        help="marker JSON file or directory; may be repeated",
    )
    parser.add_argument("--launcher-evidence", type=Path)
    parser.add_argument("--output", type=Path, help="optional no-clobber summary")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvidenceError(f"rank JSONL is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise EvidenceError(f"blank JSONL row at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvidenceError(
                    f"invalid JSONL row at {path}:{line_number}"
                ) from error
            if not isinstance(row, dict):
                raise EvidenceError(
                    f"JSONL row is not an object at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceError(f"JSON evidence is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvidenceError(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON evidence is not an object: {path}")
    return value


def load_markers(paths: Sequence[Path]) -> list[dict[str, Any]]:
    marker_files: list[Path] = []
    for path in paths:
        if path.is_dir():
            marker_files.extend(sorted(path.rglob("*.json")))
        else:
            marker_files.append(path)
    return [read_json(path) for path in marker_files]


def parse_rank_path(specification: str) -> tuple[int, Path]:
    if "=" in specification:
        rank_text, path_text = specification.split("=", 1)
        rank_text = rank_text.removeprefix("rank").strip()
        try:
            rank = int(rank_text)
        except ValueError as error:
            raise EvidenceError(
                f"invalid rank JSONL specification: {specification}"
            ) from error
        path = Path(path_text)
    else:
        path = Path(specification)
        match = _RANK_NAME.search(path.name)
        if match is None:
            raise EvidenceError(
                f"cannot infer rank from JSONL filename: {specification}"
            )
        rank = int(match.group(1))
    if rank not in range(WORLD_SIZE):
        raise EvidenceError(f"rank must be 0 or 1, got {rank}")
    return rank, path


def load_rank_rows(
    specifications: Sequence[str], *, allow_missing: bool = False
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, bool]]:
    paths: dict[int, Path] = {}
    for specification in specifications:
        rank, path = parse_rank_path(specification)
        if rank in paths:
            raise EvidenceError(f"duplicate rank JSONL input for rank {rank}")
        paths[rank] = path
    if not allow_missing and set(paths) != set(range(WORLD_SIZE)):
        raise EvidenceError("rank JSONL inputs must cover exactly ranks 0 and 1")
    rows: dict[int, list[dict[str, Any]]] = {}
    present: dict[int, bool] = {}
    for rank in range(WORLD_SIZE):
        path = paths.get(rank)
        present[rank] = path is not None and path.is_file()
        if present[rank]:
            assert path is not None
            rows[rank] = read_jsonl(path)
        else:
            rows[rank] = []
            if path is not None and not allow_missing:
                raise EvidenceError(f"rank JSONL is missing: {path}")
    return rows, present


def _field(
    record: Mapping[str, Any],
    key: str,
    *,
    source: str,
    issues: list[str],
    required: bool = True,
) -> Any:
    observations = record.get("observations", _MISSING)
    if observations is not _MISSING and not isinstance(observations, Mapping):
        issues.append(f"{source}: observations must be an object")
        observations = {}
    top = record.get(key, _MISSING)
    nested = (
        observations.get(key, _MISSING) if observations is not _MISSING else _MISSING
    )
    if top is not _MISSING and nested is not _MISSING and top != nested:
        issues.append(f"{source}: conflicting {key} values")
        return top
    value = top if top is not _MISSING else nested
    if value is _MISSING:
        if required:
            issues.append(f"{source}: missing {key}")
        return None
    return value


def _launcher_bool(
    launcher: Mapping[str, Any] | None,
    primary: str,
    alias: str,
    issues: list[str],
) -> bool:
    if launcher is None:
        return False
    primary_value = launcher.get(primary, _MISSING)
    alias_value = launcher.get(alias, _MISSING)
    if (
        primary_value is not _MISSING
        and alias_value is not _MISSING
        and primary_value != alias_value
    ):
        issues.append(f"launcher: conflicting {primary}/{alias} values")
    value = primary_value if primary_value is not _MISSING else alias_value
    if value is _MISSING:
        return False
    if not isinstance(value, bool):
        issues.append(f"launcher: {primary} must be a boolean")
        return False
    return value


def _rank_exit_codes(launcher: Mapping[str, Any] | None) -> dict[int, Any]:
    if launcher is None:
        return {}
    exit_codes = launcher.get("rank_exit_codes")
    if isinstance(exit_codes, Mapping):
        values: dict[int, Any] = {}
        for key, value in exit_codes.items():
            try:
                values[int(key)] = value
            except (TypeError, ValueError):
                return {}
        return values
    if isinstance(exit_codes, list):
        return dict(enumerate(exit_codes))
    return {}


def aggregate_classifications(classifications: Sequence[str]) -> tuple[str, int]:
    values = set(classifications)
    if "harness_error" in values:
        return "harness_error", 4
    if "protocol_violation" in values:
        return "protocol_violation", 2
    if "negative_control_missed" in values:
        return "negative_control_missed", 3
    for classification in (
        "capability_timeout",
        "reinitialization_capability_failure",
        "capability_unavailable",
        "inconclusive",
    ):
        if classification in values:
            return classification, 5
    if "expected_timeout_recovered" in values:
        return "expected_timeout_recovered", 0
    if "negative_control_detected" in values:
        return "negative_control_detected", 0
    if "capability_pass" in values:
        return "capability_pass", 0
    if values == {"valid_pass"}:
        return "valid_pass", 0
    return "harness_error", 4


def _allowed_row_shapes(litmus_id: str) -> set[tuple[tuple[int, str], ...]]:
    if litmus_id == "stale_generation_dispatch":
        return {
            ((1, "negative_control_detected"),),
            ((1, "negative_control_missed"),),
            ((1, "protocol_violation"),),
        }
    if litmus_id == "collective_same_generation_reuse":
        return {((0, "valid_pass"),), ((0, "protocol_violation"),)}
    common_single = {
        ((0, "protocol_violation"),),
        ((0, "harness_error"),),
        ((0, "capability_unavailable"),),
        ((0, "reinitialization_capability_failure"),),
    }
    common_double = {
        (
            (0, "reinitialization_capability_failure"),
            (1, "reinitialization_capability_failure"),
        ),
    }
    if litmus_id == "collective_clean_destroy_recreate":
        return (
            common_single
            | common_double
            | {
                ((0, "capability_pass"), (1, "capability_pass")),
            }
        )
    if litmus_id == "collective_partial_epoch_timeout_recreate":
        return (
            common_single
            | common_double
            | {
                ((0, "inconclusive"),),
                ((0, "expected_timeout_recovered"), (1, "capability_pass")),
            }
        )
    return set()


def _successful_epoch_contract(
    litmus_id: str, generation: int, classification: str
) -> tuple[int, int, int] | None:
    if (
        litmus_id == "stale_generation_dispatch"
        and generation == 1
        and classification == "negative_control_detected"
    ):
        return 0, 0, 1
    if (
        litmus_id == "collective_same_generation_reuse"
        and generation == 0
        and classification == "valid_pass"
    ):
        return 0, 127, 128
    if litmus_id == "collective_clean_destroy_recreate":
        if generation == 0 and classification in {
            "capability_pass",
            "capability_unavailable",
            "reinitialization_capability_failure",
        }:
            return 0, 7, 8
        if generation == 1 and classification == "capability_pass":
            return 0, 127, 128
    if litmus_id == "collective_partial_epoch_timeout_recreate":
        if generation == 0 and classification in {
            "expected_timeout_recovered",
            "capability_unavailable",
            "reinitialization_capability_failure",
            "inconclusive",
        }:
            return 0, 1, 1
        if generation == 1 and classification == "capability_pass":
            return 0, 127, 128
    return None


def _validate_success_observations(
    observations: Mapping[str, Any],
    *,
    source: str,
    generation: int,
    expected: tuple[int, int, int],
    issues: list[str],
) -> None:
    epoch_start, epoch_end, completion_count = expected
    if observations.get("epoch_start") != epoch_start:
        issues.append(f"{source}: unexpected epoch_start")
    if observations.get("epoch_end") != epoch_end:
        issues.append(f"{source}: unexpected epoch_end")
    if observations.get("completion_count") != completion_count:
        issues.append(f"{source}: unexpected completion_count")
    ledger = observations.get("epoch_ledger")
    if not isinstance(ledger, list) or len(ledger) != completion_count:
        issues.append(f"{source}: epoch_ledger has the wrong cardinality")
    else:
        for offset, entry in enumerate(ledger):
            epoch = epoch_start + offset
            if not isinstance(entry, Mapping):
                issues.append(f"{source}: epoch_ledger entry is not an object")
                continue
            if entry.get("generation") != generation:
                issues.append(f"{source}: epoch_ledger generation sequence differs")
            if entry.get("epoch") != epoch:
                issues.append(f"{source}: epoch_ledger epoch sequence differs")
            if entry.get("completion_count") != 1:
                issues.append(f"{source}: epoch completion_count must equal 1")
            expected_epoch_digest = _canonical_digest([2 * generation, 2 * epoch, 1, 2])
            if entry.get("expected_payload_digest") != expected_epoch_digest:
                issues.append(
                    f"{source}: epoch expected_payload_digest violates the formula"
                )
            if entry.get("observed_payload_digest") != expected_epoch_digest:
                issues.append(
                    f"{source}: epoch observed_payload_digest differs from expected"
                )
            if entry.get("payload_matches") is not True:
                issues.append(f"{source}: epoch_ledger contains a payload mismatch")
    expected_digest = observations.get("expected_payload_digest")
    observed_digest = observations.get("observed_payload_digest")
    if not isinstance(expected_digest, str) or not expected_digest:
        issues.append(f"{source}: expected_payload_digest is missing")
    if not isinstance(observed_digest, str) or not observed_digest:
        issues.append(f"{source}: observed_payload_digest is missing")
    if expected_digest != observed_digest:
        issues.append(f"{source}: expected and observed payload digests differ")


def _validate_result_phase(
    result: LitmusResult,
    observations: Mapping[str, Any],
    *,
    litmus_id: str,
    generation: int,
    classification: str,
    rank: int,
    shape: tuple[tuple[int, str], ...],
    issues: list[str],
) -> None:
    source = f"rank {rank} g{generation}"
    targets = tuple(event.target for event in result.transitions)

    def require_final(state: str, tail: tuple[str, ...]) -> None:
        if result.final_state != state or targets[-len(tail) :] != tail:
            issues.append(f"{source}: {classification} has the wrong final_state/phase")

    if (
        litmus_id == "stale_generation_dispatch"
        and classification == "negative_control_detected"
    ) or (
        litmus_id == "collective_same_generation_reuse"
        and classification == "valid_pass"
    ):
        require_final(
            "reusable_same_generation",
            ("enqueued", "completed", "reusable_same_generation"),
        )
    elif classification == "capability_pass":
        if generation == 0:
            require_final("recreated", ("retiring", "destroyed", "recreated"))
        else:
            require_final(
                "reusable_same_generation",
                ("enqueued", "completed", "reusable_same_generation"),
            )
    elif classification == "expected_timeout_recovered":
        require_final(
            "recreated",
            ("failed_unknown", "aborting", "destroyed", "recreated"),
        )
        details = observations.get("fault_details")
        if (
            not isinstance(details, Mapping)
            or details.get("fault_observed") is not True
        ):
            issues.append(f"{source}: recovered timeout lacks fault_observed=true")
        if rank == 0:
            if observations.get("local_enqueued") is not True:
                issues.append(f"{source}: fault rank must be locally enqueued")
            if observations.get("local_enqueue_role") != "submit_fault":
                issues.append(f"{source}: fault rank must have submit_fault role")
            if (
                not isinstance(details, Mapping)
                or details.get("work_handle_present") is not True
            ):
                issues.append(f"{source}: fault rank must retain a Work handle")
        else:
            if observations.get("local_enqueued") is not False:
                issues.append(f"{source}: non-fault rank must not enqueue locally")
            if observations.get("local_enqueue_role") != "skip_fault":
                issues.append(f"{source}: non-fault rank must have skip_fault role")
    elif classification == "capability_unavailable":
        if litmus_id == "collective_clean_destroy_recreate":
            require_final("destroyed", ("retiring", "destroyed"))
        elif litmus_id == "collective_partial_epoch_timeout_recreate":
            require_final("destroyed", ("failed_unknown", "aborting", "destroyed"))
    elif classification == "inconclusive":
        if result.final_state != "enqueued" or any(
            phase in targets for phase in ("failed_unknown", "destroyed", "recreated")
        ):
            issues.append(f"{source}: inconclusive has the wrong final_state/phase")
    elif classification == "reinitialization_capability_failure":
        if generation == 0 and len(shape) == 2:
            require_final("recreated", ("destroyed", "recreated"))
        elif generation == 0 and litmus_id == "collective_clean_destroy_recreate":
            if result.final_state not in {"retiring", "destroyed"}:
                issues.append(f"{source}: clean capability failure has wrong phase")
        elif generation == 0:
            if result.final_state not in {"aborting", "destroyed"}:
                issues.append(f"{source}: partial capability failure has wrong phase")
        elif result.final_state not in {
            "epoch_open",
            "enqueued",
            "completed",
            "reusable_same_generation",
        }:
            issues.append(f"{source}: generation-1 capability failure has wrong phase")


def adjudicate_evidence(
    rank_rows: Mapping[int, Sequence[Mapping[str, Any]]],
    markers: Sequence[Mapping[str, Any]] = (),
    launcher: Mapping[str, Any] | None = None,
    *,
    rank_file_presence: Mapping[int, bool] | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    classifications: list[str] = []
    identity: dict[str, set[Any]] = {
        "protocol_id": set(),
        "run_id": set(),
        "litmus_id": set(),
        "world_size": set(),
        "schedule_digest": set(),
    }
    generations: dict[int, list[int]] = {rank: [] for rank in range(WORLD_SIZE)}
    row_classifications: dict[tuple[int, int], str] = {}
    row_observations: dict[tuple[int, int], Mapping[str, Any]] = {}
    validated_results: dict[tuple[int, int], LitmusResult] = {}

    if rank_file_presence is None:
        rank_file_presence = {rank: rank in rank_rows for rank in range(WORLD_SIZE)}

    if set(rank_rows) != set(range(WORLD_SIZE)):
        issues.append("rank evidence must cover exactly ranks 0 and 1")

    def record_identity(
        record: Mapping[str, Any], source: str, *, require_protocol: bool = True
    ) -> None:
        for key in identity:
            required = require_protocol or key not in {"protocol_id", "world_size"}
            value = _field(record, key, source=source, issues=issues, required=required)
            if value is not None:
                try:
                    identity[key].add(value)
                except TypeError:
                    issues.append(f"{source}: {key} must be a scalar")

    for input_rank in range(WORLD_SIZE):
        rows = rank_rows.get(input_rank, ())
        for index, row in enumerate(rows):
            source = f"rank {input_rank} row {index}"
            if not isinstance(row, Mapping):
                issues.append(f"{source}: row must be an object")
                continue
            try:
                parsed_result = LitmusResult.from_dict(row)
            except (KeyError, TypeError, ValueError) as error:
                issues.append(
                    f"{source}: invalid LitmusResult: {type(error).__name__}: {error}"
                )
                parsed_result = None
            record_identity(row, source)
            rank = _field(row, "rank", source=source, issues=issues)
            generation = _field(row, "generation", source=source, issues=issues)
            classification = _field(row, "classification", source=source, issues=issues)
            if not isinstance(rank, int) or isinstance(rank, bool):
                issues.append(f"{source}: rank must be an integer")
            elif rank != input_rank:
                issues.append(f"{source}: row rank does not match its JSONL")
            if not isinstance(generation, int) or isinstance(generation, bool):
                issues.append(f"{source}: generation must be an integer")
            elif generation not in {0, 1}:
                issues.append(f"{source}: generation must be 0 or 1")
            else:
                generations[input_rank].append(generation)
            if not isinstance(classification, str):
                issues.append(f"{source}: classification must be a string")
            elif classification not in KNOWN_CLASSIFICATIONS:
                issues.append(f"{source}: unknown classification {classification}")
            else:
                classifications.append(classification)
                if isinstance(generation, int):
                    row_classifications[(input_rank, generation)] = classification
                    if parsed_result is not None:
                        validated_results[(input_rank, generation)] = parsed_result
                    observations = row.get("observations")
                    if isinstance(observations, Mapping):
                        row_observations[(input_rank, generation)] = observations
                if classification == "capability_timeout":
                    issues.append(
                        f"{source}: capability_timeout must be run-level evidence"
                    )
                passed = row.get("passed")
                error = row.get("error")
                if classification in PASS_CLASSIFICATIONS or classification == (
                    "capability_unavailable"
                ):
                    if passed is not True or error is not None:
                        issues.append(
                            f"{source}: {classification} requires "
                            "passed=true/error=null"
                        )
                elif classification in FAIL_CLASSIFICATIONS:
                    if passed is not False or not isinstance(error, str) or not error:
                        issues.append(
                            f"{source}: {classification} requires passed=false/error"
                        )

    marker_phase_ranks: dict[str, set[int]] = {}
    marker_keys: set[tuple[str, int]] = set()
    marker_records: dict[tuple[str, int], Mapping[str, Any]] = {}
    for index, marker in enumerate(markers):
        source = f"marker {index}"
        if not isinstance(marker, Mapping):
            issues.append(f"{source}: marker must be an object")
            continue
        record_identity(marker, source)
        phase = _field(marker, "phase", source=source, issues=issues)
        rank = _field(marker, "rank", source=source, issues=issues)
        if not isinstance(phase, str) or not phase:
            issues.append(f"{source}: phase must be a non-empty string")
            continue
        if not isinstance(rank, int) or isinstance(rank, bool) or rank not in {0, 1}:
            issues.append(f"{source}: rank must be 0 or 1")
            continue
        key = (phase, rank)
        if key in marker_keys:
            issues.append(f"{source}: duplicate marker for {phase}/rank {rank}")
        marker_keys.add(key)
        marker_records[key] = marker
        marker_phase_ranks.setdefault(phase, set()).add(rank)

    if launcher is not None and not isinstance(launcher, Mapping):
        issues.append("launcher evidence must be an object")
        launcher = None
    timed_out = _launcher_bool(launcher, "timed_out", "timeout", issues)
    schedule_complete = _launcher_bool(
        launcher, "schedule_complete", "complete_schedule", issues
    )
    final_phase: Any = None
    status_ranks: set[int] = set()
    status_classifications: list[str] = []
    status_records: dict[int, Mapping[str, Any]] = {}
    if launcher is not None:
        record_identity(launcher, "launcher")
        final_phase = launcher.get("final_phase", launcher.get("last_valid_phase"))
        statuses = launcher.get("rank_statuses", [])
        if not isinstance(statuses, list):
            issues.append("launcher: rank_statuses must be a list")
            statuses = []
        for index, status in enumerate(statuses):
            source = f"launcher rank status {index}"
            if not isinstance(status, Mapping):
                issues.append(f"{source}: status must be an object")
                continue
            record_identity(status, source, require_protocol=False)
            rank = _field(status, "rank", source=source, issues=issues)
            classification = _field(
                status, "classification", source=source, issues=issues
            )
            if (
                not isinstance(rank, int)
                or isinstance(rank, bool)
                or rank not in {0, 1}
            ):
                issues.append(f"{source}: rank must be 0 or 1")
            elif rank in status_ranks:
                issues.append(f"{source}: duplicate rank status")
            else:
                status_ranks.add(rank)
                status_records[rank] = status
            if classification not in KNOWN_CLASSIFICATIONS:
                issues.append(f"{source}: unknown classification {classification}")
            else:
                status_classifications.append(str(classification))

    exit_codes = _rank_exit_codes(launcher)
    nonzero_exit_without_status: list[int] = []
    for rank, exit_code in exit_codes.items():
        if rank not in range(WORLD_SIZE):
            issues.append(f"launcher: unexpected rank exit code for rank {rank}")
            continue
        if exit_code is None:
            continue
        try:
            nonzero = int(exit_code) != 0
        except (TypeError, ValueError):
            issues.append(f"launcher: invalid exit code for rank {rank}")
            continue
        if nonzero and rank not in status_ranks:
            nonzero_exit_without_status.append(rank)
    launcher_exit_value: int | None = None
    launcher_exit_missing_status = False
    if launcher is not None and not timed_out:
        launcher_exit_code = launcher.get("launcher_exit_code")
        if launcher_exit_code is not None:
            try:
                launcher_exit_value = int(launcher_exit_code)
            except (TypeError, ValueError):
                issues.append("launcher: launcher_exit_code must be an integer or null")
            else:
                launcher_exit_missing_status = (
                    launcher_exit_value != 0 and status_ranks != set(range(WORLD_SIZE))
                )
    classifications.extend(status_classifications)

    for key, values in identity.items():
        if not values:
            issues.append(f"evidence has no {key}")
        elif len(values) != 1:
            issues.append(f"evidence disagrees on {key}")
    protocol_id = next(iter(identity["protocol_id"]), None)
    run_id = next(iter(identity["run_id"]), None)
    litmus_id = next(iter(identity["litmus_id"]), None)
    world_size = next(iter(identity["world_size"]), None)
    schedule_digest = next(iter(identity["schedule_digest"]), None)
    if protocol_id is not None and protocol_id != PROTOCOL_ID:
        issues.append(f"unexpected protocol_id {protocol_id}")
    if litmus_id is not None and litmus_id not in LITMUS:
        issues.append(f"unexpected litmus_id {litmus_id}")
    if world_size is not None and world_size != WORLD_SIZE:
        issues.append(f"world_size must be {WORLD_SIZE}")
    if not isinstance(run_id, str) or not run_id:
        issues.append("run_id must be a non-empty string")
    if not isinstance(schedule_digest, str) or not schedule_digest:
        issues.append("schedule_digest must be a non-empty string")

    for rank, values in generations.items():
        if values != sorted(values) or len(values) != len(set(values)):
            issues.append(f"rank {rank}: generations must be unique and ordered")

    exit_ranks = set(exit_codes)
    marker_rank_evidence_complete = (
        isinstance(final_phase, str)
        and marker_phase_ranks.get(final_phase) == set(range(WORLD_SIZE))
    ) or (
        litmus_id == "collective_partial_epoch_timeout_recreate"
        and final_phase == "device_work"
        and marker_phase_ranks.get("fault_ready") == set(range(WORLD_SIZE))
    )
    launcher_rank_evidence_complete = (
        status_ranks == set(range(WORLD_SIZE))
        or exit_ranks == set(range(WORLD_SIZE))
        or marker_rank_evidence_complete
    )
    timeout_complete = (
        timed_out
        and schedule_complete
        and isinstance(final_phase, str)
        and final_phase in KNOWN_TIMEOUT_PHASES
        and launcher_rank_evidence_complete
    )
    if timed_out and not timeout_complete:
        issues.append(
            "launcher timeout does not establish a complete schedule, final phase, "
            "and both ranks"
        )

    total_rows = sum(len(rank_rows.get(rank, ())) for rank in range(WORLD_SIZE))
    shapes = {
        rank: tuple(
            (generation, row_classifications[(rank, generation)])
            for generation in generations[rank]
            if (rank, generation) in row_classifications
        )
        for rank in range(WORLD_SIZE)
    }
    allowed_shapes = (
        _allowed_row_shapes(str(litmus_id)) if isinstance(litmus_id, str) else set()
    )

    runtime_failure = (
        launcher.get("runtime_failure_evidence")
        if isinstance(launcher, Mapping)
        else None
    )
    launcher_rank_jsonl = (
        launcher.get("rank_jsonl") if isinstance(launcher, Mapping) else None
    )
    narrow_backend_fatal = (
        not issues
        and isinstance(launcher, Mapping)
        and launcher.get("platform") == "910b"
        and isinstance(launcher.get("source_revision"), str)
        and bool(launcher.get("source_revision"))
        and litmus_id == "collective_partial_epoch_timeout_recreate"
        and timed_out is False
        and launcher.get("signals_sent") == []
        and launcher_exit_value is not None
        and launcher_exit_value != 0
        and schedule_complete is False
        and final_phase == "ready_destroy"
        and launcher.get("evidence_issues") == []
        and exit_codes == {0: -15, 1: 4}
        and isinstance(runtime_failure, Mapping)
        and runtime_failure.get("torchrun_root_cause") == {"rank": 1, "exit_code": 4}
        and runtime_failure.get("backend_watchdog_timeout_ranks") == [0]
        and runtime_failure.get("backend_fatal_termination_ranks") == [0]
        and marker_phase_ranks
        == {
            "fault_ready": {0, 1},
            "fault_observed": {0, 1},
            "ready_destroy": {0, 1},
            "destroyed": {1},
        }
        and "recreate_start" not in marker_phase_ranks
        and rank_file_presence.get(0) is False
        and rank_file_presence.get(1) is True
        and len(rank_rows.get(0, ())) == 0
        and len(rank_rows.get(1, ())) == 1
        and shapes[0] == ()
        and shapes[1] == ((0, "harness_error"),)
        and status_ranks == {1}
        and status_classifications == ["harness_error"]
        and isinstance(launcher_rank_jsonl, Mapping)
        and isinstance(launcher_rank_jsonl.get("0"), Mapping)
        and launcher_rank_jsonl["0"].get("exists") is False
        and isinstance(launcher_rank_jsonl.get("1"), Mapping)
        and launcher_rank_jsonl["1"].get("exists") is True
    )
    if narrow_backend_fatal:
        narrow_issues: list[str] = []
        result = validated_results.get((1, 0))
        observations = row_observations.get((1, 0))
        status = status_records.get(1)
        expected_targets = (
            "enqueued",
            "completed",
            "reusable_same_generation",
            "epoch_open",
            "enqueued",
            "failed_unknown",
            "aborting",
            "destroyed",
        )
        if result is None or observations is None or status is None:
            narrow_issues.append("secondary rank evidence is incomplete")
        else:
            targets = tuple(event.target for event in result.transitions)
            if result.final_state != "destroyed" or targets != expected_targets:
                narrow_issues.append("secondary rank row has the wrong lifecycle")
            if observations.get("normalized_error_type") != "MarkerError":
                narrow_issues.append("secondary rank row is not a MarkerError")
            if observations.get("local_enqueued") is not False:
                narrow_issues.append("secondary rank must not enqueue the fault")
            if observations.get("local_enqueue_role") != "skip_fault":
                narrow_issues.append("secondary rank must have skip_fault role")
            fault_details = observations.get("fault_details")
            if (
                not isinstance(fault_details, Mapping)
                or fault_details.get("fault_observed") is not True
                or fault_details.get("work_handle_present") is not True
            ):
                narrow_issues.append("secondary row lacks rank-0 fault evidence")
            if (
                result.backend != "hccl"
                or observations.get("collective_backend") != "hccl"
            ):
                narrow_issues.append("secondary row backend must be hccl")
            if observations.get("source_revision") != launcher.get("source_revision"):
                narrow_issues.append(
                    "secondary row source_revision mismatches launcher"
                )
            if observations.get("tensor_shape") != [4]:
                narrow_issues.append("secondary row tensor_shape must equal [4]")
            if observations.get("dtype") != "int64":
                narrow_issues.append("secondary row dtype must equal int64")
            if observations.get("reduce_op") != "sum":
                narrow_issues.append("secondary row reduce_op must equal sum")
            _validate_success_observations(
                observations,
                source="rank 1 g0",
                generation=0,
                expected=(0, 1, 1),
                issues=narrow_issues,
            )
            if status.get("classification") != "harness_error":
                narrow_issues.append("secondary rank status must be harness_error")

        fault_rank_zero = marker_records.get(("fault_observed", 0))
        fault_rank_one = marker_records.get(("fault_observed", 1))
        if (
            fault_rank_zero is None
            or fault_rank_zero.get("fault_observed") is not True
            or fault_rank_zero.get("local_enqueued") is not True
            or fault_rank_zero.get("work_handle_present") is not True
        ):
            narrow_issues.append("rank-0 fault marker is incomplete")
        if (
            fault_rank_one is None
            or fault_rank_one.get("fault_observed") is not True
            or fault_rank_one.get("local_enqueued") is not False
            or fault_rank_one.get("rank_0_work_handle_present") is not True
        ):
            narrow_issues.append("rank-1 fault marker is incomplete")

        if not narrow_issues:
            return {
                "schema": SUMMARY_SCHEMA,
                "protocol_id": protocol_id,
                "run_id": run_id,
                "litmus_id": litmus_id,
                "world_size": world_size,
                "schedule_digest": schedule_digest,
                "rank_row_counts": {"0": 0, "1": 1},
                "rank_generations": {"0": [], "1": [0]},
                "rank_jsonl_present": {"0": False, "1": True},
                "marker_phases": {
                    phase: sorted(ranks)
                    for phase, ranks in sorted(marker_phase_ranks.items())
                },
                "launcher_timed_out": False,
                "schedule_complete": False,
                "final_phase": "ready_destroy",
                "classifications": {"reinitialization_capability_failure": 1},
                "aggregate_classification": ("reinitialization_capability_failure"),
                "evidence_valid": True,
                "issues": [],
                "exit_code": 5,
            }

    if launcher is not None and not timed_out and not schedule_complete:
        issues.append("non-timeout launcher evidence requires schedule_complete=true")
    if not timed_out:
        for rank in nonzero_exit_without_status:
            issues.append(f"launcher: nonzero rank {rank} exit lacks rank_status")
        if launcher_exit_missing_status:
            issues.append(
                "launcher: nonzero launcher_exit_code requires both rank_statuses"
            )

    if not timeout_complete:
        if total_rows == 0:
            if status_ranks != set(range(WORLD_SIZE)):
                issues.append("zero-row run requires statuses for both ranks")
            elif len(set(status_classifications)) != 1:
                issues.append("zero-row rank classifications disagree")
            elif status_classifications[0] not in {
                "capability_unavailable",
                "harness_error",
            }:
                issues.append(
                    "zero-row run has an invalid initial-group classification"
                )
        else:
            if any(
                rank_file_presence.get(rank) is not True for rank in range(WORLD_SIZE)
            ):
                issues.append("non-timeout row evidence requires both rank JSONL files")
            if shapes[0] != shapes[1]:
                issues.append("rank row shapes or classifications disagree")
            elif shapes[0] not in allowed_shapes:
                issues.append("rows do not match the frozen phase-to-row table")
    else:
        classifications.append("capability_timeout")
        for rank, shape in shapes.items():
            if shape and not any(
                candidate[: len(shape)] == shape for candidate in allowed_shapes
            ):
                issues.append(
                    f"rank {rank}: timeout rows violate the frozen phase-to-row table"
                )

    for (rank, generation), classification in row_classifications.items():
        expectation = _successful_epoch_contract(
            str(litmus_id), generation, classification
        )
        observations = row_observations.get((rank, generation))
        result = validated_results.get((rank, generation))
        if result is not None and observations is not None:
            _validate_result_phase(
                result,
                observations,
                litmus_id=str(litmus_id),
                generation=generation,
                classification=classification,
                rank=rank,
                shape=shapes[rank],
                issues=issues,
            )
        if expectation is not None:
            if observations is None:
                issues.append(f"rank {rank} g{generation}: observations are missing")
            else:
                _validate_success_observations(
                    observations,
                    source=f"rank {rank} g{generation}",
                    generation=generation,
                    expected=expectation,
                    issues=issues,
                )

    backend_values: set[str] = set()
    revision_values: set[str] = set()
    for (rank, generation), observations in row_observations.items():
        source = f"rank {rank} g{generation}"
        backend = observations.get("collective_backend")
        revision = observations.get("source_revision")
        if not isinstance(backend, str) or not backend:
            issues.append(f"{source}: collective_backend must be non-empty")
        else:
            backend_values.add(backend)
        if not isinstance(revision, str) or not revision:
            issues.append(f"{source}: source_revision must be non-empty")
        else:
            revision_values.add(revision)
        if observations.get("tensor_shape") != [4]:
            issues.append(f"{source}: tensor_shape must equal [4]")
        if observations.get("dtype") != "int64":
            issues.append(f"{source}: dtype must equal int64")
        if observations.get("reduce_op") != "sum":
            issues.append(f"{source}: reduce_op must equal sum")
    if len(backend_values) > 1:
        issues.append("ranks disagree on collective_backend")
    if len(revision_values) > 1:
        issues.append("ranks disagree on source_revision")

    for generation in {0, 1}:
        successful = [
            row_observations[(rank, generation)]
            for rank in range(WORLD_SIZE)
            if (rank, generation) in row_observations
            and _successful_epoch_contract(
                str(litmus_id),
                generation,
                row_classifications[(rank, generation)],
            )
            is not None
        ]
        for digest_key in ("expected_payload_digest", "observed_payload_digest"):
            values = {row.get(digest_key) for row in successful}
            if len(successful) == WORLD_SIZE and len(values) != 1:
                issues.append(
                    f"generation {generation}: ranks disagree on {digest_key}"
                )

    for generation in {0, 1}:
        per_rank = {
            row_classifications[(rank, generation)]
            for rank in range(WORLD_SIZE)
            if (rank, generation) in row_classifications
        }
        if len(per_rank) > 1:
            issues.append(f"rank classifications disagree for generation {generation}")

    for phase, ranks in marker_phase_ranks.items():
        if ranks != set(range(WORLD_SIZE)):
            issues.append(f"marker phase {phase} is missing a rank")

    if litmus_id in MARKER_PHASES:
        expected = MARKER_PHASES[str(litmus_id)]
        observed = set(marker_phase_ranks)
        if not observed.issubset(set(expected)):
            issues.append("marker set contains a phase outside the frozen schedule")
        prefix = set(expected[: len(observed)])
        if observed != prefix:
            issues.append("marker phases are not a frozen-schedule prefix")
        has_generation_one = any(1 in values for values in generations.values())
        if has_generation_one and observed != set(expected):
            issues.append("generation 1 evidence requires every frozen marker phase")
        if (
            not timeout_complete
            and classifications
            and any(value in CAPABILITY_CLASSIFICATIONS for value in classifications)
            and sum(len(values) for values in generations.values()) > 0
            and not observed
        ):
            issues.append("capability result is missing out-of-band markers")
        if (
            litmus_id == "collective_partial_epoch_timeout_recreate"
            and shapes[0] == ((0, "inconclusive"),)
            and shapes[1] == shapes[0]
        ):
            required = {"fault_ready", "fault_observed"}
            if observed != required:
                issues.append(
                    "partial inconclusive requires fault_ready/fault_observed markers"
                )
            for rank in range(WORLD_SIZE):
                fault_marker = marker_records.get(("fault_observed", rank))
                if (
                    fault_marker is None
                    or fault_marker.get("fault_observed") is not False
                ):
                    issues.append(
                        "partial inconclusive requires fault_observed=false "
                        f"for rank {rank}"
                    )
        recovered_shape = (
            (0, "expected_timeout_recovered"),
            (1, "capability_pass"),
        )
        if (
            litmus_id == "collective_partial_epoch_timeout_recreate"
            and shapes[0] == recovered_shape
            and shapes[1] == recovered_shape
        ):
            for rank in range(WORLD_SIZE):
                fault_marker = marker_records.get(("fault_observed", rank))
                if (
                    fault_marker is None
                    or fault_marker.get("fault_observed") is not True
                ):
                    issues.append(
                        "expected_timeout_recovered requires fault_observed=true "
                        f"for rank {rank} marker"
                    )
                    continue
                expected_local = rank == 0
                if fault_marker.get("local_enqueued") is not expected_local:
                    issues.append(
                        "fault_observed marker has wrong local_enqueued "
                        f"for rank {rank}"
                    )
                if rank == 0 and fault_marker.get("work_handle_present") is not True:
                    issues.append("rank 0 fault marker must retain a Work handle")
                if (
                    rank == 1
                    and fault_marker.get("rank_0_work_handle_present") is not True
                ):
                    issues.append("rank 1 fault marker must acknowledge rank 0 Work")
    elif marker_phase_ranks:
        issues.append("non-recreate litmus must not publish recreate markers")

    if timeout_complete and isinstance(litmus_id, str):
        marker_schedule = MARKER_PHASES.get(litmus_id, ())
        if final_phase in marker_schedule:
            if marker_phase_ranks.get(str(final_phase)) != set(range(WORLD_SIZE)):
                issues.append("launcher final marker phase is absent or missing a rank")

    if issues:
        classifications.append("harness_error")
    if not classifications:
        classifications.append("harness_error")
        issues.append("evidence contains no adjudicable classification")

    aggregate, exit_code = aggregate_classifications(classifications)
    return {
        "schema": SUMMARY_SCHEMA,
        "protocol_id": protocol_id,
        "run_id": run_id,
        "litmus_id": litmus_id,
        "world_size": world_size,
        "schedule_digest": schedule_digest,
        "rank_row_counts": {
            str(rank): len(rank_rows.get(rank, ())) for rank in range(WORLD_SIZE)
        },
        "rank_generations": {
            str(rank): list(generations[rank]) for rank in range(WORLD_SIZE)
        },
        "rank_jsonl_present": {
            str(rank): bool(rank_file_presence.get(rank)) for rank in range(WORLD_SIZE)
        },
        "marker_phases": {
            phase: sorted(ranks) for phase, ranks in sorted(marker_phase_ranks.items())
        },
        "launcher_timed_out": timed_out,
        "schedule_complete": schedule_complete,
        "final_phase": final_phase,
        "classifications": dict(sorted(Counter(classifications).items())),
        "aggregate_classification": aggregate,
        "evidence_valid": not issues,
        "issues": issues,
        "exit_code": exit_code,
    }


def harness_summary(message: str) -> dict[str, Any]:
    return {
        "schema": SUMMARY_SCHEMA,
        "protocol_id": None,
        "run_id": None,
        "litmus_id": None,
        "world_size": None,
        "schedule_digest": None,
        "rank_row_counts": {"0": 0, "1": 0},
        "rank_generations": {"0": [], "1": []},
        "rank_jsonl_present": {"0": False, "1": False},
        "marker_phases": {},
        "launcher_timed_out": False,
        "schedule_complete": False,
        "final_phase": None,
        "classifications": {"harness_error": 1},
        "aggregate_classification": "harness_error",
        "evidence_valid": False,
        "issues": [message],
        "exit_code": 4,
    }


def emit_summary(summary: Mapping[str, Any], output: Path | None = None) -> None:
    encoded = json.dumps(summary, sort_keys=True, ensure_ascii=False, allow_nan=False)
    print(encoded)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    try:
        launcher = (
            read_json(args.launcher_evidence)
            if args.launcher_evidence is not None
            else None
        )
        rank_rows, rank_file_presence = load_rank_rows(
            args.rank_jsonl, allow_missing=launcher is not None
        )
        markers = load_markers(args.marker)
        summary = adjudicate_evidence(
            rank_rows,
            markers,
            launcher,
            rank_file_presence=rank_file_presence,
        )
    except (EvidenceError, OSError) as error:
        summary = harness_summary(str(error))
    emit_summary(summary, args.output)
    return int(summary["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
