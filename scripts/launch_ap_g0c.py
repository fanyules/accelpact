#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import signal
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "AP-G0C"
CONFIG_SCHEMA = "accelpact.ap_g0c.config.v1"
RANK_STATUS_SCHEMA = "accelpact.ap_g0c.rank_status.v1"
LAUNCHER_SCHEMA = "accelpact.ap_g0c.launcher_evidence.v1"
MANIFEST_SCHEMA = "accelpact.ap_g0c.sha256_manifest.v1"
WORLD_SIZE = 2
SIGTERM = getattr(signal, "SIGTERM", 15)
SIGKILL = getattr(signal, "SIGKILL", 9)

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch one frozen AP-G0C Linux TP2 process pair"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "ap_g0c.json",
    )
    parser.add_argument("--platform", choices=("a100", "910b"), required=True)
    parser.add_argument("--litmus", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--torchrun", type=Path, required=True)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--termination-grace-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    if not args.run_id.strip():
        parser.error("--run-id must be non-empty")
    if not args.source_revision.strip():
        parser.error("--source-revision must be non-empty")
    if args.repetition <= 0:
        parser.error("--repetition must be positive")
    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be in [1, 65535]")
    if args.termination_grace_seconds <= 0:
        parser.error("--termination-grace-seconds must be positive")
    return args


def _require_linux() -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("AP-G0C launcher supports Linux only")


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected AP-G0C config schema")
    if payload.get("world_size") != WORLD_SIZE:
        raise ValueError("AP-G0C launcher requires world_size=2")
    if payload.get("outer_process_timeout_seconds") != 180:
        raise ValueError("AP-G0C launcher requires the frozen 180-second timeout")
    return payload


def _write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")


def _write_text_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def schedule_digest(config: dict[str, Any], args: argparse.Namespace) -> str:
    return _canonical_digest(
        {
            "config": config,
            "litmus_id": args.litmus,
            "platform": args.platform,
            "protocol_id": PROTOCOL_ID,
            "repetition": args.repetition,
            "run_id": args.run_id,
            "source_revision": args.source_revision,
        }
    )


def platform_environment(
    config: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, str], dict[str, str]]:
    platform = config.get("platforms", {}).get(args.platform)
    if not isinstance(platform, dict):
        raise ValueError(f"platform {args.platform} is not configured")
    overrides = {
        str(name): str(value)
        for name, value in platform.get("launch_environment", {}).items()
    }
    devices = platform.get("devices")
    if devices != [0, 1]:
        raise ValueError("AP-G0C launcher requires frozen logical devices [0, 1]")
    device_type = platform.get("device_type")
    if device_type == "cuda":
        overrides["CUDA_VISIBLE_DEVICES"] = "0,1"
    elif device_type == "npu":
        overrides["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
    else:
        raise ValueError(f"unsupported AP-G0C device type: {device_type}")
    overrides.update(
        {
            "MASTER_ADDR": args.master_addr,
            "MASTER_PORT": str(args.master_port),
            "PYTHONUNBUFFERED": "1",
        }
    )
    environment = os.environ.copy()
    environment.update(overrides)
    return environment, overrides


def _structured_error(stage: str, error: BaseException) -> dict[str, str]:
    return {
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _version_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)


def inspect_runtime_environment(
    config: dict[str, Any],
    args: argparse.Namespace,
    environment_overrides: dict[str, str],
    *,
    module_loader: Any = importlib.import_module,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    report: dict[str, Any] = {
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "torch": None,
        "torch_npu": None,
        "cuda": None,
        "nccl": None,
        "backend": None,
        "visible_devices": {"count": None, "names": []},
    }
    platform_config = config["platforms"][args.platform]
    visibility_names = ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")
    previous_visibility = {
        name: os.environ.get(name) for name in visibility_names if name in os.environ
    }
    try:
        for name in visibility_names:
            if name in environment_overrides:
                os.environ[name] = environment_overrides[name]
        torch = module_loader("torch")
        report["torch"] = _version_value(getattr(torch, "__version__", None))
        if platform_config["device_type"] == "npu":
            torch_npu = module_loader("torch_npu")
            report["torch_npu"] = _version_value(
                getattr(torch_npu, "__version__", None)
            )
        dist = module_loader("torch.distributed")
        device_type = str(platform_config["device_type"])
        device_api = getattr(torch, device_type)
        available = bool(device_api.is_available())
        device_count = int(device_api.device_count()) if available else 0
        names = [
            str(device_api.get_device_name(index)) for index in range(device_count)
        ]
        backend = str(platform_config["device_group_backend"])
        backend_available = bool(dist.is_backend_available(backend))
        report["backend"] = {
            "name": backend,
            "available": backend_available,
            "device_api_available": available,
        }
        report["visible_devices"] = {"count": device_count, "names": names}
        if device_type == "cuda":
            report["cuda"] = _version_value(
                getattr(getattr(torch, "version", None), "cuda", None)
            )
            cuda_api = getattr(torch, "cuda", None)
            nccl = getattr(cuda_api, "nccl", None) if cuda_api is not None else None
            nccl_version = getattr(nccl, "version", None)
            report["nccl"] = _version_value(
                nccl_version() if callable(nccl_version) else None
            )
        if not available:
            raise RuntimeError(f"torch.{device_type} is unavailable")
        if device_count != WORLD_SIZE:
            raise RuntimeError(
                f"expected exactly {WORLD_SIZE} visible devices, got {device_count}"
            )
        if not backend_available:
            raise RuntimeError(f"distributed backend {backend} is unavailable")
    except Exception as error:  # noqa: BLE001 - persist a structured preflight result
        structured = _structured_error("runtime_preflight", error)
        report["launch_error"] = structured
        return report, structured
    finally:
        for name in visibility_names:
            if name in previous_visibility:
                os.environ[name] = str(previous_visibility[name])
            elif name in os.environ and name in environment_overrides:
                del os.environ[name]
    return report, None


def build_torchrun_command(
    config: dict[str, Any], args: argparse.Namespace, torchrun_log_dir: Path
) -> list[str]:
    if not args.torchrun.is_absolute():
        raise ValueError("--torchrun must be an absolute path")
    if not args.torchrun.is_file():
        raise FileNotFoundError(f"torchrun is missing: {args.torchrun}")
    if args.litmus not in config.get("litmus_order", []):
        raise ValueError(f"unknown frozen AP-G0C litmus: {args.litmus}")
    return [
        str(args.torchrun),
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        "--max-restarts=0",
        f"--master-addr={args.master_addr}",
        f"--master-port={args.master_port}",
        f"--log-dir={torchrun_log_dir}",
        "--tee=3",
        str(REPOSITORY_ROOT / "scripts" / "run_ap_g0c.py"),
        "--config",
        str(args.config.resolve()),
        "--platform",
        args.platform,
        "--litmus",
        args.litmus,
        "--run-id",
        args.run_id,
        "--output-dir",
        str(args.run_dir.resolve()),
        "--repetition",
        str(args.repetition),
        "--source-revision",
        args.source_revision,
    ]


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _terminate_process_group(
    process: Any, *, termination_grace_seconds: float
) -> tuple[str, list[str]]:
    signals_sent: list[str] = []
    cleanup_errors: list[BaseException] = []

    def send_group_signal(name: str, signum: int) -> None:
        signals_sent.append(name)
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass
        except BaseException as error:
            cleanup_errors.append(error)

    send_group_signal("SIGTERM", SIGTERM)
    output: str | bytes | None = None
    try:
        output, _ = process.communicate(timeout=termination_grace_seconds)
    except subprocess.TimeoutExpired:
        send_group_signal("SIGKILL", SIGKILL)
        try:
            output, _ = process.communicate()
        except BaseException as error:
            cleanup_errors.append(error)
            try:
                process.wait()
            except BaseException as wait_error:
                cleanup_errors.append(wait_error)
    except BaseException as error:
        cleanup_errors.append(error)
        send_group_signal("SIGKILL", SIGKILL)
        try:
            output, _ = process.communicate()
        except BaseException as final_error:
            cleanup_errors.append(final_error)
            try:
                process.wait()
            except BaseException as wait_error:
                cleanup_errors.append(wait_error)

    if cleanup_errors:
        raise cleanup_errors[0]
    return _as_text(output), signals_sent


def _sigterm_as_exit(signum: int, _frame: Any) -> None:
    raise SystemExit(128 + signum)


def supervise(
    command: Sequence[str],
    environment: dict[str, str],
    *,
    timeout_seconds: float,
    termination_grace_seconds: float,
) -> tuple[str, int | None, bool, list[str]]:
    previous_sigterm = signal.signal(SIGTERM, _sigterm_as_exit)
    process = None
    cleanup_started = False
    try:
        process = subprocess.Popen(
            list(command),
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            cleanup_started = True
            output, signals_sent = _terminate_process_group(
                process,
                termination_grace_seconds=termination_grace_seconds,
            )
            return output, process.returncode, True, signals_sent
        return _as_text(output), process.returncode, False, []
    except BaseException:
        if process is not None and not cleanup_started:
            cleanup_started = True
            try:
                _terminate_process_group(
                    process,
                    termination_grace_seconds=termination_grace_seconds,
                )
            except BaseException:
                # The supervise failure remains authoritative after best-effort reap.
                pass
        raise
    finally:
        signal.signal(SIGTERM, previous_sigterm)


def _json_from_line(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    candidates = [stripped]
    if "{" in stripped and "}" in stripped:
        candidates.append(stripped[stripped.find("{") : stripped.rfind("}") + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def collect_rank_statuses(
    combined_output: str,
    torchrun_log_dir: Path,
    *,
    run_id: str,
    litmus_id: str,
    expected_schedule_digest: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    texts = [("combined.log", combined_output)]
    if torchrun_log_dir.is_dir():
        for path in sorted(torchrun_log_dir.rglob("*")):
            if path.is_file():
                try:
                    texts.append(
                        (
                            str(path),
                            path.read_text(encoding="utf-8", errors="replace"),
                        )
                    )
                except OSError:
                    continue

    statuses: list[dict[str, Any]] = []
    seen: set[str] = set()
    issues: list[str] = []
    for source, text in texts:
        for line_number, line in enumerate(text.splitlines(), start=1):
            payload = _json_from_line(line)
            if payload is None or payload.get("schema") != RANK_STATUS_SCHEMA:
                continue
            encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            if encoded in seen:
                continue
            seen.add(encoded)
            expected = {
                "litmus_id": litmus_id,
                "protocol_id": PROTOCOL_ID,
                "run_id": run_id,
                "schedule_digest": expected_schedule_digest,
                "world_size": WORLD_SIZE,
            }
            if any(payload.get(key) != value for key, value in expected.items()):
                issues.append(f"{source}:{line_number}: mismatched rank status")
                continue
            rank = payload.get("rank")
            if (
                not isinstance(rank, int)
                or isinstance(rank, bool)
                or rank not in {0, 1}
            ):
                issues.append(f"{source}:{line_number}: invalid rank status rank")
                continue
            statuses.append(payload)
    statuses.sort(key=lambda row: (int(row["rank"]), str(row.get("classification"))))
    return statuses, issues


def inspect_marker_prefix(
    run_dir: Path,
    *,
    run_id: str,
    litmus_id: str,
    expected_schedule_digest: str,
) -> tuple[list[str], dict[str, list[int]], list[str]]:
    expected_phases = MARKER_PHASES.get(litmus_id, ())
    prefix: list[str] = []
    phase_ranks: dict[str, list[int]] = {}
    issues: list[str] = []
    prefix_open = True
    for phase in expected_phases:
        valid_ranks: list[int] = []
        for rank in range(WORLD_SIZE):
            path = run_dir / "control" / phase / f"rank-{rank:04d}.json"
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                issues.append(f"invalid marker: {path}")
                continue
            expected = {
                "litmus_id": litmus_id,
                "phase": phase,
                "protocol_id": PROTOCOL_ID,
                "rank": rank,
                "run_id": run_id,
                "schedule_digest": expected_schedule_digest,
                "world_size": WORLD_SIZE,
            }
            if not isinstance(payload, dict) or any(
                payload.get(key) != value for key, value in expected.items()
            ):
                issues.append(f"stale or mismatched marker: {path}")
                continue
            valid_ranks.append(rank)
        if valid_ranks:
            phase_ranks[phase] = valid_ranks
        complete = valid_ranks == list(range(WORLD_SIZE))
        if prefix_open and complete:
            prefix.append(phase)
        else:
            prefix_open = False
    return prefix, phase_ranks, issues


def _schedule_state(
    litmus_id: str,
    *,
    timed_out: bool,
    statuses: Sequence[dict[str, Any]],
    marker_prefix: Sequence[str],
    evidence_issues: Sequence[str],
) -> tuple[bool, str | None]:
    final_phase = marker_prefix[-1] if marker_prefix else None
    if evidence_issues:
        return False, final_phase
    if not timed_out:
        complete = {status["rank"] for status in statuses} == set(range(WORLD_SIZE))
        return complete, final_phase
    if litmus_id == "collective_clean_destroy_recreate":
        return "ready_destroy" in marker_prefix, final_phase
    if litmus_id == "collective_partial_epoch_timeout_recreate":
        if "fault_observed" in marker_prefix:
            return True, final_phase
        if marker_prefix and marker_prefix[-1] == "fault_ready":
            return True, "device_work"
    return False, final_phase


def build_launcher_evidence(
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    combined_output: str,
    torchrun_log_dir: Path,
    launcher_exit_code: int | None,
    timed_out: bool,
    signals_sent: Sequence[str],
    launch_error: dict[str, str] | None = None,
) -> dict[str, Any]:
    digest = schedule_digest(config, args)
    statuses, status_issues = collect_rank_statuses(
        combined_output,
        torchrun_log_dir,
        run_id=args.run_id,
        litmus_id=args.litmus,
        expected_schedule_digest=digest,
    )
    marker_prefix, marker_phase_ranks, marker_issues = inspect_marker_prefix(
        args.run_dir,
        run_id=args.run_id,
        litmus_id=args.litmus,
        expected_schedule_digest=digest,
    )
    issues = status_issues + marker_issues
    if launch_error is not None:
        issues.append(
            f"{launch_error['stage']}: {launch_error['error_type']}: "
            f"{launch_error['message']}"
        )
    schedule_complete, final_phase = _schedule_state(
        args.litmus,
        timed_out=timed_out,
        statuses=statuses,
        marker_prefix=marker_prefix,
        evidence_issues=issues,
    )
    return {
        "schema": LAUNCHER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "run_id": args.run_id,
        "litmus_id": args.litmus,
        "world_size": WORLD_SIZE,
        "schedule_digest": digest,
        "platform": args.platform,
        "repetition": args.repetition,
        "source_revision": args.source_revision,
        "outer_timeout_seconds": config["outer_process_timeout_seconds"],
        "timed_out": timed_out,
        "launcher_exit_code": launcher_exit_code,
        "signals_sent": list(signals_sent),
        "rank_statuses": statuses,
        "marker_prefix": marker_prefix,
        "marker_phase_ranks": marker_phase_ranks,
        "final_phase": final_phase,
        "schedule_complete": schedule_complete,
        "launch_error": launch_error,
        "evidence_issues": issues,
        "rank_jsonl": {
            str(rank): {
                "path": f"ranks/rank-{rank:04d}.jsonl",
                "exists": (args.run_dir / "ranks" / f"rank-{rank:04d}.jsonl").is_file(),
            }
            for rank in range(WORLD_SIZE)
        },
    }


def build_adjudicator_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "adjudicate_ap_g0c.py"),
    ]
    for rank in range(WORLD_SIZE):
        command.extend(
            [
                "--rank-jsonl",
                f"{rank}={args.run_dir / 'ranks' / f'rank-{rank:04d}.jsonl'}",
            ]
        )
    control_dir = args.run_dir / "control"
    if control_dir.is_dir():
        command.extend(["--marker", str(control_dir)])
    command.extend(
        [
            "--launcher-evidence",
            str(args.run_dir / "launcher_evidence.json"),
            "--output",
            str(args.run_dir / "summary.json"),
        ]
    )
    return command


def invoke_adjudicator(command: Sequence[str], run_dir: Path) -> int:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    completed = subprocess.run(
        list(command),
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    _write_text_new(run_dir / "adjudicator.log", _as_text(completed.stdout))
    _write_json_new(
        run_dir / "adjudicator_status.json",
        {"exit_code": completed.returncode, "summary_exists": summary_path.is_file()},
    )
    return int(completed.returncode) if summary_path.is_file() else 4


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256_manifest(run_dir: Path) -> Path:
    output = run_dir / "sha256_manifest.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    artifacts = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path != output:
            artifacts[path.relative_to(run_dir).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
    _write_json_new(
        output,
        {"schema": MANIFEST_SCHEMA, "artifacts": artifacts},
    )
    return output


def launch(args: argparse.Namespace) -> int:
    _require_linux()
    config = load_config(args.config)
    if args.run_dir.exists():
        raise FileExistsError(f"refusing to reuse run directory {args.run_dir}")
    torchrun_log_dir = args.run_dir / "torchrun"
    environment, environment_overrides = platform_environment(config, args)
    command = build_torchrun_command(config, args, torchrun_log_dir)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torchrun_log_dir.mkdir()

    runtime_environment, runtime_error = inspect_runtime_environment(
        config, args, environment_overrides
    )
    _write_json_new(
        args.run_dir / "environment.json",
        {
            "base_environment": "inherited; only frozen overrides are recorded",
            "overrides": environment_overrides,
            "runtime": runtime_environment,
            "launch_error": runtime_error,
        },
    )

    combined_output = ""
    launcher_exit_code: int | None = None
    timed_out = False
    signals_sent: list[str] = []
    launch_error = runtime_error
    if launch_error is None:
        try:
            (
                combined_output,
                launcher_exit_code,
                timed_out,
                signals_sent,
            ) = supervise(
                command,
                environment,
                timeout_seconds=float(config["outer_process_timeout_seconds"]),
                termination_grace_seconds=args.termination_grace_seconds,
            )
        except OSError as error:
            launch_error = _structured_error("torchrun_launch", error)
    if launch_error is not None and not combined_output:
        combined_output = json.dumps(launch_error, sort_keys=True) + "\n"

    _write_text_new(args.run_dir / "combined.log", combined_output)
    evidence = build_launcher_evidence(
        config,
        args,
        combined_output=combined_output,
        torchrun_log_dir=torchrun_log_dir,
        launcher_exit_code=launcher_exit_code,
        timed_out=timed_out,
        signals_sent=signals_sent,
        launch_error=launch_error,
    )
    _write_json_new(args.run_dir / "launcher_evidence.json", evidence)

    adjudicator_command = build_adjudicator_command(args)
    _write_json_new(
        args.run_dir / "commands.json",
        {
            "working_directory": str(REPOSITORY_ROOT),
            "torchrun": command,
            "adjudicator": adjudicator_command,
        },
    )

    try:
        adjudicator_exit_code = invoke_adjudicator(adjudicator_command, args.run_dir)
    except (OSError, subprocess.TimeoutExpired) as error:
        _write_text_new(
            args.run_dir / "adjudicator.log",
            f"{type(error).__name__}: {error}\n",
        )
        _write_json_new(
            args.run_dir / "adjudicator_status.json",
            {"error": f"{type(error).__name__}: {error}", "summary_exists": False},
        )
        adjudicator_exit_code = 4
    write_sha256_manifest(args.run_dir)
    return adjudicator_exit_code


def main(argv: Sequence[str] | None = None) -> int:
    return launch(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
