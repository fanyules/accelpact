from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import launch_ap_g0c as launcher  # noqa: E402


class SuccessfulProcess:
    def __init__(self, output: str) -> None:
        self.output = output
        self.returncode = 0
        self.pid = 41001
        self.communicate_calls: list[float | None] = []

    def communicate(self, timeout: float | None = None) -> tuple[str, None]:
        self.communicate_calls.append(timeout)
        return self.output, None


class TimeoutProcess:
    def __init__(self) -> None:
        self.returncode = -int(launcher.SIGKILL)
        self.pid = 41002
        self.communicate_calls = 0

    def communicate(self, timeout: float | None = None) -> tuple[str, None]:
        self.communicate_calls += 1
        if self.communicate_calls <= 2:
            raise subprocess.TimeoutExpired(["torchrun"], timeout)
        return "partial rank output\n", None


class ExceptionalProcess:
    def __init__(
        self,
        initial_error: BaseException,
        *,
        cleanup_error: BaseException | None = None,
        cleanup_times_out: bool = False,
    ) -> None:
        self.initial_error = initial_error
        self.cleanup_error = cleanup_error
        self.cleanup_times_out = cleanup_times_out
        self.returncode = -int(launcher.SIGKILL)
        self.pid = 41003
        self.communicate_calls: list[float | None] = []

    def communicate(self, timeout: float | None = None) -> tuple[str, None]:
        self.communicate_calls.append(timeout)
        if len(self.communicate_calls) == 1:
            raise self.initial_error
        if len(self.communicate_calls) == 2 and self.cleanup_times_out:
            raise subprocess.TimeoutExpired(["torchrun"], timeout)
        if len(self.communicate_calls) == 2 and self.cleanup_error is not None:
            raise self.cleanup_error
        return "partial rank output\n", None


class SigtermProcess:
    def __init__(self) -> None:
        self.returncode = -int(launcher.SIGTERM)
        self.pid = 41004
        self.communicate_calls: list[float | None] = []

    def communicate(self, timeout: float | None = None) -> tuple[str, None]:
        self.communicate_calls.append(timeout)
        if len(self.communicate_calls) == 1:
            handler = launcher.signal.getsignal(launcher.SIGTERM)
            assert callable(handler)
            handler(launcher.SIGTERM, None)
        return "partial rank output\n", None


class LauncherTests(unittest.TestCase):
    def args(self, root: Path, torchrun: Path) -> argparse.Namespace:
        return argparse.Namespace(
            config=REPOSITORY_ROOT / "configs" / "ap_g0c.json",
            platform="a100",
            litmus="collective_same_generation_reuse",
            run_id="ap-g0c-launcher-unit",
            run_dir=root / "run",
            repetition=1,
            source_revision="unit-test-revision",
            torchrun=torchrun,
            master_addr="127.0.0.1",
            master_port=29417,
            termination_grace_seconds=0.25,
        )

    def rank_status(
        self, args: argparse.Namespace, config: dict[str, object], rank: int
    ) -> dict[str, object]:
        return {
            "schema": launcher.RANK_STATUS_SCHEMA,
            "protocol_id": launcher.PROTOCOL_ID,
            "run_id": args.run_id,
            "litmus_id": args.litmus,
            "world_size": launcher.WORLD_SIZE,
            "schedule_digest": launcher.schedule_digest(config, args),
            "rank": rank,
            "classification": "valid_pass",
            "error": None,
            "error_type": None,
        }

    def test_successful_launch_records_frozen_command_evidence_and_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torchrun = root / "bin" / "torchrun"
            torchrun.parent.mkdir()
            torchrun.write_text("#!/bin/sh\n", encoding="utf-8")
            args = self.args(root, torchrun.resolve())
            config = launcher.load_config(args.config)
            output = "\n".join(
                json.dumps(self.rank_status(args, config, rank)) for rank in range(2)
            )
            process = SuccessfulProcess(output + "\n")
            popen_calls: list[tuple[list[str], dict[str, object]]] = []

            def popen(command: list[str], **kwargs: object) -> SuccessfulProcess:
                popen_calls.append((command, kwargs))
                return process

            def adjudicate(command: list[str], run_dir: Path) -> int:
                self.assertIn(f"0={run_dir / 'ranks' / 'rank-0000.jsonl'}", command)
                self.assertIn(f"1={run_dir / 'ranks' / 'rank-0001.jsonl'}", command)
                launcher._write_json_new(
                    run_dir / "summary.json",
                    {"aggregate_classification": "valid_pass", "exit_code": 0},
                )
                launcher._write_text_new(run_dir / "adjudicator.log", "summary\n")
                launcher._write_json_new(
                    run_dir / "adjudicator_status.json",
                    {"exit_code": 0, "summary_exists": True},
                )
                return 0

            with (
                patch.object(launcher, "_require_linux"),
                patch.object(
                    launcher,
                    "inspect_runtime_environment",
                    return_value=(
                        {
                            "python": {"version": "3.10.test"},
                            "torch": "2.11.test",
                            "torch_npu": None,
                            "cuda": "12.8",
                            "nccl": [2, 28, 9],
                            "backend": {"name": "nccl", "available": True},
                            "visible_devices": {
                                "count": 2,
                                "names": ["A100-0", "A100-1"],
                            },
                        },
                        None,
                    ),
                ),
                patch.object(launcher.subprocess, "Popen", side_effect=popen),
                patch.object(launcher, "invoke_adjudicator", side_effect=adjudicate),
            ):
                exit_code = launcher.launch(args)

            evidence = json.loads(
                (args.run_dir / "launcher_evidence.json").read_text(encoding="utf-8")
            )
            commands = json.loads(
                (args.run_dir / "commands.json").read_text(encoding="utf-8")
            )
            environment = json.loads(
                (args.run_dir / "environment.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (args.run_dir / "sha256_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(popen_calls), 1)
        command, kwargs = popen_calls[0]
        self.assertEqual(command[0], str(torchrun.resolve()))
        self.assertIn("--nproc-per-node=2", command)
        self.assertIn("--max-restarts=0", command)
        self.assertIn("--master-addr=127.0.0.1", command)
        self.assertIn("--master-port=29417", command)
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(
            kwargs["env"]["TORCH_NCCL_BLOCKING_WAIT"],  # type: ignore[index]
            "1",
        )
        self.assertEqual(environment["overrides"]["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(environment["runtime"]["visible_devices"]["count"], 2)
        self.assertEqual(environment["runtime"]["nccl"], [2, 28, 9])
        self.assertIsNone(environment["launch_error"])
        self.assertEqual(len(evidence["rank_statuses"]), 2)
        self.assertTrue(evidence["schedule_complete"])
        self.assertFalse(evidence["timed_out"])
        self.assertNotIn("rank_exit_codes", evidence)
        self.assertEqual(commands["torchrun"], command)
        self.assertIn("summary.json", manifest["artifacts"])
        self.assertIn("combined.log", manifest["artifacts"])
        self.assertNotIn("sha256_manifest.json", manifest["artifacts"])

    def test_timeout_terminates_then_kills_the_new_process_group(self) -> None:
        process = TimeoutProcess()
        with (
            patch.object(launcher.subprocess, "Popen", return_value=process) as popen,
            patch.object(launcher.os, "killpg", create=True) as killpg,
        ):
            output, return_code, timed_out, signals_sent = launcher.supervise(
                ["/abs/torchrun"],
                {},
                timeout_seconds=180,
                termination_grace_seconds=0.25,
            )

        self.assertTrue(timed_out)
        self.assertEqual(return_code, -int(launcher.SIGKILL))
        self.assertEqual(output, "partial rank output\n")
        self.assertEqual(signals_sent, ["SIGTERM", "SIGKILL"])
        self.assertEqual(
            killpg.call_args_list,
            [
                call(process.pid, launcher.SIGTERM),
                call(process.pid, launcher.SIGKILL),
            ],
        )
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_normal_supervision_never_signals_the_process_group(self) -> None:
        process = SuccessfulProcess("complete\n")
        with (
            patch.object(launcher.subprocess, "Popen", return_value=process),
            patch.object(launcher.os, "killpg", create=True) as killpg,
        ):
            output, return_code, timed_out, signals_sent = launcher.supervise(
                ["/abs/torchrun"],
                {},
                timeout_seconds=180,
                termination_grace_seconds=0.25,
            )

        self.assertEqual(output, "complete\n")
        self.assertEqual(return_code, 0)
        self.assertFalse(timed_out)
        self.assertEqual(signals_sent, [])
        killpg.assert_not_called()

    def test_keyboard_interrupt_terminates_and_reaps_before_reraising(self) -> None:
        interruption = KeyboardInterrupt("operator interrupt")
        process = ExceptionalProcess(interruption)
        with (
            patch.object(launcher.subprocess, "Popen", return_value=process),
            patch.object(launcher.os, "killpg", create=True) as killpg,
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            launcher.supervise(
                ["/abs/torchrun"],
                {},
                timeout_seconds=180,
                termination_grace_seconds=0.25,
            )

        self.assertIs(raised.exception, interruption)
        self.assertEqual(process.communicate_calls, [180, 0.25])
        killpg.assert_called_once_with(process.pid, launcher.SIGTERM)

    def test_other_exception_escalates_to_kill_and_preserves_original(self) -> None:
        original = RuntimeError("communicate failed")
        process = ExceptionalProcess(original, cleanup_times_out=True)
        with (
            patch.object(launcher.subprocess, "Popen", return_value=process),
            patch.object(launcher.os, "killpg", create=True) as killpg,
            self.assertRaises(RuntimeError) as raised,
        ):
            launcher.supervise(
                ["/abs/torchrun"],
                {},
                timeout_seconds=180,
                termination_grace_seconds=0.25,
            )

        self.assertIs(raised.exception, original)
        self.assertEqual(process.communicate_calls, [180, 0.25, None])
        self.assertEqual(
            killpg.call_args_list,
            [
                call(process.pid, launcher.SIGTERM),
                call(process.pid, launcher.SIGKILL),
            ],
        )

    def test_cleanup_exception_does_not_mask_supervise_exception(self) -> None:
        original = ValueError("primary supervise failure")
        process = ExceptionalProcess(
            original,
            cleanup_error=OSError("cleanup communicate failure"),
        )
        with (
            patch.object(launcher.subprocess, "Popen", return_value=process),
            patch.object(launcher.os, "killpg", create=True) as killpg,
            self.assertRaises(ValueError) as raised,
        ):
            launcher.supervise(
                ["/abs/torchrun"],
                {},
                timeout_seconds=180,
                termination_grace_seconds=0.25,
            )

        self.assertIs(raised.exception, original)
        self.assertEqual(process.communicate_calls, [180, 0.25, None])
        self.assertEqual(
            killpg.call_args_list,
            [
                call(process.pid, launcher.SIGTERM),
                call(process.pid, launcher.SIGKILL),
            ],
        )

    def test_sigterm_becomes_exit_after_child_group_is_reaped(self) -> None:
        process = SigtermProcess()
        previous_sigterm = launcher.signal.getsignal(launcher.SIGTERM)
        with (
            patch.object(launcher.subprocess, "Popen", return_value=process),
            patch.object(launcher.os, "killpg", create=True) as killpg,
            self.assertRaises(SystemExit) as raised,
        ):
            launcher.supervise(
                ["/abs/torchrun"],
                {},
                timeout_seconds=180,
                termination_grace_seconds=0.25,
            )

        self.assertEqual(raised.exception.code, 128 + int(launcher.SIGTERM))
        self.assertEqual(process.communicate_calls, [180, 0.25])
        killpg.assert_called_once_with(process.pid, launcher.SIGTERM)
        self.assertEqual(launcher.signal.getsignal(launcher.SIGTERM), previous_sigterm)

    def test_marker_evidence_uses_only_the_complete_frozen_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id = "marker-prefix-run"
            litmus_id = "collective_clean_destroy_recreate"
            digest = "digest-1"
            for phase in ("ready_destroy", "destroyed"):
                for rank in range(2):
                    launcher._write_json_new(
                        root / "control" / phase / f"rank-{rank:04d}.json",
                        {
                            "protocol_id": launcher.PROTOCOL_ID,
                            "run_id": run_id,
                            "litmus_id": litmus_id,
                            "world_size": launcher.WORLD_SIZE,
                            "schedule_digest": digest,
                            "phase": phase,
                            "rank": rank,
                        },
                    )
            launcher._write_json_new(
                root / "control" / "recreate_start" / "rank-0000.json",
                {
                    "protocol_id": launcher.PROTOCOL_ID,
                    "run_id": run_id,
                    "litmus_id": litmus_id,
                    "world_size": launcher.WORLD_SIZE,
                    "schedule_digest": digest,
                    "phase": "recreate_start",
                    "rank": 0,
                },
            )
            prefix, phase_ranks, issues = launcher.inspect_marker_prefix(
                root,
                run_id=run_id,
                litmus_id=litmus_id,
                expected_schedule_digest=digest,
            )

        self.assertEqual(prefix, ["ready_destroy", "destroyed"])
        self.assertEqual(phase_ranks["recreate_start"], [0])
        self.assertEqual(issues, [])

    def test_partial_timeout_after_fault_ready_is_device_work(self) -> None:
        complete, phase = launcher._schedule_state(
            "collective_partial_epoch_timeout_recreate",
            timed_out=True,
            statuses=[],
            marker_prefix=["fault_ready"],
            evidence_issues=[],
        )
        self.assertTrue(complete)
        self.assertEqual(phase, "device_work")

        complete, phase = launcher._schedule_state(
            "collective_partial_epoch_timeout_recreate",
            timed_out=True,
            statuses=[],
            marker_prefix=["fault_ready", "fault_observed", "ready_destroy"],
            evidence_issues=[],
        )
        self.assertTrue(complete)
        self.assertEqual(phase, "ready_destroy")

    def test_runtime_environment_records_versions_devices_and_import_error(
        self,
    ) -> None:
        class Nccl:
            @staticmethod
            def version() -> tuple[int, int, int]:
                return (2, 28, 9)

        class DeviceApi:
            nccl = Nccl()

            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 2

            @staticmethod
            def get_device_name(index: int) -> str:
                return f"A100-{index}"

        class TorchVersion:
            cuda = "12.8"

        class Torch:
            __version__ = "2.11.test"
            version = TorchVersion()
            cuda = DeviceApi()

        class Dist:
            @staticmethod
            def is_backend_available(name: str) -> bool:
                return name == "nccl"

        def loader(name: str) -> object:
            return {"torch": Torch(), "torch.distributed": Dist()}[name]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torchrun = root / "torchrun"
            torchrun.write_text("", encoding="utf-8")
            args = self.args(root, torchrun.resolve())
            config = launcher.load_config(args.config)
            report, error = launcher.inspect_runtime_environment(
                config,
                args,
                {"CUDA_VISIBLE_DEVICES": "0,1"},
                module_loader=loader,
            )

            def missing_loader(_name: str) -> object:
                raise ModuleNotFoundError("torch is missing")

            missing_report, missing_error = launcher.inspect_runtime_environment(
                config,
                args,
                {"CUDA_VISIBLE_DEVICES": "0,1"},
                module_loader=missing_loader,
            )

        self.assertIsNone(error)
        self.assertEqual(report["torch"], "2.11.test")
        self.assertEqual(report["cuda"], "12.8")
        self.assertEqual(report["nccl"], [2, 28, 9])
        self.assertEqual(report["visible_devices"]["names"], ["A100-0", "A100-1"])
        self.assertEqual(missing_error["error_type"], "ModuleNotFoundError")
        self.assertEqual(missing_report["launch_error"], missing_error)

    def test_npu_runtime_preflight_never_queries_nccl(self) -> None:
        class TrapNccl:
            @staticmethod
            def version() -> None:
                raise AssertionError("NCCL must not be queried for an NPU platform")

        class CudaApi:
            nccl = TrapNccl()

        class NpuApi:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 2

            @staticmethod
            def get_device_name(index: int) -> str:
                return f"910B-{index}"

        class Torch:
            __version__ = "2.10.test"
            cuda = CudaApi()
            npu = NpuApi()

        class TorchNpu:
            __version__ = "2.10.test-npu"

        class Dist:
            @staticmethod
            def is_backend_available(name: str) -> bool:
                return name == "hccl"

        def loader(name: str) -> object:
            return {
                "torch": Torch(),
                "torch.distributed": Dist(),
                "torch_npu": TorchNpu(),
            }[name]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torchrun = root / "torchrun"
            torchrun.write_text("", encoding="utf-8")
            args = self.args(root, torchrun.resolve())
            args.platform = "910b"
            config = launcher.load_config(args.config)
            report, error = launcher.inspect_runtime_environment(
                config,
                args,
                {"ASCEND_RT_VISIBLE_DEVICES": "0,1"},
                module_loader=loader,
            )

        self.assertIsNone(error)
        self.assertEqual(report["torch_npu"], "2.10.test-npu")
        self.assertIsNone(report["cuda"])
        self.assertIsNone(report["nccl"])
        self.assertEqual(
            report["backend"],
            {
                "name": "hccl",
                "available": True,
                "device_api_available": True,
            },
        )
        self.assertEqual(report["visible_devices"]["names"], ["910B-0", "910B-1"])

    def test_torchrun_path_preflight_leaves_no_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root, (root / "missing-torchrun").resolve())
            with patch.object(launcher, "_require_linux"):
                with self.assertRaises(FileNotFoundError):
                    launcher.launch(args)
            self.assertFalse(args.run_dir.exists())

    def test_runtime_import_failure_is_structured_and_skips_torchrun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torchrun = root / "torchrun"
            torchrun.write_text("", encoding="utf-8")
            args = self.args(root, torchrun.resolve())
            launch_error = {
                "stage": "runtime_preflight",
                "error_type": "ModuleNotFoundError",
                "message": "torch is missing",
            }
            runtime = {
                "python": {"version": "3.10.test"},
                "torch": None,
                "torch_npu": None,
                "cuda": None,
                "nccl": None,
                "backend": None,
                "visible_devices": {"count": None, "names": []},
                "launch_error": launch_error,
            }

            def adjudicate(_command: list[str], run_dir: Path) -> int:
                launcher._write_json_new(
                    run_dir / "summary.json",
                    {"aggregate_classification": "harness_error", "exit_code": 4},
                )
                launcher._write_text_new(run_dir / "adjudicator.log", "summary\n")
                launcher._write_json_new(
                    run_dir / "adjudicator_status.json",
                    {"exit_code": 4, "summary_exists": True},
                )
                return 4

            with (
                patch.object(launcher, "_require_linux"),
                patch.object(
                    launcher,
                    "inspect_runtime_environment",
                    return_value=(runtime, launch_error),
                ),
                patch.object(launcher.subprocess, "Popen") as popen,
                patch.object(launcher, "invoke_adjudicator", side_effect=adjudicate),
            ):
                exit_code = launcher.launch(args)

            environment = json.loads(
                (args.run_dir / "environment.json").read_text(encoding="utf-8")
            )
            evidence = json.loads(
                (args.run_dir / "launcher_evidence.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 4)
        popen.assert_not_called()
        self.assertEqual(environment["launch_error"], launch_error)
        self.assertEqual(environment["runtime"]["launch_error"], launch_error)
        self.assertEqual(evidence["launch_error"], launch_error)
        self.assertFalse(evidence["schedule_complete"])

    def test_existing_run_directory_is_never_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torchrun = root / "torchrun"
            torchrun.write_text("", encoding="utf-8")
            args = self.args(root, torchrun.resolve())
            args.run_dir.mkdir()
            with patch.object(launcher, "_require_linux"):
                with self.assertRaises(FileExistsError):
                    launcher.launch(args)


if __name__ == "__main__":
    unittest.main()
