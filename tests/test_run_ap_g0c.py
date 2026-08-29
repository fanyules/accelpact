from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import run_ap_g0c as runner  # noqa: E402

from accelpact import read_results  # noqa: E402


class FakeTensor:
    def __init__(self, values: list[int]):
        self.values = list(values)

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list[int]:
        return list(self.values)


class FakeDeviceApi:
    def __init__(self) -> None:
        self.device_index: int | None = None
        self.seeds: list[int] = []

    def set_device(self, device_index: int) -> None:
        self.device_index = device_index

    def manual_seed_all(self, seed: int) -> None:
        self.seeds.append(seed)


class FakeTorch:
    int64 = "int64"

    def __init__(self) -> None:
        self.cuda = FakeDeviceApi()
        self.seeds: list[int] = []

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def device(self, value: str) -> str:
        return value

    def tensor(self, values: list[int], *, dtype: object, device: object) -> FakeTensor:
        del dtype, device
        return FakeTensor(values)


class FakeWork:
    def __init__(self, behavior: str = "success") -> None:
        self.behavior = behavior
        self.wait_timeouts: list[object] = []

    def wait(self, timeout: object) -> bool:
        self.wait_timeouts.append(timeout)
        if self.behavior == "timeout":
            raise TimeoutError("expected fake collective timeout")
        if self.behavior == "false":
            return False
        return True


class FakeReduceOp:
    SUM = "sum"


class FakeDist:
    ReduceOp = FakeReduceOp

    def __init__(self, behaviors: list[str] | None = None) -> None:
        self.behaviors = list(behaviors or [])
        self.all_reduce_calls: list[dict[str, object]] = []
        self.destroy_calls: list[object | None] = []
        self.init_calls: list[dict[str, object]] = []
        self.new_group_calls: list[dict[str, object]] = []
        self.groups: list[object] = []

    def init_process_group(self, **kwargs: object) -> None:
        self.init_calls.append(kwargs)

    def new_group(self, **kwargs: object) -> object:
        self.new_group_calls.append(kwargs)
        group = object()
        self.groups.append(group)
        return group

    def all_reduce(self, tensor: FakeTensor, **kwargs: object) -> FakeWork:
        behavior = self.behaviors.pop(0) if self.behaviors else "success"
        self.all_reduce_calls.append({"behavior": behavior, **kwargs})
        if behavior == "raise":
            raise RuntimeError("fake dispatch failure")
        if behavior == "success":
            generation, epoch, _rank, _one = tensor.values
            tensor.values = [2 * generation, 2 * epoch, 1, 2]
        return FakeWork(behavior)

    def destroy_process_group(self, group: object | None = None) -> None:
        self.destroy_calls.append(group)


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_config = runner.load_config(
            REPOSITORY_ROOT / "configs" / "ap_g0c.json"
        )

    def args(
        self,
        directory: Path,
        litmus_id: str,
        *,
        config_path: Path | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            config=config_path or REPOSITORY_ROOT / "configs" / "ap_g0c.json",
            platform="a100",
            litmus=litmus_id,
            run_id="ap-g0c-unit-run",
            output_dir=directory,
            repetition=1,
            source_revision="unit-test-revision",
            raw_rank_log_reference=None,
        )

    def context(
        self,
        directory: Path,
        litmus_id: str,
        *,
        rank: int = 0,
        config: dict[str, object] | None = None,
        dist: FakeDist | None = None,
    ) -> tuple[runner.RankContext, FakeDist]:
        selected = deepcopy(config or self.base_config)
        fake_torch = FakeTorch()
        fake_dist = dist or FakeDist()
        environment = {
            "LOCAL_RANK": str(rank),
            "RANK": str(rank),
            "TORCH_NCCL_BLOCKING_WAIT": "1",
            "WORLD_SIZE": "2",
        }
        with patch.dict(os.environ, environment, clear=False):
            context = runner.build_context(
                self.args(directory, litmus_id),
                selected,
                runtime=(fake_torch, fake_dist, fake_torch.cuda),
            )
        return context, fake_dist

    def small_config(self) -> dict[str, object]:
        config = deepcopy(self.base_config)
        config["same_generation_epochs"] = 3
        config["clean_recreate_warmup_epochs"] = 2
        config["recovery_epochs"] = 2
        config["oob_phase_timeout_seconds"] = 0.05
        return config

    def test_epoch_loop_uses_exact_payload_and_same_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context, dist = self.context(
                Path(directory),
                "collective_same_generation_reuse",
                config=self.small_config(),
            )
            evidence = runner._new_evidence(context, 0, 0)
            runner._run_epochs(context, object(), evidence, count=3)

        self.assertEqual(evidence.oracle.state, "reusable_same_generation")
        self.assertEqual(len(evidence.ledger), 3)
        self.assertTrue(all(row["payload_matches"] for row in evidence.ledger))
        self.assertEqual(context.backend_dispatch_count, 3)
        self.assertEqual(len(dist.all_reduce_calls), 3)
        targets = [transition.target for transition in evidence.oracle.history]
        self.assertEqual(targets.count("epoch_open"), 2)

    def test_hard_link_markers_are_validated_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rank0, _ = self.context(root, "collective_clean_destroy_recreate")
            rank1, _ = self.context(root, "collective_clean_destroy_recreate", rank=1)
            runner.publish_marker(rank0, "ready_destroy")
            runner.publish_marker(rank1, "ready_destroy")
            rows = runner.wait_for_markers(rank0, "ready_destroy")
            self.assertEqual(set(rows), {0, 1})
            self.assertEqual(
                list((root / "control" / "ready_destroy").glob("*.tmp")), []
            )
            with self.assertRaises(runner.MarkerError):
                runner.publish_marker(rank0, "ready_destroy")

    def test_stale_generation_control_detects_before_vendor_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context, dist = self.context(Path(directory), "stale_generation_dispatch")
            outcome = runner.stale_generation_dispatch(context, object())

        self.assertEqual(outcome.classification, "negative_control_detected")
        self.assertEqual(len(outcome.rows), 1)
        row = outcome.rows[0]
        self.assertTrue(row.passed)
        self.assertEqual(row.observations["stale_request_backend_dispatch_count"], 0)
        self.assertEqual(len(dist.all_reduce_calls), 1)

    def test_stale_generation_guard_miss_is_not_a_protocol_violation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context, dist = self.context(Path(directory), "stale_generation_dispatch")
            with patch.object(runner, "_guard_generation", return_value=None):
                outcome = runner.stale_generation_dispatch(context, object())

        self.assertEqual(outcome.classification, "negative_control_missed")
        self.assertFalse(outcome.rows[0].passed)
        self.assertEqual(len(dist.all_reduce_calls), 0)
        self.assertEqual(
            outcome.rows[0].observations["classification"],
            "negative_control_missed",
        )
        self.assertEqual(outcome.rows[0].observations["request_generation"], 0)
        self.assertEqual(outcome.rows[0].observations["current_generation"], 1)
        self.assertEqual(outcome.rows[0].observations["retired_generation"], 0)
        self.assertEqual(
            outcome.rows[0].observations["stale_request_backend_dispatch_count"],
            0,
        )
        self.assertTrue(outcome.rows[0].observations["would_dispatch"])

    def test_clean_recreate_writes_two_generation_rows(self) -> None:
        events: list[str] = []
        replacement = object()
        config = self.small_config()
        with tempfile.TemporaryDirectory() as directory:
            context, dist = self.context(
                Path(directory),
                "collective_clean_destroy_recreate",
                config=config,
            )
            original_group = object()

            def phase(
                _context: runner.RankContext,
                name: str,
                extra: dict[str, object] | None = None,
            ) -> dict[int, dict[str, object]]:
                del extra
                events.append(f"phase:{name}")
                rows = {0: {}, 1: {}}
                _context.marker_cache[name] = rows
                return rows

            def create(_context: runner.RankContext) -> object:
                events.append("new_group")
                return replacement

            original_destroy = dist.destroy_process_group

            def destroy(group: object | None = None) -> None:
                events.append("destroy")
                original_destroy(group)

            dist.destroy_process_group = destroy
            with (
                patch.object(runner, "publish_and_wait", side_effect=phase),
                patch.object(runner, "_create_device_group", side_effect=create),
            ):
                outcome = runner.collective_clean_destroy_recreate(
                    context, original_group
                )

        self.assertEqual(outcome.classification, "capability_pass")
        self.assertEqual(
            [row.final_state for row in outcome.rows],
            [
                "recreated",
                "reusable_same_generation",
            ],
        )
        self.assertEqual(
            [row.observations["generation"] for row in outcome.rows], [0, 1]
        )
        self.assertEqual(
            [row.observations["backend_dispatch_count"] for row in outcome.rows],
            [2, 2],
        )
        self.assertEqual(
            events,
            [
                "phase:ready_destroy",
                "destroy",
                "phase:destroyed",
                "phase:recreate_start",
                "new_group",
            ],
        )

    def test_recreate_unavailable_writes_only_existing_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context, _ = self.context(
                Path(directory),
                "collective_clean_destroy_recreate",
                config=self.small_config(),
            )
            with (
                patch.object(
                    runner,
                    "publish_and_wait",
                    return_value={0: {}, 1: {}},
                ),
                patch.object(
                    runner,
                    "_create_device_group",
                    side_effect=NotImplementedError("backend is unavailable"),
                ),
            ):
                outcome = runner.collective_clean_destroy_recreate(context, object())

        self.assertEqual(outcome.classification, "capability_unavailable")
        self.assertEqual(len(outcome.rows), 1)
        self.assertEqual(outcome.rows[0].final_state, "destroyed")
        self.assertTrue(outcome.rows[0].passed)

    def test_partial_timeout_recreates_and_recovers(self) -> None:
        config = self.small_config()
        dist = FakeDist(["success", "timeout", "success", "success"])
        replacement = object()
        with tempfile.TemporaryDirectory() as directory:
            context, _ = self.context(
                Path(directory),
                "collective_partial_epoch_timeout_recreate",
                config=config,
                dist=dist,
            )
            with (
                patch.object(
                    runner,
                    "publish_and_wait",
                    return_value={0: {}, 1: {}},
                ),
                patch.object(runner, "_create_device_group", return_value=replacement),
            ):
                outcome = runner.collective_partial_epoch_timeout_recreate(
                    context, object()
                )

        self.assertEqual(outcome.classification, "expected_timeout_recovered")
        self.assertEqual(len(outcome.rows), 2)
        self.assertEqual(
            [row.observations["classification"] for row in outcome.rows],
            ["expected_timeout_recovered", "capability_pass"],
        )
        self.assertEqual(outcome.rows[0].final_state, "recreated")
        targets = [transition.target for transition in outcome.rows[0].transitions]
        self.assertIn("failed_unknown", targets)
        self.assertFalse(outcome.rows[0].observations["old_object_reused"])

    def test_partial_sync_dispatch_error_is_inconclusive(self) -> None:
        config = self.small_config()
        dist = FakeDist(["success", "raise"])
        phases: list[tuple[str, dict[str, object] | None]] = []

        def phase(
            _context: runner.RankContext,
            name: str,
            extra: dict[str, object] | None = None,
        ) -> dict[int, dict[str, object]]:
            del _context
            phases.append((name, extra))
            return {0: {}, 1: {}}

        with tempfile.TemporaryDirectory() as directory:
            context, _ = self.context(
                Path(directory),
                "collective_partial_epoch_timeout_recreate",
                config=config,
                dist=dist,
            )
            with patch.object(runner, "publish_and_wait", side_effect=phase):
                outcome = runner.collective_partial_epoch_timeout_recreate(
                    context, object()
                )

        self.assertEqual(outcome.classification, "inconclusive")
        self.assertEqual(
            [name for name, _extra in phases],
            [
                "fault_ready",
                "fault_observed",
            ],
        )
        fault_marker = phases[-1][1]
        assert fault_marker is not None
        self.assertFalse(fault_marker["fault_observed"])
        self.assertFalse(fault_marker["work_handle_present"])
        self.assertNotIn(
            "failed_unknown",
            [transition.target for transition in outcome.rows[0].transitions],
        )

    def test_partial_completed_fault_uses_false_frozen_marker(self) -> None:
        config = self.small_config()
        dist = FakeDist(["success", "success"])
        phases: list[tuple[str, dict[str, object] | None]] = []

        def phase(
            _context: runner.RankContext,
            name: str,
            extra: dict[str, object] | None = None,
        ) -> dict[int, dict[str, object]]:
            del _context
            phases.append((name, extra))
            return {0: {}, 1: {}}

        with tempfile.TemporaryDirectory() as directory:
            context, _ = self.context(
                Path(directory),
                "collective_partial_epoch_timeout_recreate",
                config=config,
                dist=dist,
            )
            with patch.object(runner, "publish_and_wait", side_effect=phase):
                outcome = runner.collective_partial_epoch_timeout_recreate(
                    context, object()
                )

        self.assertEqual(outcome.classification, "inconclusive")
        self.assertEqual(
            [name for name, _extra in phases],
            [
                "fault_ready",
                "fault_observed",
            ],
        )
        fault_marker = phases[-1][1]
        assert fault_marker is not None
        self.assertFalse(fault_marker["fault_observed"])
        self.assertTrue(fault_marker["work_handle_present"])

    def test_run_writes_one_rank_jsonl_and_refuses_overwrite(self) -> None:
        config = self.small_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            args = self.args(
                root,
                "collective_same_generation_reuse",
                config_path=config_path,
            )
            fake_torch = FakeTorch()
            fake_dist = FakeDist()
            environment = {
                "LOCAL_RANK": "0",
                "RANK": "0",
                "TORCH_NCCL_BLOCKING_WAIT": "1",
                "WORLD_SIZE": "2",
            }
            with patch.dict(os.environ, environment, clear=False):
                return_code = runner.run(
                    args, runtime=(fake_torch, fake_dist, fake_torch.cuda)
                )
                with self.assertRaises(FileExistsError):
                    runner.run(args, runtime=(fake_torch, fake_dist, fake_torch.cuda))
            rows = read_results(root / "ranks" / "rank-0000.jsonl")

        self.assertEqual(return_code, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].observations["classification"], "valid_pass")
        self.assertEqual(len(fake_dist.destroy_calls), 2)
        self.assertIsNone(fake_dist.destroy_calls[-1])
        self.assertEqual(fake_torch.seeds, [])
        self.assertEqual(fake_torch.cuda.seeds, [])

    def test_pre_destroy_failure_keeps_old_group_for_cleanup(self) -> None:
        config = self.small_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            args = self.args(
                root,
                "collective_clean_destroy_recreate",
                config_path=config_path,
            )
            fake_torch = FakeTorch()
            fake_dist = FakeDist()
            environment = {
                "LOCAL_RANK": "0",
                "RANK": "0",
                "TORCH_NCCL_BLOCKING_WAIT": "1",
                "WORLD_SIZE": "2",
            }
            with (
                patch.dict(os.environ, environment, clear=False),
                patch.object(
                    runner,
                    "publish_and_wait",
                    side_effect=runner.MarkerError("pre-destroy marker failure"),
                ),
            ):
                return_code = runner.run(
                    args, runtime=(fake_torch, fake_dist, fake_torch.cuda)
                )

        self.assertEqual(return_code, 4)
        self.assertEqual(len(fake_dist.destroy_calls), 2)
        self.assertIs(fake_dist.destroy_calls[0], fake_dist.groups[0])
        self.assertIsNone(fake_dist.destroy_calls[1])

    def test_post_destroy_recreate_failure_preserves_oob_only_gap(self) -> None:
        class RecreateUnavailableDist(FakeDist):
            def new_group(self, **kwargs: object) -> object:
                self.new_group_calls.append(kwargs)
                if self.groups:
                    raise NotImplementedError("backend is unavailable")
                group = object()
                self.groups.append(group)
                return group

        config = self.small_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            args = self.args(
                root,
                "collective_clean_destroy_recreate",
                config_path=config_path,
            )
            fake_torch = FakeTorch()
            fake_dist = RecreateUnavailableDist()
            environment = {
                "LOCAL_RANK": "0",
                "RANK": "0",
                "TORCH_NCCL_BLOCKING_WAIT": "1",
                "WORLD_SIZE": "2",
            }
            with (
                patch.dict(os.environ, environment, clear=False),
                patch.object(
                    runner,
                    "publish_and_wait",
                    return_value={0: {}, 1: {}},
                ),
            ):
                return_code = runner.run(
                    args, runtime=(fake_torch, fake_dist, fake_torch.cuda)
                )

        self.assertEqual(return_code, 5)
        self.assertEqual(fake_dist.destroy_calls, [fake_dist.groups[0]])

    def test_result_write_failure_still_destroys_replacement_group(self) -> None:
        config = self.small_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            args = self.args(
                root,
                "collective_clean_destroy_recreate",
                config_path=config_path,
            )
            fake_torch = FakeTorch()
            fake_dist = FakeDist()
            environment = {
                "LOCAL_RANK": "0",
                "RANK": "0",
                "TORCH_NCCL_BLOCKING_WAIT": "1",
                "WORLD_SIZE": "2",
            }
            with (
                patch.dict(os.environ, environment, clear=False),
                patch.object(
                    runner,
                    "publish_and_wait",
                    return_value={0: {}, 1: {}},
                ),
                patch.object(
                    runner,
                    "append_result",
                    side_effect=OSError("synthetic write failure"),
                ),
            ):
                return_code = runner.run(
                    args, runtime=(fake_torch, fake_dist, fake_torch.cuda)
                )

        self.assertEqual(return_code, 4)
        self.assertEqual(len(fake_dist.groups), 2)
        self.assertEqual(len(fake_dist.destroy_calls), 3)
        self.assertIs(fake_dist.destroy_calls[0], fake_dist.groups[0])
        self.assertIs(fake_dist.destroy_calls[1], fake_dist.groups[1])
        self.assertIsNone(fake_dist.destroy_calls[2])

    def test_initial_group_unavailable_produces_no_rank_row(self) -> None:
        class UnavailableDist(FakeDist):
            def new_group(self, **kwargs: object) -> object:
                del kwargs
                raise NotImplementedError("backend is unavailable")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root, "collective_same_generation_reuse")
            fake_torch = FakeTorch()
            fake_dist = UnavailableDist()
            environment = {
                "LOCAL_RANK": "0",
                "RANK": "0",
                "TORCH_NCCL_BLOCKING_WAIT": "1",
                "WORLD_SIZE": "2",
            }
            with patch.dict(os.environ, environment, clear=False):
                return_code = runner.run(
                    args, runtime=(fake_torch, fake_dist, fake_torch.cuda)
                )

        self.assertEqual(return_code, 5)
        self.assertFalse((root / "ranks" / "rank-0000.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
