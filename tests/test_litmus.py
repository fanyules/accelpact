from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from accelpact import (  # noqa: E402
    InvalidTransition,
    LifecycleOracle,
    LitmusResult,
    ResourceKind,
    append_result,
    read_results,
)


class LifecycleOracleTests(unittest.TestCase):
    def test_buffer_happy_path(self) -> None:
        oracle = LifecycleOracle(ResourceKind.BUFFER, "buffer-0")
        for state in (
            "write_pending",
            "ready",
            "published",
            "consumed",
            "reclaimable",
        ):
            oracle.transition(state)
        self.assertEqual(oracle.state, "reclaimable")
        self.assertEqual(len(oracle.history), 5)
        self.assertEqual(oracle.allowed_targets, frozenset())

    def test_graph_commit_and_abort_branches(self) -> None:
        committed = LifecycleOracle("graph", "graph-committed")
        committed.transition("capturing")
        committed.transition("committed")
        committed.transition("replayable")
        self.assertEqual(committed.state, "replayable")

        for terminal in ("clean_fallback", "poisoned"):
            aborted = LifecycleOracle("graph", f"graph-{terminal}")
            aborted.transition("capturing")
            aborted.transition("aborted")
            aborted.transition(terminal)
            self.assertEqual(aborted.state, terminal)

    def test_collective_success_reuses_only_the_same_generation(self) -> None:
        oracle = LifecycleOracle("collective", "collective-generation-7")
        oracle.transition("enqueued")
        oracle.transition("completed")
        oracle.transition("reusable_same_generation")
        self.assertEqual(oracle.state, "reusable_same_generation")

    def test_collective_success_loops_across_multiple_epochs(self) -> None:
        oracle = LifecycleOracle("collective", "collective-generation-7")
        for epoch in range(3):
            for state in ("enqueued", "completed", "reusable_same_generation"):
                oracle.transition(state)
            if epoch < 2:
                oracle.transition("epoch_open")

        self.assertEqual(oracle.state, "reusable_same_generation")
        self.assertEqual(len(oracle.history), 11)

    def test_collective_clean_retire_destroys_and_recreates(self) -> None:
        oracle = LifecycleOracle("collective", "collective-generation-7")
        for state in (
            "enqueued",
            "completed",
            "reusable_same_generation",
            "retiring",
            "destroyed",
            "recreated",
        ):
            oracle.transition(state)

        self.assertEqual(oracle.state, "recreated")

    def test_collective_unknown_failure_requires_destroy_and_recreate(self) -> None:
        oracle = LifecycleOracle("collective", "collective-generation-7")
        oracle.transition("enqueued")
        oracle.transition("failed_unknown")
        with self.assertRaises(InvalidTransition):
            oracle.transition("reusable_same_generation")
        for state in ("aborting", "destroyed", "recreated"):
            oracle.transition(state)
        self.assertEqual(oracle.state, "recreated")

    def test_collective_clean_and_failure_branches_cannot_cross(self) -> None:
        clean = LifecycleOracle("collective", "collective-clean")
        for state in ("enqueued", "completed", "reusable_same_generation"):
            clean.transition(state)
        clean_history = clean.history
        with self.assertRaises(InvalidTransition):
            clean.transition("failed_unknown")
        self.assertEqual(clean.state, "reusable_same_generation")
        self.assertEqual(clean.history, clean_history)

        failed = LifecycleOracle("collective", "collective-failed")
        for state in ("enqueued", "failed_unknown"):
            failed.transition(state)
        failed_history = failed.history
        with self.assertRaises(InvalidTransition):
            failed.transition("retiring")
        self.assertEqual(failed.state, "failed_unknown")
        self.assertEqual(failed.history, failed_history)

    def test_illegal_transition_does_not_mutate_state_or_history(self) -> None:
        oracle = LifecycleOracle("buffer", "buffer-illegal")
        with self.assertRaisesRegex(InvalidTransition, "owned -> published"):
            oracle.transition("published")
        self.assertEqual(oracle.state, "owned")
        self.assertEqual(oracle.history, ())

    def test_illegal_cross_branch_graph_transition_is_rejected(self) -> None:
        oracle = LifecycleOracle("graph", "graph-illegal")
        oracle.transition("capturing")
        oracle.transition("committed")
        with self.assertRaises(InvalidTransition):
            oracle.transition("clean_fallback")
        self.assertEqual(oracle.state, "committed")


class LitmusResultTests(unittest.TestCase):
    def completed_buffer_result(self, litmus_id: str) -> LitmusResult:
        oracle = LifecycleOracle("buffer", "buffer-0")
        for state in (
            "write_pending",
            "ready",
            "published",
            "consumed",
            "reclaimable",
        ):
            oracle.transition(state)
        return LitmusResult.from_oracle(
            oracle,
            litmus_id=litmus_id,
            backend="cpu-reference",
            passed=True,
            observations={"payload_matches": True, "rank": 0},
        )

    def test_result_round_trip_validates_transition_history(self) -> None:
        result = self.completed_buffer_result("buffer-publication")
        restored = LitmusResult.from_dict(result.to_dict())
        self.assertEqual(restored, result)

        invalid = result.to_dict()
        invalid["transitions"][2]["source"] = "owned"
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            LitmusResult.from_dict(invalid)

    def test_failed_result_requires_error_and_passed_result_rejects_error(self) -> None:
        oracle = LifecycleOracle("graph", "graph-0")
        with self.assertRaisesRegex(ValueError, "failed result"):
            LitmusResult.from_oracle(
                oracle,
                litmus_id="capture-abort",
                backend="cpu-reference",
                passed=False,
            )
        with self.assertRaisesRegex(ValueError, "passed result"):
            LitmusResult.from_oracle(
                oracle,
                litmus_id="capture-abort",
                backend="cpu-reference",
                passed=True,
                error="unexpected",
            )

    def test_jsonl_append_and_read_preserves_two_results(self) -> None:
        first = self.completed_buffer_result("buffer-1")
        second = self.completed_buffer_result("buffer-2")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "results.jsonl"
            append_result(path, first)
            append_result(path, second)
            self.assertEqual(read_results(path), [first, second])
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(all(isinstance(json.loads(line), dict) for line in lines))

    def test_non_finite_observation_is_rejected_before_write(self) -> None:
        oracle = LifecycleOracle("buffer", "buffer-0")
        with self.assertRaises(ValueError):
            LitmusResult.from_oracle(
                oracle,
                litmus_id="non-finite",
                backend="cpu-reference",
                passed=True,
                observations={"latency_ms": float("nan")},
            )


if __name__ == "__main__":
    unittest.main()
