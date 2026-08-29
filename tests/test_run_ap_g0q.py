from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import run_ap_g0q as runner  # noqa: E402

from accelpact import (  # noqa: E402
    LifecycleOracle,
    LitmusResult,
    ResourceKind,
    read_results,
)


class FakeBackend:
    name = "cuda"

    def __init__(self) -> None:
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class RunnerContractTests(unittest.TestCase):
    def test_frozen_litmus_order_and_defaults(self) -> None:
        self.assertEqual(len(runner.POSITIVE_LITMUS), 6)
        self.assertEqual(runner.NEGATIVE_CONTROLS, ("missing_join", "rebound_input"))
        self.assertEqual(runner.DEFAULT_ITERATIONS, 128)
        self.assertEqual(runner.DEFAULT_SEED, 20260829)
        self.assertEqual(set(runner.LITMUS_FUNCTIONS), set(runner.LITMUS_ORDER))

    def test_missing_join_is_detected_without_becoming_runtime_violation(self) -> None:
        with patch.object(
            runner,
            "capture_multistream_affine_graph",
            side_effect=RuntimeError("capture streams did not rejoin origin"),
        ):
            result = runner.missing_join(FakeBackend(), 128, 20260829)
        self.assertTrue(result.passed)
        self.assertEqual(
            result.observations["classification"], "negative_control_detected"
        )
        self.assertFalse(result.observations["runtime_violation"])

    def test_unsupported_is_passed_and_excluded_from_protocol_failure(self) -> None:
        result = runner.unsupported_result(
            "graph_cross_stream_join", "npu", "graph API unavailable"
        )
        summary = runner.summarize([result])
        self.assertTrue(result.passed)
        self.assertEqual(result.observations["classification"], "unsupported")
        self.assertEqual(summary["protocol_failures"], [])
        self.assertTrue(summary["runtime_protocol_pass"])

    def test_protocol_violation_is_the_only_failed_result_class(self) -> None:
        oracle = LifecycleOracle(ResourceKind.BUFFER, "buffer-0")
        result = runner.make_result(
            oracle,
            litmus_id="event_rerecord_single_waiter",
            backend="cuda",
            classification="protocol_violation",
            error="stale value",
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.error, "stale value")

    def test_capture_abort_outer_failure_is_retained_as_harness_error(self) -> None:
        result = runner.failure_result(
            "capture_abort_eager_recovery", "npu", RuntimeError("context poisoned")
        )
        self.assertEqual(result.final_state, "poisoned")
        self.assertEqual(result.observations["classification"], "harness_error")

    def test_measured_unsupported_feature_has_no_protocol_failure(self) -> None:
        backend = FakeBackend()

        def unsupported(*_args: object) -> LitmusResult:
            raise runner.UnsupportedFeature("not implemented")

        with patch.dict(
            runner.LITMUS_FUNCTIONS,
            {"graph_cross_stream_join": unsupported},
        ):
            result = runner.measured_result(
                "graph_cross_stream_join", backend, 128, 20260829
            )
        self.assertEqual(result.observations["classification"], "unsupported")
        self.assertTrue(result.passed)

    def test_unavailable_backend_writes_eight_unsupported_jsonl_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            argv = ["run_ap_g0q.py", "--backend", "cuda", "--output", str(output)]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    runner,
                    "resolve_backend",
                    return_value=(None, "cuda:0 unavailable"),
                ),
            ):
                self.assertEqual(runner.main(), 0)
            rows = read_results(output)
        self.assertEqual([row.litmus_id for row in rows], list(runner.LITMUS_ORDER))
        self.assertTrue(
            all(row.observations["classification"] == "unsupported" for row in rows)
        )


if __name__ == "__main__":
    unittest.main()
