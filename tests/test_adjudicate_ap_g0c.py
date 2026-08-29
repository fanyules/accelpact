from __future__ import annotations

import contextlib
import copy
import importlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

adjudicator = importlib.import_module("adjudicate_ap_g0c")


RUN_ID = "synthetic-ap-g0c-run"
SCHEDULE_DIGEST = "schedule-digest-001"


def transition_rows(targets: list[str]) -> tuple[list[dict[str, object]], str]:
    source = "epoch_open"
    rows = []
    for index, target in enumerate(targets):
        rows.append({"index": index, "source": source, "target": target})
        source = target
    return rows, source


def result_targets(litmus_id: str, generation: int, classification: str) -> list[str]:
    success = ["enqueued", "completed", "reusable_same_generation"]
    if classification == "protocol_violation":
        return ["enqueued"]
    if litmus_id == "collective_clean_destroy_recreate" and generation == 0:
        if classification == "reinitialization_capability_failure":
            return success + ["retiring"]
        if classification == "capability_unavailable":
            return success + ["retiring", "destroyed"]
        if classification == "capability_pass":
            return success + ["retiring", "destroyed", "recreated"]
    if litmus_id == "collective_partial_epoch_timeout_recreate" and generation == 0:
        fault = success + ["epoch_open", "enqueued"]
        if classification == "inconclusive":
            return fault
        if classification in {
            "expected_timeout_recovered",
            "capability_unavailable",
            "reinitialization_capability_failure",
        }:
            return fault + ["failed_unknown", "aborting", "destroyed", "recreated"]
    return success


def result_row(
    rank: int,
    generation: int,
    classification: str,
    *,
    litmus_id: str = "collective_same_generation_reuse",
    schedule_digest: str = SCHEDULE_DIGEST,
) -> dict[str, object]:
    passed = classification in adjudicator.PASS_CLASSIFICATIONS or (
        classification == "capability_unavailable"
    )
    transitions, final_state = transition_rows(
        result_targets(litmus_id, generation, classification)
    )
    observations: dict[str, object] = {
        "protocol_id": adjudicator.PROTOCOL_ID,
        "run_id": RUN_ID,
        "litmus_id": litmus_id,
        "rank": rank,
        "world_size": adjudicator.WORLD_SIZE,
        "generation": generation,
        "schedule_digest": schedule_digest,
        "classification": classification,
        "collective_backend": "nccl",
        "source_revision": "bb25b98",
        "tensor_shape": [4],
        "dtype": "int64",
        "reduce_op": "sum",
    }
    if litmus_id == "collective_partial_epoch_timeout_recreate" and generation == 0:
        observations.update(
            {
                "fault_details": {
                    "fault_observed": classification != "inconclusive",
                    "work_handle_present": classification != "inconclusive",
                },
                "local_enqueued": rank == 0,
                "local_enqueue_role": "submit_fault" if rank == 0 else "skip_fault",
            }
        )
    expected = adjudicator._successful_epoch_contract(
        litmus_id, generation, classification
    )
    if expected is not None:
        epoch_start, epoch_end, completion_count = expected
        digest = f"payload-g{generation}"
        observations.update(
            {
                "epoch_start": epoch_start,
                "epoch_end": epoch_end,
                "completion_count": completion_count,
                "epoch_ledger": [
                    {
                        "epoch": epoch_start + offset,
                        "generation": generation,
                        "completion_count": 1,
                        "expected_payload_digest": adjudicator._canonical_digest(
                            [
                                2 * generation,
                                2 * (epoch_start + offset),
                                1,
                                2,
                            ]
                        ),
                        "observed_payload_digest": adjudicator._canonical_digest(
                            [
                                2 * generation,
                                2 * (epoch_start + offset),
                                1,
                                2,
                            ]
                        ),
                        "payload_matches": True,
                    }
                    for offset in range(completion_count)
                ],
                "expected_payload_digest": digest,
                "observed_payload_digest": digest,
            }
        )
    return {
        "schema": "accelpact.stateful_litmus_result.v1",
        "litmus_id": litmus_id,
        "backend": "nccl",
        "subject_id": f"comm-g{generation:03d}-rank{rank:04d}",
        "resource": "collective",
        "passed": passed,
        "initial_state": "epoch_open",
        "final_state": final_state,
        "transitions": transitions,
        "observations": observations,
        "error": None if passed else f"synthetic {classification}",
    }


def marker(
    rank: int,
    phase: str,
    *,
    litmus_id: str = "collective_clean_destroy_recreate",
    fault_observed: bool | None = None,
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "protocol_id": adjudicator.PROTOCOL_ID,
        "run_id": RUN_ID,
        "litmus_id": litmus_id,
        "phase": phase,
        "rank": rank,
        "world_size": adjudicator.WORLD_SIZE,
        "schedule_digest": SCHEDULE_DIGEST,
    }
    if fault_observed is not None:
        payload["fault_observed"] = fault_observed
    payload.update(extra)
    return payload


def launcher_timeout(
    litmus_id: str = "collective_clean_destroy_recreate",
    *,
    schedule_complete: bool = True,
) -> dict[str, object]:
    return {
        "protocol_id": adjudicator.PROTOCOL_ID,
        "run_id": RUN_ID,
        "litmus_id": litmus_id,
        "world_size": adjudicator.WORLD_SIZE,
        "schedule_digest": SCHEDULE_DIGEST,
        "timed_out": True,
        "schedule_complete": schedule_complete,
        "final_phase": "recreate_start",
        "rank_exit_codes": {"0": None, "1": None},
    }


def rank_status(
    rank: int,
    classification: str,
    *,
    litmus_id: str = "collective_same_generation_reuse",
) -> dict[str, object]:
    return {
        "schema": "accelpact.ap_g0c.rank_status.v1",
        "protocol_id": adjudicator.PROTOCOL_ID,
        "run_id": RUN_ID,
        "litmus_id": litmus_id,
        "rank": rank,
        "world_size": adjudicator.WORLD_SIZE,
        "schedule_digest": SCHEDULE_DIGEST,
        "classification": classification,
        "error_type": None,
        "error": None,
    }


def launcher_non_timeout(
    *,
    schedule_complete: bool,
    launcher_exit_code: int,
    statuses: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "protocol_id": adjudicator.PROTOCOL_ID,
        "run_id": RUN_ID,
        "litmus_id": "collective_same_generation_reuse",
        "world_size": 2,
        "schedule_digest": SCHEDULE_DIGEST,
        "timed_out": False,
        "schedule_complete": schedule_complete,
        "launcher_exit_code": launcher_exit_code,
        "rank_statuses": statuses,
    }


class AdjudicatorTests(unittest.TestCase):
    def test_valid_same_generation_rows_pass(self) -> None:
        rows = {
            0: [result_row(0, 0, "valid_pass")],
            1: [result_row(1, 0, "valid_pass")],
        }
        summary = adjudicator.adjudicate_evidence(rows)
        self.assertEqual(summary["aggregate_classification"], "valid_pass")
        self.assertEqual(summary["exit_code"], 0)
        self.assertTrue(summary["evidence_valid"])

    def test_schedule_mismatch_is_harness_error_and_has_frozen_priority(self) -> None:
        rows = {
            0: [result_row(0, 0, "protocol_violation", schedule_digest="left")],
            1: [result_row(1, 0, "protocol_violation", schedule_digest="right")],
        }
        summary = adjudicator.adjudicate_evidence(rows)
        self.assertEqual(summary["aggregate_classification"], "harness_error")
        self.assertEqual(summary["exit_code"], 4)
        self.assertIn("evidence disagrees on schedule_digest", summary["issues"])

    def test_stale_generation_negative_control_miss_returns_three(self) -> None:
        litmus_id = "stale_generation_dispatch"
        rows = {
            0: [result_row(0, 1, "negative_control_missed", litmus_id=litmus_id)],
            1: [result_row(1, 1, "negative_control_missed", litmus_id=litmus_id)],
        }
        summary = adjudicator.adjudicate_evidence(rows)
        self.assertEqual(summary["aggregate_classification"], "negative_control_missed")
        self.assertEqual(summary["exit_code"], 3)

    def test_reinitialization_failure_is_capability_result(self) -> None:
        litmus_id = "collective_clean_destroy_recreate"
        rows = {
            0: [
                result_row(
                    0, 0, "reinitialization_capability_failure", litmus_id=litmus_id
                )
            ],
            1: [
                result_row(
                    1, 0, "reinitialization_capability_failure", litmus_id=litmus_id
                )
            ],
        }
        markers = [marker(rank, "ready_destroy") for rank in range(2)]
        summary = adjudicator.adjudicate_evidence(rows, markers)
        self.assertEqual(
            summary["aggregate_classification"],
            "reinitialization_capability_failure",
        )
        self.assertEqual(summary["exit_code"], 5)
        self.assertTrue(summary["evidence_valid"])

    def test_complete_timeout_is_run_level_without_fabricated_rows(self) -> None:
        litmus_id = "collective_clean_destroy_recreate"
        markers = [
            marker(rank, phase, litmus_id=litmus_id)
            for phase in adjudicator.MARKER_PHASES[litmus_id]
            for rank in range(2)
        ]
        launcher = launcher_timeout(litmus_id)
        del launcher["rank_exit_codes"]
        summary = adjudicator.adjudicate_evidence({0: [], 1: []}, markers, launcher)
        self.assertEqual(summary["aggregate_classification"], "capability_timeout")
        self.assertEqual(summary["exit_code"], 5)
        self.assertEqual(summary["rank_row_counts"], {"0": 0, "1": 0})
        self.assertEqual(summary["classifications"], {"capability_timeout": 1})
        self.assertTrue(summary["evidence_valid"])

    def test_incomplete_timeout_is_harness_error(self) -> None:
        summary = adjudicator.adjudicate_evidence(
            {0: [], 1: []}, launcher=launcher_timeout(schedule_complete=False)
        )
        self.assertEqual(summary["aggregate_classification"], "harness_error")
        self.assertEqual(summary["exit_code"], 4)
        self.assertFalse(summary["evidence_valid"])

    def test_generation_cardinality_mismatch_is_harness_error(self) -> None:
        rows = {
            0: [result_row(0, 0, "valid_pass")],
            1: [result_row(1, 1, "valid_pass")],
        }
        summary = adjudicator.adjudicate_evidence(rows)
        self.assertEqual(summary["aggregate_classification"], "harness_error")
        self.assertEqual(summary["exit_code"], 4)

    def test_missing_marker_rank_is_harness_error(self) -> None:
        litmus_id = "collective_clean_destroy_recreate"
        rows = {
            0: [
                result_row(
                    0, 0, "reinitialization_capability_failure", litmus_id=litmus_id
                )
            ],
            1: [
                result_row(
                    1, 0, "reinitialization_capability_failure", litmus_id=litmus_id
                )
            ],
        }
        summary = adjudicator.adjudicate_evidence(rows, [marker(0, "ready_destroy")])
        self.assertEqual(summary["aggregate_classification"], "harness_error")
        self.assertEqual(summary["exit_code"], 4)

    def test_litmus_result_schema_and_state_history_are_validated(self) -> None:
        rows = {
            0: [result_row(0, 0, "valid_pass")],
            1: [result_row(1, 0, "valid_pass")],
        }
        rows[1][0]["final_state"] = "epoch_open"
        summary = adjudicator.adjudicate_evidence(rows)
        self.assertEqual(summary["aggregate_classification"], "harness_error")
        self.assertTrue(
            any("invalid LitmusResult" in issue for issue in summary["issues"])
        )

    def test_phase_to_row_classification_table_is_exact(self) -> None:
        rows = {
            0: [result_row(0, 0, "capability_pass")],
            1: [result_row(1, 0, "capability_pass")],
        }
        summary = adjudicator.adjudicate_evidence(rows)
        self.assertEqual(summary["aggregate_classification"], "harness_error")
        self.assertIn(
            "rows do not match the frozen phase-to-row table", summary["issues"]
        )

    def test_success_epoch_ledger_and_payload_digests_are_validated(self) -> None:
        rows = {
            0: [result_row(0, 0, "valid_pass")],
            1: [result_row(1, 0, "valid_pass")],
        }
        observations = rows[1][0]["observations"]
        assert isinstance(observations, dict)
        observations["completion_count"] = 127
        observations["observed_payload_digest"] = "wrong-digest"
        summary = adjudicator.adjudicate_evidence(rows)
        self.assertEqual(summary["aggregate_classification"], "harness_error")
        self.assertTrue(any("completion_count" in issue for issue in summary["issues"]))
        self.assertTrue(any("payload digest" in issue for issue in summary["issues"]))

    def test_each_success_epoch_obeys_frozen_generation_and_payload_formula(
        self,
    ) -> None:
        rows = {
            0: [result_row(0, 0, "valid_pass")],
            1: [result_row(1, 0, "valid_pass")],
        }
        observations = rows[1][0]["observations"]
        assert isinstance(observations, dict)
        ledger = observations["epoch_ledger"]
        assert isinstance(ledger, list)
        entry = ledger[3]
        assert isinstance(entry, dict)
        entry.update(
            {
                "generation": 1,
                "epoch": 99,
                "completion_count": 2,
                "expected_payload_digest": "wrong",
                "observed_payload_digest": "also-wrong",
            }
        )
        summary = adjudicator.adjudicate_evidence(rows)
        self.assertEqual(summary["aggregate_classification"], "harness_error")
        joined = "\n".join(summary["issues"])
        self.assertIn("generation sequence", joined)
        self.assertIn("epoch sequence", joined)
        self.assertIn("completion_count", joined)
        self.assertIn("expected_payload_digest", joined)
        self.assertIn("observed_payload_digest", joined)

    def test_cross_rank_frozen_collective_fields_are_validated(self) -> None:
        rows = {
            0: [result_row(0, 0, "valid_pass")],
            1: [result_row(1, 0, "valid_pass")],
        }
        observations = rows[1][0]["observations"]
        assert isinstance(observations, dict)
        observations.update(
            {
                "collective_backend": "hccl",
                "source_revision": "different",
                "tensor_shape": [8],
                "dtype": "float32",
                "reduce_op": "max",
            }
        )
        summary = adjudicator.adjudicate_evidence(rows)
        self.assertEqual(summary["aggregate_classification"], "harness_error")
        joined = "\n".join(summary["issues"])
        self.assertIn("collective_backend", joined)
        self.assertIn("source_revision", joined)
        self.assertIn("tensor_shape", joined)
        self.assertIn("dtype", joined)
        self.assertIn("reduce_op", joined)

    def test_launcher_cleanup_harness_overrides_existing_pass_rows(self) -> None:
        rows = {
            0: [result_row(0, 0, "valid_pass")],
            1: [result_row(1, 0, "valid_pass")],
        }
        launcher = {
            "protocol_id": adjudicator.PROTOCOL_ID,
            "run_id": RUN_ID,
            "litmus_id": "collective_same_generation_reuse",
            "world_size": 2,
            "schedule_digest": SCHEDULE_DIGEST,
            "schedule_complete": True,
            "rank_exit_codes": {"0": 0, "1": 4},
            "rank_statuses": [
                rank_status(0, "valid_pass"),
                rank_status(1, "harness_error"),
            ],
        }
        summary = adjudicator.adjudicate_evidence(rows, launcher=launcher)
        self.assertEqual(summary["aggregate_classification"], "harness_error")
        self.assertEqual(summary["exit_code"], 4)

    def test_nonzero_rank_exit_without_status_is_harness_error(self) -> None:
        rows = {
            0: [result_row(0, 0, "valid_pass")],
            1: [result_row(1, 0, "valid_pass")],
        }
        launcher = {
            "protocol_id": adjudicator.PROTOCOL_ID,
            "run_id": RUN_ID,
            "litmus_id": "collective_same_generation_reuse",
            "world_size": 2,
            "schedule_digest": SCHEDULE_DIGEST,
            "schedule_complete": True,
            "rank_exit_codes": {"0": 0, "1": 4},
            "rank_statuses": [rank_status(0, "valid_pass")],
        }
        summary = adjudicator.adjudicate_evidence(rows, launcher=launcher)
        self.assertEqual(summary["aggregate_classification"], "harness_error")
        self.assertTrue(
            any("nonzero rank 1 exit" in issue for issue in summary["issues"])
        )

    def test_non_timeout_rows_still_require_both_rank_files(self) -> None:
        rows = {
            0: [result_row(0, 0, "valid_pass")],
            1: [result_row(1, 0, "valid_pass")],
        }
        summary = adjudicator.adjudicate_evidence(
            rows, rank_file_presence={0: True, 1: False}
        )
        self.assertEqual(summary["aggregate_classification"], "harness_error")
        self.assertTrue(any("both rank JSONL" in issue for issue in summary["issues"]))

    def test_timeout_marker_final_phase_must_exist_for_both_ranks(self) -> None:
        litmus_id = "collective_clean_destroy_recreate"
        markers = [
            marker(rank, phase, litmus_id=litmus_id)
            for phase in ("ready_destroy", "destroyed")
            for rank in range(2)
        ]
        summary = adjudicator.adjudicate_evidence(
            {0: [], 1: []}, markers, launcher_timeout(litmus_id)
        )
        self.assertEqual(summary["aggregate_classification"], "harness_error")
        self.assertTrue(
            any("final marker phase" in issue for issue in summary["issues"])
        )

    def test_partial_device_work_timeout_accepts_complete_fault_ready_markers(
        self,
    ) -> None:
        litmus_id = "collective_partial_epoch_timeout_recreate"
        launcher = launcher_timeout(litmus_id)
        del launcher["rank_exit_codes"]
        launcher["final_phase"] = "device_work"
        markers = [
            marker(rank, "fault_ready", litmus_id=litmus_id) for rank in range(2)
        ]
        summary = adjudicator.adjudicate_evidence({0: [], 1: []}, markers, launcher)
        self.assertEqual(summary["aggregate_classification"], "capability_timeout")
        self.assertTrue(summary["evidence_valid"])

    def test_non_timeout_launcher_requires_complete_schedule(self) -> None:
        rows = {
            0: [result_row(0, 0, "valid_pass")],
            1: [result_row(1, 0, "valid_pass")],
        }
        launcher = launcher_non_timeout(
            schedule_complete=False,
            launcher_exit_code=0,
            statuses=[rank_status(0, "valid_pass"), rank_status(1, "valid_pass")],
        )
        summary = adjudicator.adjudicate_evidence(rows, launcher=launcher)
        self.assertEqual(summary["aggregate_classification"], "harness_error")
        self.assertTrue(
            any("schedule_complete=true" in issue for issue in summary["issues"])
        )

    def test_nonzero_launcher_exit_requires_both_rank_statuses(self) -> None:
        rows = {
            0: [result_row(0, 0, "valid_pass")],
            1: [result_row(1, 0, "valid_pass")],
        }
        incomplete = launcher_non_timeout(
            schedule_complete=True,
            launcher_exit_code=1,
            statuses=[rank_status(0, "valid_pass")],
        )
        summary = adjudicator.adjudicate_evidence(rows, launcher=incomplete)
        self.assertEqual(summary["aggregate_classification"], "harness_error")

        complete = launcher_non_timeout(
            schedule_complete=True,
            launcher_exit_code=1,
            statuses=[rank_status(0, "valid_pass"), rank_status(1, "valid_pass")],
        )
        summary = adjudicator.adjudicate_evidence(rows, launcher=complete)
        self.assertEqual(summary["aggregate_classification"], "valid_pass")
        self.assertTrue(summary["evidence_valid"])

    def test_normal_pass_is_bound_to_reusable_final_state(self) -> None:
        rows = {
            0: [result_row(0, 0, "valid_pass")],
            1: [result_row(1, 0, "valid_pass")],
        }
        transitions, final_state = transition_rows(
            [
                "enqueued",
                "completed",
                "reusable_same_generation",
                "retiring",
                "destroyed",
                "recreated",
            ]
        )
        rows[1][0]["transitions"] = transitions
        rows[1][0]["final_state"] = final_state
        summary = adjudicator.adjudicate_evidence(rows)
        self.assertEqual(summary["aggregate_classification"], "harness_error")
        self.assertTrue(
            any("final_state/phase" in issue for issue in summary["issues"])
        )

    def test_clean_capability_pass_binds_g0_and_g1_final_states(self) -> None:
        litmus_id = "collective_clean_destroy_recreate"
        rows = {
            rank: [
                result_row(rank, 0, "capability_pass", litmus_id=litmus_id),
                result_row(rank, 1, "capability_pass", litmus_id=litmus_id),
            ]
            for rank in range(2)
        }
        markers = [
            marker(rank, phase, litmus_id=litmus_id)
            for phase in adjudicator.MARKER_PHASES[litmus_id]
            for rank in range(2)
        ]
        summary = adjudicator.adjudicate_evidence(rows, markers)
        self.assertEqual(summary["aggregate_classification"], "capability_pass")
        self.assertTrue(summary["evidence_valid"])

        wrong = copy.deepcopy(rows)
        transitions = wrong[1][0]["transitions"]
        assert isinstance(transitions, list)
        transitions.pop()
        wrong[1][0]["final_state"] = "destroyed"
        summary = adjudicator.adjudicate_evidence(wrong, markers)
        self.assertEqual(summary["aggregate_classification"], "harness_error")

    def test_expected_timeout_recovery_binds_fault_roles_and_true_markers(
        self,
    ) -> None:
        litmus_id = "collective_partial_epoch_timeout_recreate"
        rows = {
            rank: [
                result_row(
                    rank,
                    0,
                    "expected_timeout_recovered",
                    litmus_id=litmus_id,
                ),
                result_row(rank, 1, "capability_pass", litmus_id=litmus_id),
            ]
            for rank in range(2)
        }
        markers = []
        for phase in adjudicator.MARKER_PHASES[litmus_id]:
            for rank in range(2):
                extra: dict[str, object] = {}
                if phase == "fault_observed":
                    extra = {
                        "fault_observed": True,
                        "local_enqueued": rank == 0,
                        "work_handle_present": rank == 0,
                        "rank_0_work_handle_present": True,
                    }
                markers.append(marker(rank, phase, litmus_id=litmus_id, **extra))
        summary = adjudicator.adjudicate_evidence(rows, markers)
        self.assertEqual(
            summary["aggregate_classification"], "expected_timeout_recovered"
        )
        self.assertTrue(summary["evidence_valid"])

        wrong_role = copy.deepcopy(rows)
        observations = wrong_role[1][0]["observations"]
        assert isinstance(observations, dict)
        observations["local_enqueue_role"] = "submit_fault"
        summary = adjudicator.adjudicate_evidence(wrong_role, markers)
        self.assertEqual(summary["aggregate_classification"], "harness_error")

        wrong_markers = copy.deepcopy(markers)
        fault_marker = next(
            item
            for item in wrong_markers
            if item["phase"] == "fault_observed" and item["rank"] == 1
        )
        fault_marker["fault_observed"] = False
        summary = adjudicator.adjudicate_evidence(rows, wrong_markers)
        self.assertEqual(summary["aggregate_classification"], "harness_error")

    def test_partial_inconclusive_uses_fault_observed_false(self) -> None:
        litmus_id = "collective_partial_epoch_timeout_recreate"
        rows = {
            0: [result_row(0, 0, "inconclusive", litmus_id=litmus_id)],
            1: [result_row(1, 0, "inconclusive", litmus_id=litmus_id)],
        }
        markers = [
            marker(rank, "fault_ready", litmus_id=litmus_id) for rank in range(2)
        ] + [
            marker(
                rank,
                "fault_observed",
                litmus_id=litmus_id,
                fault_observed=False,
            )
            for rank in range(2)
        ]
        summary = adjudicator.adjudicate_evidence(rows, markers)
        self.assertEqual(summary["aggregate_classification"], "inconclusive")
        self.assertEqual(summary["exit_code"], 5)
        self.assertTrue(summary["evidence_valid"])

    def test_cli_timeout_can_have_no_rank_jsonl_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = launcher_timeout()
            launcher["final_phase"] = "device_work"
            launcher_path = root / "launcher.json"
            launcher_path.write_text(json.dumps(launcher), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                return_code = adjudicator.main(
                    ["--launcher-evidence", str(launcher_path)]
                )
            self.assertEqual(return_code, 5)

    def test_cli_writes_summary_once_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rank_paths = [root / f"rank-{rank:04d}.jsonl" for rank in range(2)]
            for rank, path in enumerate(rank_paths):
                path.write_text(
                    json.dumps(result_row(rank, 0, "valid_pass")) + "\n",
                    encoding="utf-8",
                )
            output = root / "summary.json"
            arguments = [
                "--rank-jsonl",
                str(rank_paths[0]),
                "--rank-jsonl",
                str(rank_paths[1]),
                "--output",
                str(output),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(adjudicator.main(arguments), 0)
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["aggregate_classification"], "valid_pass")
            with self.assertRaises(FileExistsError):
                adjudicator.main(arguments)


if __name__ == "__main__":
    unittest.main()
