#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import replace
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

DEFAULT_ITERATIONS = 128
DEFAULT_SEED = 20260829
POSITIVE_LITMUS = (
    "event_rerecord_single_waiter",
    "event_multi_waiter",
    "buffer_reuse_with_ready_and_consumed_events",
    "graph_static_input_generations",
    "graph_cross_stream_join",
    "capture_abort_eager_recovery",
)
NEGATIVE_CONTROLS = ("missing_join", "rebound_input")
LITMUS_ORDER = POSITIVE_LITMUS + NEGATIVE_CONTROLS
RESOURCE_BY_LITMUS = {
    "event_rerecord_single_waiter": ResourceKind.BUFFER,
    "event_multi_waiter": ResourceKind.BUFFER,
    "buffer_reuse_with_ready_and_consumed_events": ResourceKind.BUFFER,
    "graph_static_input_generations": ResourceKind.GRAPH,
    "graph_cross_stream_join": ResourceKind.GRAPH,
    "capture_abort_eager_recovery": ResourceKind.GRAPH,
    "missing_join": ResourceKind.GRAPH,
    "rebound_input": ResourceKind.GRAPH,
}


class UnsupportedFeature(RuntimeError):
    pass


class TorchDeviceBackend:
    """Small CUDA/NPU spelling adapter; it does not hide lifecycle semantics."""

    def __init__(self, torch: Any, name: str, device_index: int):
        if name not in {"cuda", "npu"}:
            raise ValueError(f"unsupported backend name: {name}")
        self.torch = torch
        self.name = name
        self.api = getattr(torch, name)
        self.device_index = device_index
        self.api.set_device(device_index)
        self.device = torch.device(f"{name}:{device_index}")

    def synchronize(self) -> None:
        self.api.synchronize()

    def stream(self) -> Any:
        return self.api.Stream(device=self.device_index)

    def current_stream(self) -> Any:
        return self.api.current_stream(device=self.device_index)

    def stream_context(self, stream: Any) -> Any:
        return self.api.stream(stream)

    def event(self, *, external: bool = False) -> Any:
        if external:
            event_type = getattr(self.api, "ExternalEvent", None)
            if event_type is None:
                raise UnsupportedFeature(f"{self.name} ExternalEvent is unavailable")
            return event_type()
        return self.api.Event(enable_timing=False)

    def capture_event(self) -> Any:
        return self.event(external=self.name == "npu")

    def reset_external_event(self, event: Any, stream: Any) -> None:
        reset = getattr(event, "reset", None)
        if not callable(reset):
            raise UnsupportedFeature(f"{self.name} event reset is unavailable")
        reset(stream)

    def new_graph(self) -> Any:
        type_name = "CUDAGraph" if self.name == "cuda" else "NPUGraph"
        graph_type = getattr(self.api, type_name, None)
        if graph_type is None:
            raise UnsupportedFeature(f"{self.name} graph capture is unavailable")
        return graph_type()

    def graph_context(self, graph: Any, stream: Any) -> Any:
        context = getattr(self.api, "graph", None)
        if context is None:
            raise UnsupportedFeature(f"{self.name} graph context is unavailable")
        return context(graph, stream=stream)

    def is_capturing(self) -> bool | None:
        query = getattr(self.api, "is_current_stream_capturing", None)
        return bool(query()) if callable(query) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AccelPact AP-G0Q TP1 litmus")
    parser.add_argument("--backend", choices=("auto", "cuda", "npu"), default="auto")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--litmus",
        action="append",
        choices=LITMUS_ORDER,
        help="run only this litmus; repeat to select multiple litmus",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.device_index < 0:
        parser.error("--device-index must be non-negative")
    return args


def resolve_backend(
    requested: str, device_index: int
) -> tuple[TorchDeviceBackend | None, str | None]:
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        return None, f"PyTorch is unavailable: {error}"

    candidates = ("cuda", "npu") if requested == "auto" else (requested,)
    reasons = []
    for name in candidates:
        if name == "npu":
            try:
                importlib.import_module("torch_npu")
            except ImportError as error:
                reasons.append(f"torch_npu is unavailable: {error}")
                continue
        api = getattr(torch, name, None)
        available = getattr(api, "is_available", None) if api is not None else None
        try:
            if not callable(available) or not available():
                reasons.append(f"{name}:{device_index} is unavailable")
                continue
            if device_index >= int(api.device_count()):
                reasons.append(
                    f"{name}:{device_index} is outside the visible device set"
                )
                continue
            return TorchDeviceBackend(torch, name, device_index), None
        except Exception as error:  # noqa: BLE001 - report backend qualification failure
            reasons.append(
                f"{name} qualification failed: {type(error).__name__}: {error}"
            )
    return None, "; ".join(reasons) or "no accelerator backend is available"


def set_seed(backend: TorchDeviceBackend, seed: int) -> None:
    backend.torch.manual_seed(seed)
    manual_seed_all = getattr(backend.api, "manual_seed_all", None)
    if callable(manual_seed_all):
        manual_seed_all(seed)


def complete_buffer_oracle(subject_id: str) -> LifecycleOracle:
    oracle = LifecycleOracle(ResourceKind.BUFFER, subject_id)
    for state in (
        "write_pending",
        "ready",
        "published",
        "consumed",
        "reclaimable",
    ):
        oracle.transition(state)
    return oracle


def complete_graph_oracle(subject_id: str) -> LifecycleOracle:
    oracle = LifecycleOracle(ResourceKind.GRAPH, subject_id)
    for state in ("capturing", "committed", "replayable"):
        oracle.transition(state)
    return oracle


def make_result(
    oracle: LifecycleOracle,
    *,
    litmus_id: str,
    backend: str,
    classification: str,
    observations: dict[str, Any] | None = None,
    error: str | None = None,
) -> LitmusResult:
    is_failure = classification in {"protocol_violation", "harness_error"}
    values = {"classification": classification, **(observations or {})}
    result_error = error or (
        f"{classification}: see observations" if is_failure else None
    )
    return LitmusResult.from_oracle(
        oracle,
        litmus_id=litmus_id,
        backend=backend,
        passed=not is_failure,
        observations=values,
        error=result_error,
    )


def unsupported_result(litmus_id: str, backend: str, reason: str) -> LitmusResult:
    oracle = LifecycleOracle(RESOURCE_BY_LITMUS[litmus_id], f"{litmus_id}-unsupported")
    return make_result(
        oracle,
        litmus_id=litmus_id,
        backend=backend,
        classification="unsupported",
        observations={"reason": reason},
    )


def failure_result(litmus_id: str, backend: str, error: BaseException) -> LitmusResult:
    oracle = LifecycleOracle(RESOURCE_BY_LITMUS[litmus_id], f"{litmus_id}-failed")
    if litmus_id == "capture_abort_eager_recovery":
        for state in ("capturing", "aborted", "poisoned"):
            oracle.transition(state)
    classification = (
        "protocol_violation" if isinstance(error, AssertionError) else "harness_error"
    )
    return make_result(
        oracle,
        litmus_id=litmus_id,
        backend=backend,
        classification=classification,
        observations={"error_type": type(error).__name__},
        error=f"{type(error).__name__}: {error}",
    )


def _tensor_values(tensor: Any) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().tolist()]


def event_rerecord_single_waiter(
    backend: TorchDeviceBackend, iterations: int, seed: int
) -> LitmusResult:
    del seed
    torch = backend.torch
    producer = backend.stream()
    waiter = backend.stream()
    event = backend.event()
    slot = torch.zeros(1, dtype=torch.int64, device=backend.device)
    observed = torch.empty(iterations, dtype=torch.int64, device=backend.device)
    for generation in range(1, iterations + 1):
        with backend.stream_context(producer):
            slot.fill_(generation)
            event.record(producer)
        with backend.stream_context(waiter):
            waiter.wait_event(event)
            observed[generation - 1].copy_(slot[0])
        waiter.synchronize()
    actual = _tensor_values(observed)
    expected = list(range(1, iterations + 1))
    if actual != expected:
        raise AssertionError("single-waiter event re-record exposed a stale generation")
    return make_result(
        complete_buffer_oracle("event-rerecord-buffer"),
        litmus_id="event_rerecord_single_waiter",
        backend=backend.name,
        classification="valid_pass",
        observations={"iterations": iterations, "event_reset_count": 0},
    )


def event_multi_waiter(
    backend: TorchDeviceBackend, iterations: int, seed: int
) -> LitmusResult:
    del seed
    torch = backend.torch
    producer = backend.stream()
    waiters = (backend.stream(), backend.stream())
    event = backend.event()
    slot = torch.zeros(1, dtype=torch.int64, device=backend.device)
    outputs = [
        torch.empty(iterations, dtype=torch.int64, device=backend.device)
        for _ in waiters
    ]
    reset_count = 0
    for generation in range(1, iterations + 1):
        with backend.stream_context(producer):
            slot.fill_(generation)
            event.record(producer)
        for waiter, output in zip(waiters, outputs, strict=True):
            with backend.stream_context(waiter):
                waiter.wait_event(event)
                output[generation - 1].copy_(slot[0])
        for waiter in waiters:
            waiter.synchronize()
        if backend.name == "npu":
            backend.reset_external_event(event, producer)
            producer.synchronize()
            reset_count += 1
    expected = list(range(1, iterations + 1))
    if any(_tensor_values(output) != expected for output in outputs):
        raise AssertionError("multi-waiter event exposed a stale generation")
    return make_result(
        complete_buffer_oracle("event-multi-waiter-buffer"),
        litmus_id="event_multi_waiter",
        backend=backend.name,
        classification="valid_pass",
        observations={
            "iterations": iterations,
            "waiter_count": len(waiters),
            "event_reset_count": reset_count,
            "reset_after_all_waiters_synchronized": backend.name == "npu",
        },
    )


def buffer_reuse_with_ready_and_consumed_events(
    backend: TorchDeviceBackend, iterations: int, seed: int
) -> LitmusResult:
    del seed
    torch = backend.torch
    producer = backend.stream()
    consumer = backend.stream()
    slot = torch.zeros(1, dtype=torch.int64, device=backend.device)
    observed = torch.empty(iterations, dtype=torch.int64, device=backend.device)
    retained_events = []
    for generation in range(1, iterations + 1):
        ready = backend.event()
        consumed = backend.event()
        retained_events.extend((ready, consumed))
        with backend.stream_context(producer):
            if generation > 1:
                producer.wait_event(retained_events[-3])
            slot.fill_(generation)
            ready.record(producer)
        with backend.stream_context(consumer):
            consumer.wait_event(ready)
            observed[generation - 1].copy_(slot[0])
            consumed.record(consumer)
    consumer.synchronize()
    actual = _tensor_values(observed)
    if actual != list(range(1, iterations + 1)):
        raise AssertionError("ready/consumed protocol allowed premature buffer reuse")
    return make_result(
        complete_buffer_oracle("ready-consumed-buffer"),
        litmus_id="buffer_reuse_with_ready_and_consumed_events",
        backend=backend.name,
        classification="valid_pass",
        observations={
            "iterations": iterations,
            "two_event_handshake": True,
            "fresh_event_pair_per_generation": True,
            "retained_event_count": len(retained_events),
        },
    )


def capture_affine_graph(backend: TorchDeviceBackend) -> tuple[Any, Any, Any]:
    torch = backend.torch
    static_input = torch.zeros(256, dtype=torch.float32, device=backend.device)
    static_output = torch.empty_like(static_input)
    graph = backend.new_graph()
    capture_stream = backend.stream()
    with backend.stream_context(capture_stream):
        static_output.copy_(static_input * 2.0 + 1.0)
    capture_stream.synchronize()
    backend.synchronize()
    with backend.graph_context(graph, capture_stream):
        static_output.copy_(static_input * 2.0 + 1.0)
    return graph, static_input, static_output


def capture_multistream_affine_graph(
    backend: TorchDeviceBackend, *, include_child_to_origin_join: bool
) -> tuple[Any, Any, Any]:
    torch = backend.torch
    static_input = torch.zeros(256, dtype=torch.float32, device=backend.device)
    child_output = torch.empty_like(static_input)
    static_output = torch.empty_like(static_input)
    graph = backend.new_graph()
    origin = backend.stream()
    child = backend.stream()
    origin_to_child = backend.capture_event()
    child_to_origin = backend.capture_event()

    # Compile and initialize the exact operator/stream path before capture.
    with backend.stream_context(origin):
        origin_to_child.record(origin)
    with backend.stream_context(child):
        child.wait_event(origin_to_child)
        child_output.copy_(static_input * 2.0)
        child_to_origin.record(child)
    origin.wait_event(child_to_origin)
    backend.synchronize()
    with backend.graph_context(graph, origin):
        origin_to_child.record(origin)
        with backend.stream_context(child):
            child.wait_event(origin_to_child)
            child_output.copy_(static_input * 2.0)
            if include_child_to_origin_join:
                child_to_origin.record(child)
        if include_child_to_origin_join:
            origin.wait_event(child_to_origin)
            static_output.copy_(child_output + 1.0)
    return graph, static_input, static_output


def graph_static_input_generations(
    backend: TorchDeviceBackend, iterations: int, seed: int
) -> LitmusResult:
    del seed
    torch = backend.torch
    graph, static_input, static_output = capture_affine_graph(backend)
    captured_pointer = int(static_input.data_ptr())
    for generation in range(1, iterations + 1):
        static_input.fill_(float(generation))
        graph.replay()
        backend.synchronize()
        expected = torch.full_like(static_output, generation * 2.0 + 1.0)
        if not bool(torch.equal(static_output, expected)):
            raise AssertionError("graph replay returned the wrong input generation")
        if int(static_input.data_ptr()) != captured_pointer:
            raise AssertionError("static graph input storage was rebound")
    return make_result(
        complete_graph_oracle("static-input-graph"),
        litmus_id="graph_static_input_generations",
        backend=backend.name,
        classification="valid_pass",
        observations={
            "iterations": iterations,
            "captured_input_pointer_stable": True,
        },
    )


def graph_cross_stream_join(
    backend: TorchDeviceBackend, iterations: int, seed: int
) -> LitmusResult:
    del seed
    torch = backend.torch
    try:
        graph, static_input, static_output = capture_multistream_affine_graph(
            backend, include_child_to_origin_join=True
        )
    except Exception as error:
        message = str(error).lower()
        if "unsupported" in message or "not support" in message:
            raise UnsupportedFeature(str(error)) from error
        raise
    for generation in range(1, iterations + 1):
        static_input.fill_(float(generation))
        graph.replay()
        backend.synchronize()
        expected = torch.full_like(static_output, generation * 2.0 + 1.0)
        if not bool(torch.equal(static_output, expected)):
            raise AssertionError("cross-stream graph join exposed stale output")
    return make_result(
        complete_graph_oracle("cross-stream-graph"),
        litmus_id="graph_cross_stream_join",
        backend=backend.name,
        classification="valid_pass",
        observations={"iterations": iterations, "explicit_join": True},
    )


def capture_abort_eager_recovery(
    backend: TorchDeviceBackend, iterations: int, seed: int
) -> LitmusResult:
    del iterations
    torch = backend.torch
    oracle = LifecycleOracle(ResourceKind.GRAPH, "capture-abort-graph")
    oracle.transition("capturing")
    graph = backend.new_graph()
    capture_stream = backend.stream()
    rng_enqueued = False
    capture_error: BaseException | None = None
    try:
        with backend.graph_context(graph, capture_stream):
            random_values = torch.rand(256, device=backend.device)
            rng_enqueued = True
            random_values.sum().item()
    except Exception as error:  # noqa: BLE001 - this is the expected capture abort
        capture_error = error

    if capture_error is None:
        oracle.transition("committed")
        oracle.transition("replayable")
        return make_result(
            oracle,
            litmus_id="capture_abort_eager_recovery",
            backend=backend.name,
            classification="unsupported",
            observations={
                "rng_enqueued_before_item": rng_enqueued,
                "reason": "Tensor.item() did not invalidate capture",
            },
        )

    oracle.transition("aborted")
    recovery: dict[str, Any] = {
        "rng_enqueued_before_item": rng_enqueued,
        "capture_error_type": type(capture_error).__name__,
        "capture_error": str(capture_error),
    }
    recovery_error: BaseException | None = None
    try:
        recovery["graph_reset_before_primary_recovery"] = False
        recovery["capturing_after_abort"] = backend.is_capturing()
        if recovery["capturing_after_abort"] is True:
            raise RuntimeError("stream remained in capture mode after abort")
        backend.synchronize()

        set_seed(backend, seed + 1)
        eager_random = torch.rand(256, device=backend.device)
        allocator_probe = torch.empty(4096, dtype=torch.float32, device=backend.device)
        test_stream = backend.stream()
        with backend.stream_context(test_stream):
            allocator_probe.fill_(7.0)
        test_stream.synchronize()
        backend.synchronize()
        recovery["eager_rng_finite"] = bool(torch.isfinite(eager_random).all().item())
        recovery["allocator_probe_ok"] = bool((allocator_probe == 7.0).all().item())
        recovery["new_stream_ok"] = True
        if not recovery["eager_rng_finite"] or not recovery["allocator_probe_ok"]:
            raise RuntimeError("post-abort eager probe returned invalid values")
    except Exception as error:  # noqa: BLE001 - preserve poisoned-runtime evidence
        recovery_error = error
        recovery["recovery_error_type"] = type(error).__name__
        recovery["recovery_error"] = str(error)

    reset = getattr(graph, "reset", None)
    if recovery_error is None:
        try:
            if callable(reset):
                reset()
                recovery["post_recovery_graph_cleanup"] = "reset_ok"
            else:
                recovery["post_recovery_graph_cleanup"] = "reset_unavailable"
        except Exception as error:  # noqa: BLE001 - cleanup remains diagnostic
            recovery["post_recovery_graph_cleanup"] = (
                f"reset_failed: {type(error).__name__}: {error}"
            )
    else:
        recovery["diagnostic_reset_attempted"] = callable(reset)
        if callable(reset):
            try:
                reset()
                backend.synchronize()
                diagnostic = torch.rand(16, device=backend.device)
                backend.synchronize()
                recovery["diagnostic_reset_retry_rng_finite"] = bool(
                    torch.isfinite(diagnostic).all().item()
                )
            except Exception as error:  # noqa: BLE001 - retain diagnostic only
                recovery["diagnostic_reset_retry_error"] = (
                    f"{type(error).__name__}: {error}"
                )

    if not rng_enqueued and recovery_error is None:
        oracle.transition("clean_fallback")
        return make_result(
            oracle,
            litmus_id="capture_abort_eager_recovery",
            backend=backend.name,
            classification="unsupported",
            observations={**recovery, "reason": "RNG itself was not captureable"},
        )
    if recovery_error is None:
        oracle.transition("clean_fallback")
        return make_result(
            oracle,
            litmus_id="capture_abort_eager_recovery",
            backend=backend.name,
            classification="valid_pass",
            observations=recovery,
        )
    oracle.transition("poisoned")
    return make_result(
        oracle,
        litmus_id="capture_abort_eager_recovery",
        backend=backend.name,
        classification="protocol_violation",
        observations=recovery,
        error=(
            "post-abort recovery failed: "
            f"{type(recovery_error).__name__}: {recovery_error}"
        ),
    )


def missing_join(
    backend: TorchDeviceBackend, iterations: int, seed: int
) -> LitmusResult:
    del iterations, seed
    oracle = LifecycleOracle(ResourceKind.GRAPH, "missing-join-negative-control")
    oracle.transition("capturing")
    capture_error: BaseException | None = None
    try:
        capture_multistream_affine_graph(backend, include_child_to_origin_join=False)
    except Exception as error:  # noqa: BLE001 - expected negative-control rejection
        capture_error = error

    detected = capture_error is not None
    cleanup_error = None
    if detected:
        oracle.transition("aborted")
        try:
            backend.synchronize()
            oracle.transition("clean_fallback")
        except Exception as error:  # noqa: BLE001 - harness recovery failure
            cleanup_error = f"{type(error).__name__}: {error}"
            oracle.transition("poisoned")
    else:
        oracle.transition("committed")
        oracle.transition("replayable")
    classification = (
        "negative_control_detected" if detected else "negative_control_missed"
    )
    if cleanup_error is not None:
        classification = "harness_error"
    return make_result(
        oracle,
        litmus_id="missing_join",
        backend=backend.name,
        classification=classification,
        observations={
            "negative_control": True,
            "runtime_violation": False,
            "same_multistream_lowering_without_child_to_origin_join": True,
            "capture_end_rejected_missing_join": detected,
            "capture_error": str(capture_error) if capture_error is not None else None,
            "cleanup_error": cleanup_error,
        },
    )


def rebound_input(
    backend: TorchDeviceBackend, iterations: int, seed: int
) -> LitmusResult:
    del iterations, seed
    torch = backend.torch
    graph, static_input, static_output = capture_affine_graph(backend)
    captured_pointer = int(static_input.data_ptr())
    rebound = torch.full_like(static_input, 9.0)
    rebound_pointer = int(rebound.data_ptr())
    graph.replay()
    backend.synchronize()
    expected_from_rebound = rebound * 2.0 + 1.0
    stale_generation_detected = not bool(
        torch.equal(static_output, expected_from_rebound)
    )
    pointer_rebound_detected = rebound_pointer != captured_pointer
    detected = stale_generation_detected and pointer_rebound_detected
    return make_result(
        complete_graph_oracle("rebound-input-negative-control"),
        litmus_id="rebound_input",
        backend=backend.name,
        classification=(
            "negative_control_detected" if detected else "negative_control_missed"
        ),
        observations={
            "negative_control": True,
            "runtime_violation": False,
            "captured_pointer": captured_pointer,
            "rebound_pointer": rebound_pointer,
            "stale_generation_detected": stale_generation_detected,
        },
    )


LITMUS_FUNCTIONS: dict[str, Callable[[TorchDeviceBackend, int, int], LitmusResult]] = {
    "event_rerecord_single_waiter": event_rerecord_single_waiter,
    "event_multi_waiter": event_multi_waiter,
    "buffer_reuse_with_ready_and_consumed_events": (
        buffer_reuse_with_ready_and_consumed_events
    ),
    "graph_static_input_generations": graph_static_input_generations,
    "graph_cross_stream_join": graph_cross_stream_join,
    "capture_abort_eager_recovery": capture_abort_eager_recovery,
    "missing_join": missing_join,
    "rebound_input": rebound_input,
}


def measured_result(
    litmus_id: str,
    backend: TorchDeviceBackend,
    iterations: int,
    seed: int,
) -> LitmusResult:
    is_negative = litmus_id in NEGATIVE_CONTROLS
    try:
        backend.synchronize()
        started_ns = time.perf_counter_ns()
        result = LITMUS_FUNCTIONS[litmus_id](backend, iterations, seed)
        backend.synchronize()
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        return replace(
            result,
            observations={
                **dict(result.observations),
                "measurement_boundary": "device_synchronize_before_and_after",
                "wall_ms": elapsed_ms,
            },
        )
    except UnsupportedFeature as error:
        return unsupported_result(litmus_id, backend.name, str(error))
    except Exception as error:  # noqa: BLE001 - every litmus must retain a result
        if is_negative:
            oracle = LifecycleOracle(
                RESOURCE_BY_LITMUS[litmus_id], f"{litmus_id}-control-error"
            )
            return make_result(
                oracle,
                litmus_id=litmus_id,
                backend=backend.name,
                classification="harness_error",
                observations={
                    "negative_control": True,
                    "runtime_violation": False,
                    "control_error_type": type(error).__name__,
                    "control_error": str(error),
                },
            )
        return failure_result(litmus_id, backend.name, error)


def summarize(results: list[LitmusResult]) -> dict[str, Any]:
    classifications = [str(result.observations["classification"]) for result in results]
    protocol_failures = [
        result.litmus_id
        for result in results
        if result.observations["classification"] == "protocol_violation"
    ]
    negative_misses = [
        result.litmus_id
        for result in results
        if result.litmus_id in NEGATIVE_CONTROLS
        and result.observations["classification"]
        in {"negative_control_missed", "harness_error"}
    ]
    harness_errors = [
        result.litmus_id
        for result in results
        if result.observations["classification"] == "harness_error"
    ]
    return {
        "result_count": len(results),
        "classifications": {
            name: classifications.count(name) for name in sorted(set(classifications))
        },
        "protocol_failures": protocol_failures,
        "harness_errors": harness_errors,
        "negative_control_misses": negative_misses,
        "runtime_protocol_pass": not protocol_failures and not harness_errors,
        "oracle_sensitivity_pass": not negative_misses,
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    backend, unavailable_reason = resolve_backend(args.backend, args.device_index)
    results: list[LitmusResult] = []
    selected_litmus = tuple(args.litmus) if args.litmus else LITMUS_ORDER
    if backend is None:
        backend_name = args.backend if args.backend != "auto" else "unavailable"
        for litmus_id in selected_litmus:
            result = unsupported_result(
                litmus_id, backend_name, unavailable_reason or "backend unavailable"
            )
            append_result(args.output, result)
            results.append(result)
    else:
        set_seed(backend, args.seed)
        for litmus_id in selected_litmus:
            result = measured_result(litmus_id, backend, args.iterations, args.seed)
            append_result(args.output, result)
            results.append(result)

    summary = summarize(results)
    print(json.dumps(summary, sort_keys=True))
    if summary["protocol_failures"]:
        return 2
    if summary["harness_errors"]:
        return 4
    if summary["negative_control_misses"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
