from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar


class ResourceKind(str, Enum):
    BUFFER = "buffer"
    GRAPH = "graph"
    COLLECTIVE = "collective"


_INITIAL_STATE = {
    ResourceKind.BUFFER: "owned",
    ResourceKind.GRAPH: "idle",
    ResourceKind.COLLECTIVE: "epoch_open",
}

_ALLOWED_TRANSITIONS = {
    ResourceKind.BUFFER: {
        "owned": frozenset({"write_pending"}),
        "write_pending": frozenset({"ready"}),
        "ready": frozenset({"published"}),
        "published": frozenset({"consumed"}),
        "consumed": frozenset({"reclaimable"}),
        "reclaimable": frozenset(),
    },
    ResourceKind.GRAPH: {
        "idle": frozenset({"capturing"}),
        "capturing": frozenset({"committed", "aborted"}),
        "committed": frozenset({"replayable"}),
        "aborted": frozenset({"clean_fallback", "poisoned"}),
        "replayable": frozenset(),
        "clean_fallback": frozenset(),
        "poisoned": frozenset(),
    },
    ResourceKind.COLLECTIVE: {
        "epoch_open": frozenset({"enqueued"}),
        "enqueued": frozenset({"completed", "failed_unknown"}),
        "completed": frozenset({"reusable_same_generation"}),
        "failed_unknown": frozenset({"aborting"}),
        "aborting": frozenset({"destroyed"}),
        "destroyed": frozenset({"recreated"}),
        "reusable_same_generation": frozenset(),
        "recreated": frozenset(),
    },
}


class InvalidTransition(ValueError):
    def __init__(self, resource: ResourceKind, source: str, target: str):
        self.resource = resource
        self.source = source
        self.target = target
        super().__init__(f"illegal {resource.value} transition: {source} -> {target}")


@dataclass(frozen=True)
class Transition:
    index: int
    source: str
    target: str

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "source": self.source, "target": self.target}


class LifecycleOracle:
    """Exact lifecycle oracle for the three AccelPact Gate-0 resources."""

    def __init__(self, resource: ResourceKind | str, subject_id: str):
        self.resource = ResourceKind(resource)
        if not subject_id:
            raise ValueError("subject_id must be non-empty")
        self.subject_id = subject_id
        self.initial_state = _INITIAL_STATE[self.resource]
        self.state = self.initial_state
        self._history: list[Transition] = []

    @property
    def history(self) -> tuple[Transition, ...]:
        return tuple(self._history)

    @property
    def allowed_targets(self) -> frozenset[str]:
        return _ALLOWED_TRANSITIONS[self.resource][self.state]

    def transition(self, target: str) -> Transition:
        if target not in self.allowed_targets:
            raise InvalidTransition(self.resource, self.state, target)
        event = Transition(len(self._history), self.state, target)
        self.state = target
        self._history.append(event)
        return event


def _validated_history(
    resource: ResourceKind, initial_state: str, transitions: tuple[Transition, ...]
) -> str:
    if initial_state != _INITIAL_STATE[resource]:
        raise ValueError(f"invalid initial state for {resource.value}: {initial_state}")
    current = initial_state
    for expected_index, event in enumerate(transitions):
        if event.index != expected_index:
            raise ValueError("transition indices must be contiguous from zero")
        if event.source != current:
            raise ValueError("transition history is not contiguous")
        if event.target not in _ALLOWED_TRANSITIONS[resource][current]:
            raise InvalidTransition(resource, current, event.target)
        current = event.target
    return current


@dataclass(frozen=True)
class LitmusResult:
    litmus_id: str
    backend: str
    subject_id: str
    resource: ResourceKind
    passed: bool
    initial_state: str
    final_state: str
    transitions: tuple[Transition, ...]
    observations: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    SCHEMA: ClassVar[str] = "accelpact.stateful_litmus_result.v1"

    def __post_init__(self) -> None:
        if not self.litmus_id or not self.backend or not self.subject_id:
            raise ValueError("litmus_id, backend, and subject_id must be non-empty")
        resource = ResourceKind(self.resource)
        object.__setattr__(self, "resource", resource)
        transitions = tuple(self.transitions)
        object.__setattr__(self, "transitions", transitions)
        expected_final = _validated_history(resource, self.initial_state, transitions)
        if self.final_state != expected_final:
            raise ValueError("final_state does not match transition history")
        if self.passed and self.error is not None:
            raise ValueError("a passed result cannot contain an error")
        if not self.passed and not self.error:
            raise ValueError("a failed result must contain an error")
        encoded_observations = json.dumps(
            dict(self.observations), allow_nan=False, sort_keys=True
        )
        object.__setattr__(self, "observations", json.loads(encoded_observations))

    @classmethod
    def from_oracle(
        cls,
        oracle: LifecycleOracle,
        *,
        litmus_id: str,
        backend: str,
        passed: bool,
        observations: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> LitmusResult:
        return cls(
            litmus_id=litmus_id,
            backend=backend,
            subject_id=oracle.subject_id,
            resource=oracle.resource,
            passed=passed,
            initial_state=oracle.initial_state,
            final_state=oracle.state,
            transitions=oracle.history,
            observations=observations or {},
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "litmus_id": self.litmus_id,
            "backend": self.backend,
            "subject_id": self.subject_id,
            "resource": self.resource.value,
            "passed": self.passed,
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "transitions": [event.to_dict() for event in self.transitions],
            "observations": dict(self.observations),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LitmusResult:
        if payload.get("schema") != cls.SCHEMA:
            raise ValueError("unexpected stateful litmus result schema")
        rows = payload.get("transitions")
        if not isinstance(rows, list):
            raise TypeError("transitions must be a list")
        transitions = tuple(
            Transition(
                index=int(row["index"]),
                source=str(row["source"]),
                target=str(row["target"]),
            )
            for row in rows
        )
        observations = payload.get("observations", {})
        if not isinstance(observations, dict):
            raise TypeError("observations must be an object")
        passed = payload.get("passed")
        if not isinstance(passed, bool):
            raise TypeError("passed must be a boolean")
        error = payload.get("error")
        if error is not None and not isinstance(error, str):
            raise TypeError("error must be a string or null")
        return cls(
            litmus_id=str(payload["litmus_id"]),
            backend=str(payload["backend"]),
            subject_id=str(payload["subject_id"]),
            resource=ResourceKind(payload["resource"]),
            passed=passed,
            initial_state=str(payload["initial_state"]),
            final_state=str(payload["final_state"]),
            transitions=transitions,
            observations=observations,
            error=error,
        )


def append_result(path: Path, result: LitmusResult) -> None:
    line = json.dumps(
        result.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def read_results(path: Path) -> list[LitmusResult]:
    results = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at line {line_number}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL record at line {line_number}"
                ) from error
            if not isinstance(payload, dict):
                raise TypeError(f"JSONL record at line {line_number} is not an object")
            results.append(LitmusResult.from_dict(payload))
    return results
