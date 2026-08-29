# AccelPact

AccelPact studies stateful host-runtime protocols across disjoint accelerator
stacks. Its scope is stream/event ordering, buffer ownership, graph
capture/replay, allocator reuse, collective epochs, and failure recovery. It
does not compare cross-platform floating-point values and does not inspect
barriers inside accelerator kernels.

The repository is currently at **AP-G0Q**, a small qualification gate. The gate
asks whether a backend-neutral lifecycle oracle can distinguish valid protocol
executions, expected negative controls, unsupported API combinations, and
genuine poisoned or stale runtime state on the current A100/CUDA and
Ascend 910B/CANN stacks.

The protocol is frozen in [`docs/AP_G0Q_PROTOCOL.md`](docs/AP_G0Q_PROTOCOL.md).
No automatic repair, model workload, or large fuzzing framework is implemented
before this gate produces a stable current-stack signal.

## Local checks

```bash
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```

