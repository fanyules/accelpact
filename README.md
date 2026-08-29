# AccelPact

AccelPact studies stateful host-runtime protocols across disjoint accelerator
stacks. Its scope is stream/event ordering, buffer ownership, graph
capture/replay, allocator reuse, collective epochs, and failure recovery. It
does not compare cross-platform floating-point values and does not inspect
barriers inside accelerator kernels.

The repository has completed the **AP-G0Q** and **AP-G0C** adjudication slices.
AP-G0Q qualifies oracle sensitivity and retains the known A100 capture-abort
recovery failure as a regression seed. AP-G0C qualifies the two-rank collective
ledger on both platforms and exposes a reproducible recovery-capability
difference after an intentionally incomplete epoch.

| AP-G0C gate | A100/NCCL | 910B/HCCL |
| --- | --- | --- |
| stale generation control | detected | detected |
| 128 matched epochs in one generation | pass | pass |
| clean destroy/recreate, five fresh pairs | 5/5 pass | 5/5 pass |
| incomplete epoch then recreate, five fresh pairs | 5/5 recovered | 5/5 in-process reinitialization failure |

On 910B, the HCCL watchdog reports the designated incomplete collective, one
rank remains inside communicator teardown, and the surviving rank cannot finish
the out-of-band destroy handshake. The exact pattern reproduced in all five
fresh process pairs. Because public process-group reinitialization is outside
the qualified PyTorch contract, AP-G0C records this as a capability limitation,
not a protocol violation.

The AP-G0Q protocol is frozen in
[`docs/AP_G0Q_PROTOCOL.md`](docs/AP_G0Q_PROTOCOL.md).
No automatic repair, model workload, or large fuzzing framework is implemented.
The TP2 communicator-generation protocol is frozen in
[`docs/AP_G0C_PROTOCOL.md`](docs/AP_G0C_PROTOCOL.md); its canonical campaign
summary is [`results/ap_g0c_tp2_summary.json`](results/ap_g0c_tp2_summary.json).
The next frozen discovery gate is the TP1 native allocator-retirement protocol
in [`docs/AP_G0R_PROTOCOL.md`](docs/AP_G0R_PROTOCOL.md).

## Local checks

```bash
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```
