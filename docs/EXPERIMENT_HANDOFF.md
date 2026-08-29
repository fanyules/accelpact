# AccelPact 项目与实验交接

最后更新：2026-08-29（Asia/Shanghai）

项目：**AccelPact: Stateful Protocol Conformance across Disjoint Accelerator Runtimes**

目标会议：ASPLOS / EuroSys 方向

公开仓库：<https://github.com/fanyules/accelpact>

本文是项目的标准交接入口，覆盖研究边界、当前结论、仓库结构、两套加速器环境、
运行方法、证据位置、下一阶段 gate 和停止条件。私有登录信息、主机地址、本机桥接
路径及临时凭据不属于仓库内容；需要连接服务器时，从单独保存的私有环境交接中读取。

## 1. 一页状态

截至本次交接，项目不是“已经成功”，也不是“已经失败”，而是处于：

> **基础 oracle 与实验纪律已经成立；发现了一个已知 CUDA 回归种子和一个可重复的
> HCCL 恢复能力差异，但尚未找到满足论文主张门槛的新当前合同违反。**

| Gate | 状态 | 已得到的结论 | 能否计入新协议违反 |
| --- | --- | --- | --- |
| AP-G0Q：TP1 event/buffer/graph | 已裁决 | A100 capture-abort 后 eager RNG/allocator recovery 5/5 失败；910B 5/5 clean fallback | 否；A100 属于公开已知失败类别，保留为回归种子 |
| AP-G0C：TP2 collective generation/recovery | 已裁决 | 两端正常 epoch 与 clean recreate 均通过；910B incomplete epoch 后进程内重建 5/5 失败，A100 5/5 恢复 | 否；当前只称恢复能力限制，不称公共合同违反 |
| AP-G0R：TP1 native allocator retirement | 协议已冻结，尚未实现/部署/运行 | 检查 `record_stream`、manual handback、同流回收和双 consumer retirement | 未知；这是下一项 discovery gate |
| 完整 AP-G0 系统 | 未开始 | 生成、缩减、局部修复与工作负载集成尚未投资 | 需先达到多根因发现门槛 |

AP-G0R 的冻结 revision 是 `17bbdb6`。该提交只冻结协议和配置，不包含 runner、
launcher、adjudicator、测试或实验结果。

## 2. 研究问题与不可越界的边界

AccelPact 不要求 CUDA 与 CANN 产生相同具体 trace，也不比较两端浮点结果。核心问题是：

> 能否把一个共同的资源所有权与 happens-before 协议，分别 lowering 成 CUDA/NCCL
> 与 CANN/HCCL 的合法执行，再检查 stale generation、过早回收、非原子失败恢复、
> 重复发布和合法执行卡住等状态化错误？

抽象资源状态机目前覆盖：

```text
Buffer:
owned -> write-pending -> ready -> published -> consumed -> reclaimable

Graph:
idle -> capturing -> committed | aborted
committed -> replayable
aborted -> clean-fallback | poisoned

Collective:
epoch-open(g) -> enqueued -> completed -> reusable-same-generation
                         \-> failed-unknown -> aborting -> destroyed(g)
                                                    \-> recreated(g+1)
```

项目边界固定为 host runtime 层：

- stream/event 顺序；
- buffer ownership、publication 与 allocator retirement；
- graph capture/replay 及失败后的状态清理；
- collective epoch、destroy/recreate 与进程级恢复；
- backend-specific legal lowering 与共同抽象 invariant。

以下不属于主张范围：

- accelerator kernel 内部 barrier 或 DMA/vector/matrix pipeline；
- A100 与 910B 的跨平台浮点逐值一致；
- 用故意非法调用制造 runtime bug；
- 在没有稳定新违反之前构建大规模 fuzzing、通用 repair 或模型集成；
- 把文档明确允许的后端差异写成 bug。

与相邻工作的区分见 [`IDEA_ASSESSMENT.md`](IDEA_ASSESSMENT.md)。当前最大重叠风险
是通用 CUDA/API sequence fuzzing；AccelPact 必须靠跨 runtime 抽象 trace oracle、
stateful metamorphic relations，以及后续的协议级缩减/局部修复形成独立贡献。

## 3. 当前 Git 与仓库状态

### 3.1 关键 revisions

| Revision | 含义 |
| --- | --- |
| `62f1d18` | 冻结 AP-G0Q 协议 |
| `76c60c4` | 实现首版 TP1 backend-neutral oracle 与 runner |
| `850f222` | 修正 NPU event 生命周期 lowering |
| `bb25b98` | cross-stream graph capture 改用 graph-internal Event |
| `6717ab6` | 完成 AP-G0Q 裁决记录 |
| `689a79f` | 冻结 AP-G0C TP2 协议 |
| `354f059` | 实现 AP-G0C runner/launcher/adjudicator |
| `8cf40b5` | 修正 NPU runtime preflight |
| `d936074` | 修正 marker/adjudicator 时序 |
| `bf81c36` | 冻结 AP-G0C 最终可运行版本与 backend-fatal 裁决逻辑 |
| `f94ce00` | 记录 AP-G0C 最终 campaign 与 canonical summary |
| `17bbdb6` | 冻结 AP-G0R allocator-retirement 协议与配置，尚未上机 |

仓库使用 GitHub `origin`。不要创建 `*_v2`、`*_final` 一类副本；修改 canonical
文件并依靠 commit、branch、tag 或 release 保存历史。

### 3.2 目录职责

| 路径 | 内容 |
| --- | --- |
| `src/accelpact/litmus.py` | Buffer/Graph/Collective 生命周期 oracle、结果 schema、JSONL I/O |
| `scripts/run_ap_g0q.py` | AP-G0Q 单进程 TP1 runner |
| `scripts/run_ap_g0c.py` | AP-G0C 每 rank runner |
| `scripts/launch_ap_g0c.py` | AP-G0C 有界 torchrun supervisor、环境与 manifest 证据 |
| `scripts/adjudicate_ap_g0c.py` | AP-G0C evidence validator 与分类器 |
| `configs/ap_g0q.json` | AP-G0Q 冻结矩阵 |
| `configs/ap_g0c.json` | AP-G0C 冻结 TP2 矩阵 |
| `configs/ap_g0r.json` | AP-G0R 冻结 allocator-retirement 矩阵 |
| `docs/AP_G0Q_PROTOCOL.md` | AP-G0Q 合同、负控与裁决规则 |
| `docs/AP_G0C_PROTOCOL.md` | AP-G0C epoch/recovery 合同 |
| `docs/AP_G0R_PROTOCOL.md` | AP-G0R allocation-stream、retirement ledger 与证据合同 |
| `results/ap_g0c_tp2_summary.json` | 唯一 tracked 的 AP-G0C canonical compact result |
| `results/raw_work/` | 本地取回的原始证据，Git 忽略，不得作为公开摘要替代品 |
| `ccfa.yaml` | 项目阶段、gate 状态与当前 claim boundary |

### 3.3 本地检查

在仓库根目录运行：

```bash
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
python -m json.tool configs/ap_g0r.json
```

本次 AP-G0R freeze 前，84 个 unittest 全部通过；协议/配置通过 JSON、YAML、
`git diff --check` 与路径隐私检查。远端加速器环境不依赖 pytest，使用标准库
`unittest` 即可。

## 4. 计算环境

### 4.1 910B 端

| 项目 | 当前值 |
| --- | --- |
| 加速器 | 8 × Ascend 910B4-1，单卡 64 GiB HBM |
| 正式容器 | `graphbudget_gbq0_v0230` |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.23.0` |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cpu |
| torch-npu | 2.10.0.post4 |
| CANN/HCCL | 9.1.0 |
| Driver / npu-smi | 25.5.1 |
| 仓库挂载 | host `/data` 与 container `/data` 对应 |

`torch` 显示 `+cpu` 是该 torch-npu 组合的正常版本字符串，不代表实验运行在 CPU。
应以 `torch_npu`、NPU 可见性和实际 device result 判断。

2026-08-29 23:13 CST 只读复核：容器运行正常，8 张卡健康，AICore 利用率为 0，
没有 NPU compute process；空闲时约 3.2–3.4 GiB HBM 为 runtime/driver 基础占用。

### 4.2 A100 端

| 项目 | 当前值 |
| --- | --- |
| 加速器 | 4 × NVIDIA A100 PCIe 40 GB，无 NVLink |
| Python 环境 | `/root/miniconda3/envs/rimlink-vllm023` |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| NCCL | 2.28.9 |
| NVIDIA driver | 570.169 |

2026-08-29 23:14 CST 只读复核：4 张 A100 均可见，没有 NVIDIA compute process。
该环境没有单独安装 pytest；运行仓库 unittest 不需要新增软件。

### 4.3 两端关系

两台服务器的编译器、runtime、stream/event、graph、allocator、collective 与失败恢复
实现彼此独立，这正是研究价值。两端之间的 1 GbE 只适合：

- 代码归档与 SHA-256；
- 测试序列、seed、event/stream/allocator 状态；
- collective epoch、credit、trace digest；
- JSONL、manifest 和最小复现。

不要把它设计成频繁传输 activation、梯度或大 tensor 的数据平面。

## 5. 访问与部署原则

公开仓库只记录逻辑角色，不记录私有地址。标准路径是：

```text
本地工作站
  -> 已建立的 gateway bridge
  -> 910B host/container
  -> 既有 key-based SSH
  -> A100 host
```

使用方法：

1. 从私有环境交接读取 bridge 路径、gateway alias 与 A100 target；
2. 先执行 bridge `--ping`；
3. 只使用既有认证连接，不把密码、token、private key 写入命令、日志或仓库；
4. 本机 checkout 是唯一编辑源，服务器的 revision 目录只运行、不编辑；
5. 服务器按离线环境处理，不从服务器临时安装包或下载代码；
6. 新代码先在本地提交，再用 `git archive` 形成不可变部署包；
7. A100 与 910B 使用同一 archive，并在三处核对 SHA-256；
8. 每个正式 cell 使用新 result directory 和 fresh process；
9. 每次运行前后检查目标设备没有其他 compute process。

本地制作部署包：

```bash
REV=$(git rev-parse HEAD)
STAGING_DIR=/path/to/private/staging
git status --short
git archive --format=tar.gz --output="${STAGING_DIR}/accelpact-${REV}.tar.gz" "$REV"
sha256sum "${STAGING_DIR}/accelpact-${REV}.tar.gz"
```

若 `git status --short` 非空，不要把 dirty tree 当成正式实验 revision。上传、转发与
目标主机名由私有环境交接提供。远端建议解压到 `/data/AccelPact-<short-revision>`，
不要覆盖既有证据目录。

## 6. 通用运行前后检查

### 6.1 Git/代码一致性

在本地和部署端记录：

```bash
git rev-parse HEAD
sha256sum <deployment-archive>
python -m unittest discover -s tests -v
```

若部署目录来自 `git archive` 而没有 `.git`，以 archive SHA-256、传入的
`--source-revision` 和 manifest 为准。

### 6.2 设备空闲

910B：

```bash
npu-smi info
```

A100：

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

若发现无关 compute process，停止本次 cell，不要用宽泛的进程清理命令。只处理本次
launcher 创建且 PID/PGID 明确的进程；处理后重新执行设备检查。

### 6.3 结果目录

每次 launcher 必须写入一个原本不存在的目录。目录至少保留：

- 完整 command 与冻结 environment override；
- runtime、allocator 与 device inventory；
- combined stdout/stderr；
- process/rank exit 与 timeout 状态；
- JSONL、marker、adjudication summary；
- file-level SHA-256 manifest。

不得覆盖、补写或手工修正正式 run directory。构造失误的 commissioning attempt
也保留，但必须在汇总中明确排除出 scientific matrix。

## 7. 如何使用 AP-G0Q

AP-G0Q 已完成，通常只用于回归复现，不应重新解释成新 discovery。

查看 runner 入口：

```bash
python scripts/run_ap_g0q.py --help
```

单 cell 示例：

```bash
python scripts/run_ap_g0q.py \
  --backend cuda \
  --device-index 0 \
  --iterations 128 \
  --seed 20260829 \
  --litmus capture_abort_eager_recovery \
  --output results/<new-run>/capture_abort.jsonl
```

910B 在正式容器内把 `--backend cuda` 改为 `--backend npu`。每个需要确认的异常必须
在至少 5 个 fresh process 中重跑；4/5 一致才进入候选异常。负控 `missing_join` 与
`rebound_input` 只验证 generator/oracle sensitivity，不能成为 runtime violation。

AP-G0Q 最终裁决 run：`ap_g0q_adjudication_20260829T192202CST`，使用 source
`bb25b98`。

## 8. 如何使用 AP-G0C

AP-G0C 是 world size 2、每端 device 0/1 的 TP2 gate。正式运行优先使用 launcher，
不要直接拼 torchrun 后再人工判断日志。

```bash
python scripts/launch_ap_g0c.py \
  --config configs/ap_g0c.json \
  --platform <a100-or-910b> \
  --litmus <litmus-id> \
  --run-id <campaign-id> \
  --run-dir results/<campaign>/<platform>/<cell>/<repetition> \
  --repetition 1 \
  --source-revision bf81c36ac89785cef48f32de046e65061160d4ea \
  --torchrun <absolute-path-to-torchrun> \
  --master-port <unique-local-port>
```

冻结 litmus 顺序：

1. `stale_generation_dispatch`；
2. `collective_same_generation_reuse`；
3. `collective_clean_destroy_recreate`；
4. `collective_partial_epoch_timeout_recreate`。

launcher 已负责：设置两张逻辑卡可见、180 秒外层 deadline、独立 process group、
timeout 后按本次 PGID 回收、收集 rank status/marker、调用 adjudicator 并写 manifest。
若需独立重验已有 evidence：

```bash
python scripts/adjudicate_ap_g0c.py \
  --launcher-evidence <run-dir>/launcher_evidence.json \
  --rank-jsonl 0=<rank0-jsonl> \
  --rank-jsonl 1=<rank1-jsonl> \
  --marker <marker-dir> \
  --output <new-summary-path>
```

实际文件名以 run manifest 为准；adjudicator 的 `--help` 是参数权威来源。

## 9. AP-G0Q 与 AP-G0C 已完成结果

### 9.1 AP-G0Q

最终 matched recovery 结果：

- A100：`capture_abort_eager_recovery` 5/5 为
  `protocol_violation -> poisoned`；eager RNG 持续报告 capture-state 错误，
  `graph.reset()` 未恢复；
- 910B：同一抽象序列 5/5 为 `valid_pass -> clean_fallback`；
- 910B 独立 `rebound_input` 负控在 fresh process 中被正确检测；
- 两个早期 910B construct attempt 分别暴露了 ExternalEvent 多 waiter 使用错误和
  graph cross-stream join lowering 错误，它们是 harness 修正，不是 CANN 违反。

证据 archive SHA-256：

- A100：`e165944cc6826b5d2aff1f1e6c93aef5fff3bec73cd9984c4d205fee30b4b94b`；
- 910B：`78bc067516717412ec589d4b9747f34eaf955f6ffcdf6578cf98427dcbaab551`。

### 9.2 AP-G0C

最终 campaign：`ap_g0c_tp2_20260829T214339CST`，source `bf81c36`。

| Litmus | A100/NCCL | 910B/HCCL |
| --- | --- | --- |
| stale generation dispatch | detected | detected |
| 128 matched same-generation epochs | pass | pass |
| clean destroy/recreate | 5/5 capability pass | 5/5 capability pass |
| incomplete epoch then recreate | 5/5 expected-timeout recovered | 5/5 in-process reinitialization capability failure |

910B 的五个失败 pair 都到达 `fault_ready`、`fault_observed` 和 `ready_destroy`。
指定 rank 的 HCCL watchdog 观察到 incomplete collective；另一 rank 完成 group destroy，
但进程内新 communicator generation 无法完成。五次 rank exit pattern 均为 rank 0
`-15`、rank 1 `4`。

这说明当前栈存在稳定的进程内恢复能力差异，但 incomplete collective 是故意 stimulus，
而公共 process-group reinitialization 不是本 gate 已资格化的承诺，因此分类为
`reinitialization_capability_failure`，不是 `protocol_violation`。

canonical summary：[`results/ap_g0c_tp2_summary.json`](../results/ap_g0c_tp2_summary.json)。
完整性统计：24 个 run summary、24 个 manifest、452 个 manifested artifact，
0 个校验错误；两端 preflight/postflight 均为 0 个 compute process。

证据 archive SHA-256：

- A100：`79f1955fdbca6f9ffa7ef25875eb4bee7f11b1e71d86d33830ea836c28cc39f0`；
- 910B：`ca0e478613d7e64f0677fcdb1412c95911f1c7662451e9818cafd6999314aec6`。

## 10. AP-G0R：下一项正式工作

协议：[`AP_G0R_PROTOCOL.md`](AP_G0R_PROTOCOL.md)

配置：[`ap_g0r.json`](../configs/ap_g0r.json)

冻结 revision：`17bbdb6`

AP-G0R 已有完整实验合同，但以下文件尚不存在：

- `scripts/run_ap_g0r.py`；
- `scripts/launch_ap_g0r.py`；
- `scripts/adjudicate_ap_g0r.py`；
- 对应的 runner/launcher/adjudicator tests；
- 任何 AP-G0R accelerator result。

实现时不得悄悄改变以下冻结点：

- TP1、device 0、eager、native caching allocator；
- `float32[2,097,152]`，正整数 generation 1–128，copy-only；
- source、churn、probe 的 allocation 与 poison write 都在真实 creation stream；
- poison 为精确 `-(g + 1)`；
- 16 次 `2048 x 2048` float32 matmul 只制造 pending window；
- 每代一次 public `empty_cache`；
- pointer equality 后再次逐 consumer query completion event；
- 128 代中至少 120 代 release 时 pending；
- 至少 32 代满足同代 reuse-qualified；
- negative control 先于 baseline 和 cross-stream valid cells；
- negative control 的 8 代合法 follow-up 必须 8/8 pending、至少 2 代 reuse-qualified；
- calibration 是每平台一个独立 fresh-process artifact，8/8 pending 后才能引用；
- 每个进程 180 秒 deadline，异常 cell 至少 5 个 fresh process，4/5 一致。

实现与执行顺序：

1. 实现 backend adapter、逐代 Buffer oracle 与 retirement ledger；
2. 实现 standalone calibration artifact 及科学 cell 对其 run ID/SHA-256 的验证；
3. 实现两个 host-guard negative controls；
4. 实现 same-stream recycle baseline；
5. 实现 single-consumer `record_stream`、manual `wait_stream` handback、two-consumer
   `record_stream`；
6. 实现 no-clobber launcher、deadline、runtime inventory、phase/error evidence 和 manifest；
7. 实现严格 adjudicator，先用 fake backend/unit tests 覆盖所有分类；
8. 本地全测通过后提交一个明确 revision；
9. 用该单一 revision 同时部署到 A100 与 910B；
10. 每平台先 calibration，再两个负控，再 baseline，最后三个 valid cross-stream cells；
11. 任一 supported valid abnormal cell 用 5 个 fresh process 重验；
12. 取回 evidence，逐文件验证 manifest，再生成一个 compact tracked summary。

如果实现需要修改抽象 invariant、阈值、dtype、流角色或分类边界，必须先更新协议和
配置、重新审计并形成新的 freeze revision；不能在看过科学结果后追改合同。

## 11. 结果分类纪律

所有 gate 共用以下原则：

- negative control miss 是 oracle/generator 问题，不是 runtime violation；
- unsupported 是支持边界，不是假定通过；
- 没有观察到实际 reuse 时应是 `inconclusive`，不能写成 allocator pass；
- launcher/config/source/manifest 不一致优先归 `harness_error`；
- intentional fault 后无法恢复，只有在恢复 API 已有明确公共合同的 gate 中才可称
  `protocol_violation`；否则只能称 capability limitation；
- stale/mixed output 可以支持 host-runtime ordering failure，但若没有 premature pointer
  equality 或 trace 证据，不应提前把根因写死为 allocator；
- 跨平台耗时不做性能比较，只记录执行边界；
- publication claim 只使用完整、可校验、达到 fresh-process 确认阈值的 evidence。

## 12. 原始证据与归档

两端不可变 evidence 保存在对应 source revision 的 `/data/AccelPact-<revision>/results/`
下。本地取回副本放入 Git 忽略的 `results/raw_work/`。不要移动、重命名或覆盖既有
AP-G0Q/AP-G0C 目录。

取回后至少检查：

1. archive SHA-256 与源端一致；
2. manifest schema、run ID、source revision、config digest 一致；
3. manifest 中每个文件的 size 与 SHA-256 一致；
4. JSONL row 数、rank/generation cardinality 与协议一致；
5. launcher exit、rank exit、timeout、marker prefix 与 adjudicator summary 一致；
6. preflight/postflight 没有残留 compute process；
7. commissioning attempts 与 final scientific campaign 明确分开。

只有 compact、去除私有拓扑并通过一致性审计的 summary 才进入 Git。

## 13. 科学决策门槛

AccelPact 进入完整系统实现，至少需要：

- 两个不同根因的当前栈协议违反；
- 至少一个与 vLLM 无关；
- 至少一个产生 silent stale read、合法执行卡住或 poisoned fallback；
- 至少一个来自预先冻结的 protocol-generated sequence，而非照抄公开 issue；
- 独立 runtime reproducer；
- 后续局部 Pact repair 在真实工作负载中的开销目标约 3% 以内；
- 最终覆盖至少三类应用，例如视觉训练、diffusion/graph execution、分布式训练或
  科学计算。

目前尚未达到前两项，因此：

- 可以继续做 AP-G0R 和相邻的小型 contract-backed discovery gates；
- 不应提前实现大型 DSL、通用 reducer、repair synthesis 或多应用评测；
- 若后续只得到已知回归、文档差异、非法用法或 capability boundary，应把成果定位为
  regression/conformance suite，而不是夸大成完整 ASPLOS system。

## 14. 操作边界与异常处理

本项目仅连接研究组已授权的两台 accelerator server。正常操作包括固定脚本部署、
设备状态读取、fresh-process 实验、结果取回和 GitHub 版本管理。不进行第三方主机发现、
端口探测、凭据尝试或任意外网访问。

若 Codex 或运行环境再次对网络动作提出限制：

1. 不尝试规避、改写或关闭限制；
2. 暂停新的远端动作；
3. 明确说明仅使用既有认证连接和两台授权服务器；
4. 展示本文件中的固定实验范围、目标 revision 和具体只读/运行命令；
5. 只在允许的边界内继续。

若实验 timeout 或进程异常：

1. 保留当前 stdout/stderr、marker、launcher evidence；
2. 只回收本次 launcher 创建的 PID/PGID；
3. 不重置设备，不执行宽泛 kill，不删除旧 evidence；
4. 再次检查 accelerator process；
5. 先判断是完整合法 schedule、capability boundary 还是 harness phase 不可定位；
6. 同一问题连续三次无法推进时停止重复尝试，记录证据并重新审查协议/实现。

## 15. 新接手者的最短清单

1. 阅读本文件、`ccfa.yaml`、`AP_G0R_PROTOCOL.md` 和 `ap_g0r.json`；
2. 确认 GitHub `origin`、当前 branch、HEAD 与 clean worktree；
3. 运行 84 个 unittest、ruff check 和 format check；
4. 从私有环境交接取得 gateway/host 参数，不把它们写进仓库；
5. 只读检查 A100/910B runtime 与 compute process；
6. 从 `17bbdb6` 或之后明确的 construct-fix revision 实现 AP-G0R；
7. 先测试 adjudicator 和 guard，再运行 accelerator；
8. 同一 archive、同一 config、不同 fresh result directory 跑两端；
9. 校验 manifest 后再做科学分类；
10. 不把 known regression 或 capability difference 写成新的协议违反。

交接完成的判据是：新接手者能从一个 clean Git revision 重建代码、确认两端环境、
运行一个 bounded cell、验证其 evidence，并准确说明“观察到了什么”与“当前能主张
什么”之间的差别。
