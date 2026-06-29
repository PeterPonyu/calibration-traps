# spec_route — Direction 014: 早期轨迹信号的路线/结局预测律（post-hoc）

ROUND-1 是**纯 post-hoc CPU 分析**：只读存量 `experiments/results/` 日志，
不训练任何模型，不占 GPU。方向文档：`directions/014-spec-route-predictor.md`。

## 只读契约（硬规则）

- 对一切现有 results 命名空间 **read-only**；本目录代码绝不写入、改名、
  触碰任何既有命名空间。
- **显式排除** `grok_numerics*`（009 在跑 + 避让条款）与 `logs/009-*`。
- 产物只写 `results/spec_route/` 与 `results/figures-014/`。
- 不 import 任何 `experiments/*/` 实验代码（纯 jsonl 解析，零耦合）。

## 模块

| 文件 | 职责 |
|---|---|
| `corpus.py` | 命名空间发现 + jsonl 解析（`_meta` / step 行 / `_summary`），run 表构建 |
| `features.py` | 早期窗口特征（W-mem / W-K），泄漏纪律在此强制 |
| `labels.py` | y1 grokked / y2 delay / y3 route（001 R4 + 002 P1 定义）/ y4 rescue |
| `predict.py` | numpy ridge/logistic + 三基线 + 三 CV 方案 + permutation 检验 |
| `run_posthoc.py` | driver：`--dry-run`（语料盘点，只读）/ `--smoke`（自检，零写入）/ 正式分析 |

## 用法

```bash
python run_posthoc.py --smoke     # 端到端自检（植入信号 + 泄漏守卫），<60 s，零写入
python run_posthoc.py --dry-run   # 盘点语料：run 数、标签覆盖、通道可用性（只读）
python run_posthoc.py             # 正式分析（归执行者会话；写 results/spec_route/）
```

## 语料 tier

- **Tier-A（范数通道）**：grid_main, lr_control, lr_control_sc3, wd_sweep,
  tf_sweep, task_mul, task_s5, fine_eval, s5_rescue
- **Tier-B（+谱/方向通道）**：mech, s5_mech
- 排除：calib*, degree_staircase*, induction_*, sink_*, grok_numerics*,
  muon_plasticity（P5 触发时另行加入）

## 预登记要点（详见方向文档）

- 特征窗口内不得出现 memorize 之后的记录（W-mem）/ K 步之后的记录（W-K）；
  `features.py` 对此有断言 + smoke 的毒化记录守卫测试。
- y3 路线标签 = `wn_hidden(T_grok)/wn_hidden(T_mem)`，>1 增长路线 / <1 收缩
  路线；rot_rate 不得单独作路线判据（002 红队条款）。
- headline 量 = 信号增量 Δ（signal − config-only），逐 CV 方案报告。
- kill 判据：within-optimizer 判别 ≈ chance ⇒ 预测器只是优化器标签读出。
