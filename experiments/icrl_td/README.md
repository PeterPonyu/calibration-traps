# icrl_td — Direction 016: in-context TD 的涌现动力学

方向文档：`directions/016-icrl-td-emergence.md`（银行 ICRL-TD1 7.7 消耗，
2026-06-12 确认性核查 PROCEED）。纯仿真：随机 MRP 轨迹流在线生成，零下载。

## 模块

| 文件 | 职责 |
|---|---|
| `data.py` | 随机 MRP 生成（Dirichlet 行随机核 + 离散化状态奖励 + 精确 V）、轨迹 token 流、P4 reward-permutation 控制变体 |
| `probes.py` | TD-对齐（vs 真 TD(0) 轨迹）、Bellman 残差（vs 真值核）、核-注意力探针、涌现检测器 + `self_test()`（TD-oracle/常数预测器分离） |
| `train_icrl.py` | Config + SeqTransformer 复用（induction_emergence，importlib 反遮蔽模式，**不得修改来源目录**）+ 001 标准 hybrid 优化器 + 在线训练/评测 + jsonl |
| `run_icrl.py` | 网格 driver：optimizer×T×seed = 45 cells；`--smoke`/`--dry-run`/shard/resume |

## 用法

```bash
python run_icrl.py --smoke     # 探针自检 + CPU mini-train（muon hybrid），<60s 零写入
python run_icrl.py --dry-run   # 打印网格
python run_icrl.py             # 正式网格（归执行者；写 results/icrl_td/）
```

## 纪律备忘

- 探针全部在 held-out MRP 上对真值量（P、TD(0) 轨迹）计算——与训练 CE
  在构造上不同义反复（kill 判据 1）。
- P4 模仿控制臂失败 ⇒ P1–P3 全部作废（kill 判据 2，先报）。
- eval_every 须细于预期涌现步的 1/10（007 地板教训；smoke 后校准）。
- 涌现阈值 acc_thresh=0.7 为预登记值，正式跑前仅可经 smoke 校准修改一次
  并在方向文档记录。
