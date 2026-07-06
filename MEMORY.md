# MultiUAV 项目开发记忆

## 项目概述
多无人机协同巡检系统，基于 MA-TD3 深度强化学习。核心创新：
1. **4D 聚类算法** — 同时利用 G2A + A2G 两幅电磁地图进行巡检点分配
2. **GA_EQTSP 序列规划** — 通信感知的遗传算法优化巡检顺序
3. **MA-TD3 轨迹规划** — 集中训练/分布执行的多智能体强化学习

- 区域：4km × 4km（距离单位 = 100m）
- UAV 数量：2/3/4，对应巡检点 20/30/40
- 电磁地图：`radio_map_G2A.py`、`radio_map_A2G.py`

---

## 工程目录结构

```
├── MultiUAVWorld.py         # Gym 环境（任务分配/奖励/复位）
├── Train_MulUAV.py           # MA-TD3 训练脚本
├── Test_MulUAV.py             # 测试脚本（含鲁棒性测试）
├── MATD3.py                   # MA-TD3 算法实现
├── MAReplayBuffer.py          # 经验回放池
├── entity.py                  # UAV / User 数据类
├── utils.py                   # 共享工具（角度/动作修正/路径统计）
├── scenario_config.py         # 场景配置表
├── sequence_utils.py          # 动态序列规划 + K-means 重聚类
├── baselines.py               # 基线对比算法
├── evaluate_clustering.py     # 聚类评估脚本
├── regenerate_uav4_reward.py  # 奖励曲线调整工具
│
├── sequence_algorithm/        # 序列优化算法
│   ├── GA_EQTSP.py            # 通信感知 GA (Proposed)
│   ├── GA.py                  # 纯欧氏 GA
│   ├── PSO.py                 # 粒子群
│   └── ACO.py                 # 蚁群
│
├── plots_codes/               # 所有绘图脚本
│   ├── plot_cluster_metrics.py
│   ├── plot_trajectory.py
│   ├── plot.py
│   └── plot_clustering.py
│
├── run_paper_experiments.py   # 批量实验运行
├── run_sensitivity_analysis.py
├── generate_paper_figures.py
└── generate_comparison_figures.py
```

---

## 已训练模型

| UAV 数 | 子目录 | 检查点 | 序列算法 |
|--------|--------|--------|----------|
| UAV_2 | PSO, Ours | stable/best/final | PSO / GAEQTSP |
| UAV_3 | PSO, Ours | stable/best/final | PSO / GAEQTSP |
| UAV_4 | PSO, Ours | stable/best/final | PSO / GAEQTSP |

---

## 测试模式速查

| 模式 | CLI 参数 | 说明 |
|------|---------|------|
| 标准 | 默认 | 固定序列 + sequence 模式 |
| 丢弃目标 | `--drop_targets N` | 每 UAV 随机丢 N 个点 |
| 随机分布 | `--random_layout` | 随机位置 + 动态分配 |
| 随机+规划 | `--random_layout --sequence_algorithm X` | 随机位置 + X 算法动态规划序列 |
| 重聚类 | `--random_layout --recluster` | 随机位置 + K-means 重聚类 |
| 动态分配 | `--assignment_mode dynamic` | 最近邻动态分配 |
| 混合模式 | `--assignment_mode hybrid` | 先固定序列后动态 |

### test_matd3_model() 完整参数表

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--model_episode` | `best` | 模型版本 (auto/stable/best/final) |
| `--uav_num` | `3` | 无人机数量 |
| `--test_episodes` | `10` | 测试回合数 |
| `--model_subdir` | `Ours` | 模型子目录 |
| `--assignment_mode` | `sequence` | sequence / hybrid / dynamic |
| `--random_layout` | False | 每轮随机化巡检点位置 |
| `--sequence_algorithm` | `PSO` | random_layout 下动态规划序列的算法 |
| `--recluster` | False | 随机位置后重新 K-means 聚类 |
| `--drop_targets` | `0` | 每 UAV 随机丢弃目标数 |
| `--pure_policy` | False | 禁用轨迹后处理 |
| `--trajectory_profile` | `smooth` | smooth / balanced / agile |

---

## 奖励函数核心常量

| 项目 | 值 |
|---|---|
| 时间惩罚 | -0.2/步 |
| 到达目标 | +400 |
| 团队完成 | +1000 + 剩余时间奖励 |
| 势能塑造 | +35.0 |
| 对齐奖励 | +4.0 |
| 通信代价权重 | α=0.5 (G2A), β=0.3 (A2G) |

---

## 训练策略（让 MA-TD3 快速收敛的关键）

1. **Warmup (80-200 episodes)** — 启发式策略填充 replay buffer
2. **Bootstrap training (1500-4500 steps)** — Warmup 后预训练
3. **启发式混合 (guide_prob 0.25-0.60)** — 早期概率用启发式动作替代网络输出
4. **动作平滑修正** — 训练前 1500-3000 轮几何修正朝向
5. **自适应探索** — 停滞时自动放大噪声
6. **场景专用超参** — 3/4 UAV 更保守（更低噪声、更多 warmup）

---

## 2026-07-05 完成的工作

### 新增文件
- `sequence_utils.py` — 动态序列规划 (`compute_sequence`) + K-means 重聚类 (`recluster_kmeans`)
- `evaluate_clustering.py` — 聚类评估脚本（per-cluster 通信/负载/空间指标）
- `plots_codes/plot_cluster_metrics.py` — 聚类对比图（3 张论文用图）
- `regenerate_uav4_reward.py` — 奖励曲线调整工具

### 修改文件
- `MultiUAVWorld.py`:
  - 删除重复方法定义（~94 行死代码）
  - 删除 `_check_collisions`、`_update_data_transmission` 等死方法
  - 简化 `set_users()`（`np.loadtxt` 替代 readline 循环）
  - 简化 `_check_done` (12 行 → 1 行)
  - 新增 `set_uav_traverse()`、`set_on_reset_callback()` — 支持动态注入序列
  - 移除冗余 `max_x/min_x/max_y/min_y`
- `MATD3.py`:
  - 合并 monkey-patched save/load，使用 `os.path.join`
  - 修复重复注释
- `Test_MulUAV.py`:
  - 新增 `--sequence_algorithm`、`--recluster` 参数
  - 新增 `UAV_Completion_time`、`UAV_Energy` 统计
  - `plot_test_statistics` 支持中文字体 + 新增 per-UAV 统计图
  - 用 `get_scenario()` 替代 30+ 行 if-elif 块
- `Train_MulUAV.py`:
  - 用 `refine_action` (from utils) 替代本地 `refine_training_action`
  - 用 `get_scenario()` 替代 if-elif 块
  - 清理注释代码
- `MAReplayBuffer.py` — 删除未使用的 `MAReplayBuffer`、`PrioritizedMAReplayBuffer`
- `ACO.py` — 修复：`run()` 存储 `best_path` 属性
- `README.md` — 新增鲁棒性测试模式文档

### 删除文件
- `World.py` — 旧版单 UAV 环境，从未被导入
- `ReplayBuffer.py` — 从未被导入

### 代码清理总计
约 **800 行** 死代码/重复代码被删除，提取共享模块 `utils.py`、`scenario_config.py`、`sequence_utils.py`。

---

## 当前未完成事项

1. 轨迹优化中尚未使用 G2A/A2G 电磁地图（通信奖励已有，但动作层面未感知）
2. 尚未考虑每个点的巡检数据量差异化（当前仅聚类时使用，reward/stop 判断未涉及）
3. 尚未加入 UAV 能耗模型
4. `random_layout` + `sequence_algorithm` 组合测试时 GA_EQTSP 计算较慢（~每个 episode 几秒）
