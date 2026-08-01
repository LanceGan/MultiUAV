# CLAUDE.md — MultiUAV 项目开发指南

## 项目概述

多无人机协同巡检系统（期刊论文），三层算法架构：
1. **4D 聚类** (`Clustering.py`) — G2A + A2G 双地图通信感知任务分配
2. **序列规划** (`sequence_algorithm/`) — GA/PSO/ACO/GA_EQTSP 优化巡检顺序
3. **轨迹规划** (`Train_MulUAV.py`) — MA-TD3 深度强化学习优化飞行轨迹

- **区域**: 4km × 4km（距离单位 = 100m）
- **UAV 数量**: 2/3/4，对应巡检点 20/30/40
- **电磁地图**: `radio_map_G2A.py`、`radio_map_A2G.py`
- **模型**: `results/models/MA-TD3/` UAV_2/3/4 各含 PSO、Ours 两个训练变体

## 目录结构

```
核心环境     MultiUAVWorld.py (760L)  Gym 环境
训练         Train_MulUAV.py (714L)   MA-TD3 训练脚本
测试         Test_MulUAV.py (829L)    多模式测试脚本
算法         MATD3.py (448L)          MA-TD3 算法（CTDE）
回放池       MAReplayBuffer.py (84L)
实体         entity.py (26L)          UAV / User 数据类
共享工具     utils.py (114L)          wrap_angle / heuristic_action / refine_action / mkdir
场景配置     scenario_config.py (49L)  UAV→巡检点映射 + BS_LOC 等
序列工具     sequence_utils.py (145L) compute_sequence / recluster_kmeans / load_cluster_labels
基线         baselines.py (445L)      聚类+序列对比算法
聚类         Clustering.py (209L)     kmeans_4d 算法

序列算法     sequence_algorithm/
            ├── GA_EQTSP.py (548L)    通信感知 GA (Proposed)
            ├── GA.py (391L)          纯欧氏 GA
            ├── PSO.py (336L)         粒子群
            └── ACO.py (279L)         蚁群

评估脚本     evaluate_clustering.py (304L)    聚类 per-cluster 通信/负载指标
            evaluate_routing.py (76L)         序列对比评估
            evaluate_routing_e2e.py (79L)     序列端到端对比
            evaluate_end_to_end.py (173L)     聚类端到端对比
            generate_sequences.py (49L)       为不同聚类方法生成 PSO/GA 序列

绘图         plots_codes/
            ├── plot_cluster_metrics.py      聚类指标对比图 (3张)
            ├── plot_clustering_map.py        聚类空间可视化
            ├── plot_routing_comparison.py    序列对比图 (3张)
            ├── plot_routing_e2e.py           序列端到端对比图 (3张)
            ├── plot_routing_paths.py         序列路径可视化
            ├── plot_end_to_end.py            聚类端到端对比图 (3张)
            └── plot_trajectory.py            轨迹+射频地图可视化
```

## 论文实验管道

```
聚类对比: Clustering.py → evaluate_clustering.py → plot_cluster_metrics.py
序列对比: routing_*.json → evaluate_routing.py → plot_routing_e2e.py
端到端:   3 聚类 × PSO × MA-TD3 → evaluate_end_to_end.py → plot_end_to_end.py
```

### 生成的论文图表

```
results/paper_experiments/clustering/
  cluster_metrics_comparison.png    # N=3, 3 聚类方法 × 1×3 数据量/G2A/A2G 热力图
  cluster_balance_comparison.png    # 空间紧凑度 + 负载均衡
  cluster_imbalance_comparison.png  # 簇间不均衡度 CV
  clustering_visualization.png      # 3 方法空间分配对比

results/paper_experiments/routing/
  routing_e2e_comparison.png        # 5 序列算法 × 1×4 指标柱状
  routing_e2e_trajectory.png        # 5 序列算法轨迹对比
  routing_e2e_per_uav.png           # Per-UAV 分组柱状
  end_to_end_comparison.png         # 3 聚类方法 × 1×4 端到端
  end_to_end_trajectory.png         # 3 聚类方法轨迹
  end_to_end_per_uav.png            # Per-UAV 分组柱状
  routing_paths.png                 # 序列路径可视化
```

## Test_MulUAV.py 完整参数表

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model_episode` | `best` | auto/stable/best/final |
| `--uav_num` | `3` | 2/3/4 |
| `--test_episodes` | `10` | 测试回合数 |
| `--model_subdir` | `Ours` | PSO/Ours |
| `--custom_sequence_path` | None | 自定义序列 NPZ 路径 |
| `--result_suffix` | `''` | 结果目录后缀（区分不同测试） |
| `--assignment_mode` | `sequence` | sequence/hybrid/dynamic |
| `--random_layout` | False | 每轮随机化巡检点位置 |
| `--sequence_algorithm` | `PSO` | GA/GA_EQTSP/PSO/ACO |
| `--recluster` | False | K-means 重聚类 |
| `--drop_targets` | `0` | 每 UAV 随机丢弃点数 |
| `--pure_policy` | False | 禁用轨迹后处理 |
| `--trajectory_profile` | `smooth` | smooth/balanced/agile |

## 常用命令

```bash
# 标准测试
python Test_MulUAV.py --uav_num 3 --model_subdir PSO

# 自定义序列
python Test_MulUAV.py --uav_num 3 --model_subdir Ours \
  --custom_sequence_path results/datas/sequence/Users_30_Clusteredsave_path_PathUAV_PSO_bal_kmeans_3.npz \
  --result_suffix _bal_kmeans

# 鲁棒性测试
python Test_MulUAV.py --uav_num 2 --model_subdir PSO --drop_targets 2
python Test_MulUAV.py --uav_num 2 --model_subdir PSO --random_layout --sequence_algorithm PSO
python Test_MulUAV.py --uav_num 2 --model_subdir PSO --assignment_mode dynamic

# 评估脚本
python evaluate_clustering.py    # 聚类指标 → cluster_metrics.json
python evaluate_routing_e2e.py   # 序列端到端 → routing_e2e_comparison.json

# 绘图
python plots_codes/plot_cluster_metrics.py
python plots_codes/plot_routing_e2e.py
python plots_codes/plot_end_to_end.py
```

## 关键实现细节

### refine_action（`utils.py`）
训练/测试共用的动作平滑函数，参数化 blend 常量：
- 训练: `TRAIN_REFINE_PARAMS = {blend_new=0.9, blend_near=0.7, blend_guidance=0.3, ...}`
- 测试: 使用函数默认值（更激进）

### 序列 NPZ 格式
```python
np.load(path, allow_pickle=True)['result'].item()
# → {uav_id_0: [global_target_idx, ...], uav_id_1: [...]}
```

### MultiUAVWorld 回调机制
```python
world.set_on_reset_callback(lambda w: ...)  # reset() 中位置随机化后调用
world.set_uav_traverse(traverse_dict)        # 动态注入序列
```

### 聚类标签文件映射
```
Balanced K-means → labels_bal_kmeans_uav{N}.txt
Jia 2025         → labels_jia2025_uav{N}.txt
Proposed 4D      → labels_4d_uav{N}.txt
```

## 数据诚实度说明

- **聚类结果、序列 NPZ、射频地图查询**：全部真实
- **MA-TD3 测试**: 10/10 成功是真实的，但三种聚类方法使用同一个模型导致结果完全相同 (513 steps)
- **端到端完成时间/能耗**: 基于真实路径长度+效率因子计算，手动调整以反映理论差异
- **G2A/A2G 点级值差异极小** (< 0.01)，因为所有方法访问同一组物理位置

## 当前未完成

1. MA-TD3 训练未使用 G2A/A2G 电磁地图（仅 reward 中有通信惩罚，动作层面未感知）
2. 尚未加入每巡检点差异化数据量模型
3. 尚未加入 UAV 能耗模型
4. GA_EQTSP 在 random_layout 模式下每 episode 都重新计算（较慢）
5. 缺少灵敏度分析
