# 多无人机巡检

## 现存问题
+ 巡检序列结果有点问题，理论来说是我们提出的算法更佳，现在出现了混乱的轨迹(√目前改变了聚类算法，现在的巡检序列看起来还不错)
+ 在增加无人机数目并扩大巡检区域大小时，没有改变BS的位置(√已解决)
+ 目前看起来A2G地图还有一些问题(×，SINR太规律了，理论上来说应该和G2A的SINR差不多)

## 改进点
+ 聚类算法优化(已改善，基于Jia等人的文献进行了改进)
1. 四维特征向量重构 (4D Feature Vector):对于每一个巡航点 $w_i$，我们提取一个四维特征向量 
    + $S_i = (d_i, p_i, r_i, q_i)$：$d_i = (x_i, y_i)$：地理坐标，用于控制无人机的基础物理飞行距离。
    + $p_i$：G2A 下行连通性特征。可以使用该点在 G2A 地图上的中断概率 $P_{out}$ 来表示。
    + $r_i$：A2G 上行卸载特征。使用该点在 A2G 地图上的平均上行可达数据速率 $R_m$（或信噪比 SNR）来表示。
    + $q_i$：通信卸载量。即该巡航点需要采集和回传的数据包大小 $I_i$
2. 改进版方差距离函数 (Variance-Based Distance Function)为了在将巡航点 $i$ 分配给无人机簇 $j$ 时实现真正的“负载均衡”，我们沿用 Jia 等人的方差最小化思想，将距离函数 $L_{ij}$ 扩展为包含四个维度的加权方差和：
    $$L_{ij} = \kappa_{ij} + \omega_1 \mu_{ij} + \omega_2 \varpi_{ij} + \omega_3 \nu_{ij}$$
    在计算时，假设将巡航点 $i$ 加入到簇 $j$ 中（该簇当前包含 $z_j$ 个点），各项的物理意义与计算方式如下：
    + $\kappa_{ij}$：物理位置的方差。衡量簇内地理分布的紧凑程度。
    + $\mu_{ij}$：G2A 下行中断概率的方差。定义为 $\mu_{ij} = \frac{1}{z_j}\sum_{k=1}^{z_j}(p_k - \bar{p})^2$。这保证了分配给某架无人机的所有点中，不会出现连通性极端恶劣的任务群。
    + $\varpi_{ij}$：A2G 上行通信速率的方差。定义为 $\varpi_{ij} = \frac{1}{z_j}\sum_{k=1}^{z_j}(r_k - \bar{r})^2$。确保无人机在执行任务时，上行带宽资源的分布相对均匀。
    + $\nu_{ij}$：通信卸载量的方差。定义为 $\nu_{ij} = \frac{1}{z_j}\sum_{k=1}^{z_j}(q_k - \bar{q})^2$。严格防止某一架无人机分配到海量数据导致电量提前耗尽。
    
    (注：$\omega_1, \omega_2, \omega_3$ 为人为设定的惩罚权重系数，后续在 Python 或 MATLAB 中进行系统仿真时，可以通过网格搜索来微调这几个超参数，以达到最佳的均衡效果。)
    
## 最近的代码与训练优化记录
下面列出在本次会话中对代码、环境和训练流程所做的所有可复现优化与变更，含受影响文件与简要说明，便于记录与复现。

- 修复逻辑错误
    - 文件：`MultiUAVWorld.py`
    - 改动：修正 `_on_reach_target` 中对动态分配与预定义序列分支的条件判断，确保动态贪心分配时释放 reservation（`target_owner`/`assigned_time`），预定义序列时按序继续分配。

- 奖励函数稠密化与形状调整
    - 文件：`MultiUAVWorld.py`（函数 `_compute_rewards`）
    - 改动摘要：
        - 放大并保留原有的“前进奖励”；加入方向性奖励（朝目标方向运动的点积分量）以鼓励朝向目标的移动。
        - 增加基于与目标接近度的稠密奖励（归一化），减少停留惩罚幅度，加入邻机接近惩罚（避免碰撞/过近），并显著提高到达目标与团队完成的奖励量级（到达/终点奖励/团队 bonus）。
        - 目的：解决奖励稀疏与信号弱导致的学习难以收敛问题，改善早期探索信号。

- 训练端兼容性修复（回放数据格式）
    - 文件：`MATD3.py`
    - 改动：`train()` 中兼容 replay 中保存的“每智能体独立奖励/完成标志”格式；在必要时对 per-agent rewards/dones 做 team-level 聚合（例如取均值或最大）用于 TD 目标计算，避免训练端假设格式不一致导致的错误。

- Warmup 策略调整（热启动填充回放池）
    - 文件：`Train_MulUAV.py`
    - 改动：将 warmup（启发式策略生成成功样本）从“每回合重复执行”改为“只在训练开始时一次性执行（或当 replay 小于阈值时）”。
    - 目的：避免每集注入大量启发式样本导致 replay 偏向启发式策略，保持训练样本多样性与真实学习信号；同时快速为 replay 提供有效样本以启动训练。

- 训练超参与日志改进（便于调试）
    - 文件：`Train_MulUAV.py`
    - 改动：降低训练启动阈值 `train_memory_size`（示例中用于调试），增加训练频率 `train_freq`，记录 TensorBoard 标量（`Episode_Reward`, `Completed_Targets`, `Train/critic_loss`, `Train/q_value_mean`）。引入自适应探索 `AdaptiveExploration`（检测停滞自动增加噪声）。

- 临时调试脚本与验证
    - 新增文件：`run_env_debug.py`（用于无 PyTorch 的环境级快速验证，验证 reward 与分配逻辑能使启发式策略完成任务）。
    - 运行结果：启发式策略在环境与新奖励下能完成全部目标，验证了环境/奖励修正方向性正确。

- 清理与保存
    - 在最终提交前移除多余的临时打印（保留关键训练日志），并在训练结束处保存模型至 `./results/models/MA-TD3/`。

## 建议与注意事项
- Warmup：建议只在训练开始填充一次或在 replay 小于阈值时运行；避免每集重复注入启发式数据。
- 奖励尺度：当前 reward 放大较多，若 critic_loss 波动/变大，建议对 reward 做归一化或缩放，或对 critic 学习率/梯度裁剪进行微调。
- 日志与可视化：建议将每集 reward/critic_loss/q_mean/完成数导出为折线图以便长期监控（我可以为你生成这些图）。

如果你希望我把上述改动整理为一个 Pull Request 或继续生成训练曲线图，请告诉我你更希望的下一步。 


## Train_MulUAV.py 与 Test_MulUAV.py 参数说明

本项目中，与距离、半径、通信范围、安全距离等相关的变量通常以“百米”为单位。例如，`safe_distance=0.1` 表示 10m，`comm_range=5.0` 表示 500m。

### Train_MulUAV.py 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--uav_num` | `3` | 无人机数量。代码会根据它自动选择场景规模：2 架对应 20 个巡检点，3 架对应 30 个巡检点，4 架对应 40 个巡检点。 |
| `--uav_h` | `1.0` | 无人机飞行高度。按项目单位理解，`1.0` 通常表示 100m。 |
| `--gamma` | `0.99` | 强化学习折扣因子。越接近 1，越重视长期回报；越小，越重视当前奖励。 |
| `--buffer` | `500000` | Replay Buffer 最大容量。越大，能保存更多历史经验，但占用内存更多。 |
| `--net_width` | `256` | Actor 网络隐藏层宽度。Actor 负责输出每架无人机的动作。 |
| `--critic_width` | `512` | Critic 网络隐藏层宽度。Critic 负责估计动作价值，通常可以比 Actor 更宽。 |
| `--exploration_strategy` | `adaptive` | 探索噪声策略。目前主要使用 `adaptive`，会根据训练表现动态调整噪声。 |
| `--min_exploration` | `0.05` | 最小探索噪声。防止训练后期完全没有探索。 |
| `--max_exploration` | `0.25` | 最大探索噪声。训练初期或停滞时噪声不会超过这个值。 |
| `--safe_distance` | `0.1` | 无人机之间的安全距离，单位为百米。`0.1` 即 10m。 |
| `--comm_range` | `5.0` | 通信/感知范围，单位为百米。`5.0` 即 500m。 |
| `--total_episode` | `3000` | 最大训练回合数。实际训练可能提前结束，例如达到 stable 成功率阈值。 |
| `--T` | `2500` | 每个 episode 的最大步数。如果任务未完成但达到 `T`，该回合结束。 |
| `--warmup` | `80` | 训练前用启发式策略跑多少个 episode 来填充 replay buffer。越大越容易冷启动成功，但训练前准备更久。 |
| `--train_memory_size` | `4000` | Replay Buffer 至少积累多少条样本后才开始正式训练。 |
| `--train_freq` | `2` | 每隔多少个环境 step 执行一次网络训练。`2` 表示每 2 步训练一次。 |
| `--warmup_train_steps` | `1500` | warmup 数据收集完后，先额外训练多少次网络，相当于 bootstrap。 |
| `--guided_action_prob_start` | `0.25` | 训练初期随机使用启发式动作的概率。`0.25` 表示初期约 25% 的概率借助启发式动作。 |
| `--guided_action_decay_episodes` | `1000` | 启发式动作概率衰减到 0 所需 episode 数。越大，启发式辅助持续越久。 |
| `--guidance_close_radius` | `3.0` | 目标附近启发式动作混合半径，单位为百米。主要帮助无人机最后阶段稳定进入目标阈值。 |
| `--train_smooth_decay_episodes` | `1500` | 训练侧轨迹平滑修正持续多少 episode。超过后不再强制修正动作，让策略更自主。 |
| `--train_guidance_radius` | `5.0` | 训练侧轨迹修正半径，单位为百米。在这个范围内更强地引导动作朝目标方向收敛。 |
| `--train_near_target_radius` | `0.5` | 训练侧近目标强修正半径，单位为百米。`0.5` 即 50m，主要防止接近目标时绕圈或过冲。 |
| `--train_max_turn_deg` | `16.0` | 训练侧单步最大转向角，单位为度。越小轨迹越平滑，但太小可能导致转弯不够灵活。 |
| `--stable_window` | `100` | stable 成功率统计窗口。会统计最近 100 个 episode 的成功率。 |
| `--stable_success_threshold` | `0.95` | 触发 stable 模型保存和提前结束的成功率阈值。`0.95` 表示最近窗口成功率达到 95% 即认为稳定。 |
| `--model_root` | `./results/models/MA-TD3` | 模型保存根目录。 |
| `--model_subdir` | `Ours` | 模型保存子目录。最终模型通常保存到 `model_root/UAV_x/model_subdir/`。 |

训练参数调节建议：

| 目标 | 主要调整参数 |
|---|---|
| 更容易学会完成任务 | `warmup`、`guided_action_prob_start`、`guided_action_decay_episodes`、`guidance_close_radius` |
| 轨迹更平滑 | `train_smooth_decay_episodes`、`train_guidance_radius`、`train_max_turn_deg` |
| 训练更快开始 | 降低 `warmup`、`train_memory_size`、`warmup_train_steps` |
| 保守稳定 | 增大 `stable_window`，维持或提高 `stable_success_threshold` |
| 保存到不同目录 | 修改 `model_root` 和 `model_subdir` |

训练示例：

```powershell
python Train_MulUAV.py --uav_num 4 --model_subdir Ours
```

### Test_MulUAV.py 参数

测试脚本中的命令行参数会传入 `test_matd3_model(...)`。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--model_episode` | `auto` | 要加载的模型版本。可选 `auto`、`stable`、`best`、`final` 或具体 episode 数字。`auto` 会优先找 `stable`，再找 `best`，最后找 `final`。 |
| `--uav_num` | `3` | 测试无人机数量。必须和训练模型对应，否则 Actor 数量和场景可能不匹配。 |
| `--test_episodes` | `10` | 测试回合数。越大统计越稳定，但耗时越长。 |
| `--model_root` | `./results/models/MA-TD3` | 模型根目录。 |
| `--model_subdir` | `Ours` | 模型子目录。测试会优先读取 `model_root/UAV_x/model_subdir/`。如果子目录没有模型但旧目录有模型，会回退到 `model_root/UAV_x/`。 |
| `--T` | `2500` | 单个测试 episode 的最大步数。 |
| `--safe_distance` | `0.1` | 测试环境中的安全距离，单位为百米。建议和训练保持一致。 |
| `--comm_range` | `5.0` | 测试环境中的通信/感知范围，单位为百米。建议和训练保持一致。 |
| `--pure_policy` | 默认关闭 | 是否禁用测试时轨迹修正。加上该 flag 后，测试只使用模型原始动作，不做几何平滑后处理。 |
| `--trajectory_profile` | `smooth` | 测试侧轨迹后处理档位。当前支持 `smooth`、`balanced`、`agile`。 |
| `--guidance_radius` | `None` | 手动覆盖轨迹修正半径，单位为百米。如果不传，则使用 `trajectory_profile` 对应默认值。 |
| `--near_target_radius` | `None` | 手动覆盖近目标强修正半径，单位为百米。如果不传，则使用档位默认值。 |
| `--max_turn_deg` | `None` | 手动覆盖单步最大转向角，单位为度。如果不传，则使用档位默认值。 |

`trajectory_profile` 档位说明：

| 档位 | 含义 |
|---|---|
| `smooth` | 最平滑，默认推荐。更像真实飞行，倾向直线段和大曲率半径转弯。 |
| `balanced` | 平滑和灵活性折中。 |
| `agile` | 更灵活，允许更快转向，但轨迹可能更不平滑。 |

常用测试命令：

```powershell
python Test_MulUAV.py --uav_num 3 --model_episode stable --trajectory_profile smooth
```

如果想观察模型原始策略，而不使用测试脚本的轨迹修正：

```powershell
python Test_MulUAV.py --uav_num 3 --model_episode stable --pure_policy
```

如果轨迹仍然不够平滑，可以进一步限制转向角：

```powershell
python Test_MulUAV.py --uav_num 3 --model_episode stable --trajectory_profile smooth --max_turn_deg 12
```

### 容易混淆的参数

| 参数 | 说明 |
|---|---|
| `model_episode` | 控制加载哪一版权重，例如 `stable`、`best`、`final`。 |
| `model_subdir` | 控制去哪一个目录找权重，例如 `Ours`、`PSO`、`GA`。 |
| `train_guidance_radius` | 训练时动作平滑/引导半径。 |
| `guidance_radius` | 测试时动作平滑/引导半径。 |
| `pure_policy` | 只影响测试，不影响训练。开启后更能反映模型原始策略，但也更容易出现波浪线或绕弯。 |
| `safe_distance`、`comm_range`、各种 `radius` | 都按百米单位理解，不是米。 |
