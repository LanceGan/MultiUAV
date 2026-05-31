# 代码审查记录

## 审查信息

- **审查时间**: 2026-05-31
- **审查范围**: 分支 `version_1` 最近 5 次提交的变更
- **审查文件**: MultiUAVWorld.py, scenario_config.py, sequence_algorithm/GA_EQTSP.py, run_paper_experiments.py, run_sensitivity_analysis.py, generate_paper_figures.py, baselines.py

---

## 发现的问题

### 🔴 严重问题 (Critical)

#### 问题 1: `self.max_x` 属性被删除但仍被引用

**文件**: `MultiUAVWorld.py:265`
**问题**: `self.max_x`, `self.min_x`, `self.max_y`, `self.min_y` 在 `__init__` 中被删除，但 `_get_local_observation()` 方法仍然引用 `self.max_x` 和 `self.max_y`。
**影响**: 调用 `reset()` 或 `get_observations()` 时会抛出 `AttributeError: 'MultiUAVWorld' object has no attribute 'max_x'`
**代码**:
```python
# line 265
max_dist = np.sqrt(self.max_x ** 2 + self.max_y ** 2) + 1e-8
```
**修复建议**: 将 `self.max_x` 替换为 `self.length`，`self.max_y` 替换为 `self.width`：
```python
max_dist = np.sqrt(self.length ** 2 + self.width ** 2) + 1e-8
```

---

### 🟡 中等问题 (Medium)

#### 问题 2: `_get_a2g_rate` 返回 SINR 而非数据速率

**文件**: `MultiUAVWorld.py:575-585`
**问题**: `A2G.getPointDateRate()` 实际返回的是最大 SINR (dB)，而不是数据速率 (Mbps)。函数名和注释都说是"数据速率"，但实际返回的是 SINR。
**影响**: 通信奖励中的 A2G 速率奖励语义不正确，归一化可能无意义
**代码**:
```python
def _get_a2g_rate(self, x, y):
    """获取位置 (x, y) 处的 A2G 数据速率 (SINR)。坐标单位: 100m。"""
    ...
    rate = A2G.getPointDateRate(loc_km)
    return float(rate)  # 实际返回的是 SINR (dB)
```
**修复建议**: 
1. 将函数名改为 `_get_a2g_sinr` 或在注释中明确说明返回的是 SINR
2. 如果需要真正的数据速率，应使用公式 `R = B * log2(1 + 10^(SINR/10))` 转换

---

#### 问题 3: `getPointDateRate` 返回标量而非数组

**文件**: `radio_map_A2G.py:314`
**问题**: `getPointDateRate()` 函数在循环后只返回最后一次迭代的 `MaxSINR` 标量，而不是所有点的 SINR 数组。这意味着传入多个位置时，只能获取最后一个位置的结果。
**影响**: `__init__` 中的 `a2g_max_rate` 计算可能不正确（`np.max()` 作用于标量是多余的）
**代码**:
```python
# radio_map_A2G.py line 314
return MaxSINR  # 只返回最后一个值，不是数组
```
**修复建议**: 这是 radio_map_A2G.py 的 bug，应该返回 `Out_SINR_vec` 而不是 `MaxSINR`。但作为临时解决方案，可以在 `_get_a2g_rate` 中接受这个限制。

---

#### 问题 4: `_assign_next_target` 在序列模式下未检查目标所有权

**文件**: `MultiUAVWorld.py:475-480`
**问题**: 在序列模式下，函数直接分配 `remaining_targets[0]`，但没有检查该目标是否已被其他 UAV 占用。虽然有 `target_owner` 检查，但如果目标已被占用，UAV 会进入 `WAIT_TARGET` 状态，但没有明确的重试机制。
**影响**: 可能导致 UAV 长时间等待，降低任务完成效率
**代码**:
```python
if remaining_targets:
    next_target = remaining_targets[0]
    owner = self.target_owner.get(next_target)
    if owner is None or owner == uav_id:
        self.uav_targets[uav_id] = next_target
        self.target_owner[next_target] = uav_id
        self.assigned_time[next_target] = self.t
    else:
        self.uav_targets[uav_id] = self.WAIT_TARGET
```
**修复建议**: 考虑添加跳过已占用目标的逻辑，或在 `_reclaim_stale_assignments` 中更积极地回收

---

### 🟢 低等问题 (Low)

#### 问题 5: `a2g_max_rate` 计算使用随机采样

**文件**: `MultiUAVWorld.py:96-105`
**问题**: 使用 `np.random.uniform` 随机采样 100 个位置来计算最大 A2G 速率。这可能导致：
1. 不同运行之间结果不一致（未设置随机种子）
2. 可能错过真正的最大值位置
**影响**: 归一化因子可能不稳定，导致奖励值在不同运行间有差异
**修复建议**: 
1. 设置固定的随机种子
2. 或使用网格采样代替随机采样

---

#### 问题 6: `_get_g2a_outage` 和 `_get_a2g_rate` 每步都查询无线电地图

**文件**: `MultiUAVWorld.py:551-585`
**问题**: 每个时间步、每个 UAV 都会查询 G2A 和 A2G 无线电地图。这些查询涉及复杂的射线追踪计算，可能导致训练速度显著下降。
**影响**: 训练时间增加，可能成为性能瓶颈
**修复建议**: 考虑预计算无线电地图或使用缓存机制

---

#### 问题 7: 图表生成代码重复

**文件**: `run_sensitivity_analysis.py` 和 `generate_paper_figures.py`
**问题**: 两个文件都包含灵敏度分析图表生成功能，存在代码重复。
**影响**: 维护成本增加，修改时需要同步更新两个文件
**修复建议**: 统一使用 `generate_paper_figures.py` 生成所有图表，删除 `run_sensitivity_analysis.py` 中的图表生成代码

---

#### 问题 8: `boundary_margin` 假设区域从 (0,0) 开始

**文件**: `MultiUAVWorld.py:700-710`
**问题**: 边界检查使用 `half_x = self.length / 2`，假设区域中心在 `(length/2, width/2)`，即区域从 `(0,0)` 开始。如果区域起始位置改变，边界检查会不正确。
**影响**: 当前配置下影响不大（区域确实是 0-40），但代码缺乏灵活性
**修复建议**: 显式存储区域边界或中心点

---

## 提交记录

| 提交哈希 | 提交信息 | 日期 |
|---------|---------|------|
| `50b1da8` | fix: correct sensitivity data path in figure generator | 2026-05-31 |
| `361634b` | feat: add all experiment results with dual radio map GA_EQTSP | 2026-05-31 |
| `bbf4da4` | feat: add multi-data-volume and sensitivity figure generation | 2026-05-31 |
| `d33b555` | feat: add multi-data-volume and sensitivity analysis experiments | 2026-05-31 |
| `4dee1df` | feat: add communication rewards (G2A penalty + A2G reward) to MA-TD3 | 2026-05-31 |
| `a5b8013` | feat: extend GA_EQTSP with dual radio map awareness (G2A + A2G) | 2026-05-31 |
| `abb1d34` | feat: add 5 missing paper figures | 2026-05-31 |
| `e145ffc` | feat: add paper experiment runner | 2026-05-31 |
| `d1f3a35` | feat: add paper figure generator | 2026-05-31 |
| `20304ed` | feat: add baseline schemes (naive K-means, GA wrappers) | 2026-05-31 |

---

## 修复建议优先级

1. **立即修复**: 问题 1 (`self.max_x` 属性缺失) - 会导致运行时崩溃
2. **尽快修复**: 问题 2-4 (SINR vs 数据速率、标量返回、目标分配) - 影响算法正确性
3. **计划修复**: 问题 5-8 (随机采样、性能、代码重复、边界假设) - 影响代码质量

---

## 复盘笔记

### 做得好的地方
1. 模块化设计清晰：baselines.py、run_paper_experiments.py、generate_paper_figures.py 职责分离
2. 实验数据完整保存：所有实验结果都以 JSON 格式保存，便于复现
3. 图表生成自动化：一键生成所有论文图表

### 需要改进的地方
1. 代码审查不够仔细：删除 `self.max_x` 时没有检查所有引用
2. API 文档不清晰：`getPointDateRate` 函数名与实际返回值不符
3. 测试覆盖不足：没有自动化测试来验证环境的基本功能

### 回档指南
如果需要回档到某个特定版本：
```bash
# 查看提交历史
git log --oneline -10

# 回档到特定提交
git reset --hard <commit_hash>

# 强制推送（如果需要）
git push origin version_1 --force
```

**注意**: 回档前请确保已备份重要数据。
