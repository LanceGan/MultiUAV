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

#### 问题 2: `_get_a2g_rate` 返回 SINR 而非数据速率 ✅ 已修复

**文件**: `MultiUAVWorld.py:575-585`
**问题**: `A2G.getPointDateRate()` 实际返回的是最大 SINR (dB)，而不是数据速率 (Mbps)。函数名和注释都说是"数据速率"，但实际返回的是 SINR。
**影响**: 通信奖励中的 A2G 速率奖励语义不正确，归一化可能无意义
**修复状态**: ✅ 已修复 - 使用公式 `R = B * log2(1 + 10^(SINR/10))` 将 SINR 转换为数据速率
**修复代码**:
```python
sinr_db = A2G.getPointDateRate(loc_km)  # 实际返回 SINR (dB)
rate_mbps = self.BandWidth * np.log2(1 + 10 ** (float(sinr_db) / 10.0))
```

---

#### 问题 3: `getPointDateRate` 返回标量而非数组 ✅ 已规避

**文件**: `radio_map_A2G.py:314`
**问题**: `getPointDateRate()` 函数在循环后只返回最后一次迭代的 `MaxSINR` 标量，而不是所有点的 SINR 数组。这意味着传入多个位置时，只能获取最后一个位置的结果。
**影响**: 这是会议论文代码的遗留问题，但不影响当前使用
**规避方案**: 在 `a2g_max_rate` 计算中改为逐点查询，避免依赖数组返回值

---

#### ~~问题 4: `_assign_next_target` 在序列模式下未检查目标所有权~~ ❌ 非问题

**文件**: `MultiUAVWorld.py:475-480`
**说明**: 序列模式下，每个无人机的待巡检点已预先分配好（通过聚类算法），各无人机之间的巡检点相互独立，不存在冲突。因此 `target_owner` 检查在此模式下是冗余的保护，不会导致实际问题。

---

### 🟢 低等问题 (Low)

#### 问题 5: `a2g_max_rate` 计算使用随机采样 ✅ 已修复

**文件**: `MultiUAVWorld.py:96-105`
**问题**: 使用 `np.random.uniform` 随机采样位置来计算最大 A2G 速率，可能导致结果不一致。
**修复状态**: ✅ 已修复 - 设置固定随机种子 `np.random.seed(42)`，并改为逐点查询以规避问题 3

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

1. ✅ 已修复: 问题 1 (`self.max_x` 属性缺失) - 会导致运行时崩溃
2. ✅ 已修复: 问题 2 (SINR vs 数据速率) - 应用转换公式
3. ✅ 已规避: 问题 3 (标量返回) - 改为逐点查询
4. ❌ 非问题: 问题 4 (目标分配) - 序列模式下各无人机独立
5. ✅ 已修复: 问题 5 (随机采样) - 设置固定随机种子
6. **计划修复**: 问题 6-8 (性能、代码重复、边界假设) - 影响代码质量

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
