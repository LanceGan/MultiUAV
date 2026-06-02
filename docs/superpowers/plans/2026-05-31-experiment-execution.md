# Experiment Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run complete experiments for the multi-UAV cooperative inspection paper, including routing optimization, MA-TD3 training, baseline comparisons, and sensitivity analysis.

**Architecture:** Four-phase experiment pipeline — (1) Clustering + Routing to generate inspection sequences, (2) MA-TD3 training to optimize flight trajectories, (3) Baseline comparisons with GA/PSO/ACO, (4) Sensitivity analysis for key parameters. N=2 (20 points) as primary showcase, N=3 (30 points) and N=4 (40 points) for scalability.

**Tech Stack:** Python (NumPy, PyTorch, Matplotlib), Conda environment at `F:\Anaconda3\envs\pytorch\python.exe`

---

## File Structure

| File | Role |
|------|------|
| `run_paper_experiments.py` | Master experiment runner (clustering, routing, multi-data) |
| `run_sensitivity_analysis.py` | Sensitivity analysis script |
| `Train_MulUAV.py` | MA-TD3 training script |
| `Test_MulUAV.py` | MA-TD3 testing script |
| `generate_paper_figures.py` | Figure generation script |
| `scenario_config.py` | Shared scenario configuration |

---

### Task 1: Run Clustering Experiment

**Files:**
- Execute: `run_paper_experiments.py --experiment clustering`

- [ ] **Step 1: Run clustering for N=2,3,4**

```bash
cd F:\Projects\Py\MultiUAV && F:\Anaconda3\envs\pytorch\python.exe run_paper_experiments.py --experiment clustering
```

Expected output:
- `results/paper_experiments/clustering/clustering_results.json`
- `results/paper_experiments/clustering/labels_naive_uav{2,3,4}.txt`
- `results/paper_experiments/clustering/labels_4d_uav{2,3,4}.txt`

- [ ] **Step 2: Verify clustering results**

```bash
cat F:\Projects\Py\MultiUAV\results\paper_experiments\clustering\clustering_results.json
```

Expected: JSON with metrics for N=2,3,4 showing cluster sizes, std_dev, variance, max_min_ratio.

- [ ] **Step 3: Commit clustering results**

```bash
git add results/paper_experiments/clustering/
git commit -m "feat: add clustering experiment results for N=2,3,4"
```

---

### Task 2: Run Routing Experiment (GA_EQTSP Dual Radio Map)

**Files:**
- Execute: `run_paper_experiments.py --experiment routing`

- [ ] **Step 1: Run routing for N=2,3,4**

```bash
cd F:\Projects\Py\MultiUAV && F:\Anaconda3\envs\pytorch\python.exe run_paper_experiments.py --experiment routing
```

Expected output:
- `results/paper_experiments/routing/routing_{GA,PSO,ACO,GA_EQTSP}_uav{2,3,4}.json`
- `results/paper_experiments/routing/routing_summary_uav{2,3,4}.json`

Note: GA_EQTSP with dual radio map queries is slower than other algorithms. Expect ~5-10 minutes per cluster.

- [ ] **Step 2: Verify routing results**

```bash
cat F:\Projects\Py\MultiUAV\results\paper_experiments\routing\routing_summary_uav2.json
```

Expected: JSON with algorithms (GA, PSO, ACO, GA_EQTSP), each containing cluster results and total_path_length.

- [ ] **Step 3: Generate inspection sequences for MA-TD3 training**

The routing experiment generates `best_indices` for each cluster. These need to be converted to `.npz` format for `Train_MulUAV.py`.

Check if sequence files exist:
```bash
ls F:\Projects\Py\MultiUAV\results\datas\sequence\
```

If not, create them from routing results:
```bash
F:\Anaconda3\envs\pytorch\python.exe -c "
import json, numpy as np, os
from scenario_config import UAV_USER_MAP

for n_uav in [2, 3, 4]:
    n_users = UAV_USER_MAP[n_uav]
    routing_path = f'results/paper_experiments/routing/routing_GA_EQTSP_uav{n_uav}.json'
    if os.path.exists(routing_path):
        with open(routing_path) as f:
            data = json.load(f)
        result = {}
        for key, val in data.items():
            if key.startswith('cluster_'):
                cid = int(key.split('_')[1])
                result[cid] = val['best_indices']
        save_path = f'results/datas/sequence/Users_{n_users}_Clusteredsave_path_PathUAV_GAEQTSP_{n_uav}.npz'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, result=result)
        print(f'Saved: {save_path}')
"
```

- [ ] **Step 4: Commit routing results**

```bash
git add results/paper_experiments/routing/
git commit -m "feat: add routing experiment results with dual radio map GA_EQTSP"
```

---

### Task 3: Run MA-TD3 Training (N=2 Primary)

**Files:**
- Execute: `Train_MulUAV.py --uav_num 2 --model_subdir Ours`

- [ ] **Step 1: Train MA-TD3 for N=2 (primary showcase)**

```bash
cd F:\Projects\Py\MultiUAV && F:\Anaconda3\envs\pytorch\python.exe Train_MulUAV.py --uav_num 2 --model_subdir Ours --total_episode 3000
```

Expected: Training runs for 3000 episodes, saves models to `results/models/MA-TD3/UAV_2/Ours/`.

Note: This may take 1-3 hours depending on hardware. Monitor TensorBoard for progress:
```bash
tensorboard --logdir logs/MATD3_uav_2
```

- [ ] **Step 2: Verify training completed**

```bash
ls F:\Projects\Py\MultiUAV\results\models\MA-TD3\UAV_2\Ours\
```

Expected: Files like `matd3_actor0_ep{N}.pth`, `matd3_critic_ep{N}.pth`, `matd3_critic_epstable.pth`.

- [ ] **Step 3: Test trained model for N=2**

```bash
cd F:\Projects\Py\MultiUAV && F:\Anaconda3\envs\pytorch\python.exe Test_MulUAV.py --uav_num 2 --model_episode stable --model_subdir Ours --test_episodes 10
```

Expected: Test results saved to `results/test/UAV_2/Ours/`.

- [ ] **Step 4: Commit models**

```bash
git add results/models/MA-TD3/UAV_2/
git commit -m "feat: add trained MA-TD3 model for N=2"
```

---

### Task 4: Run MA-TD3 Training (N=3,4 for Scalability)

**Files:**
- Execute: `Train_MulUAV.py --uav_num 3,4`

- [ ] **Step 1: Train MA-TD3 for N=3**

```bash
cd F:\Projects\Py\MultiUAV && F:\Anaconda3\envs\pytorch\python.exe Train_MulUAV.py --uav_num 3 --model_subdir Ours --total_episode 3000
```

- [ ] **Step 2: Test trained model for N=3**

```bash
cd F:\Projects\Py\MultiUAV && F:\Anaconda3\envs\pytorch\python.exe Test_MulUAV.py --uav_num 3 --model_episode stable --model_subdir Ours --test_episodes 10
```

- [ ] **Step 3: Train MA-TD3 for N=4**

```bash
cd F:\Projects\Py\MultiUAV && F:\Anaconda3\envs\pytorch\python.exe Train_MulUAV.py --uav_num 4 --model_subdir Ours --total_episode 3000
```

- [ ] **Step 4: Test trained model for N=4**

```bash
cd F:\Projects\Py\MultiUAV && F:\Anaconda3\envs\pytorch\python.exe Test_MulUAV.py --uav_num 4 --model_episode stable --model_subdir Ours --test_episodes 10
```

- [ ] **Step 5: Commit all models**

```bash
git add results/models/MA-TD3/
git commit -m "feat: add trained MA-TD3 models for N=2,3,4"
```

---

### Task 5: Run Baseline Comparison Experiments

**Files:**
- Execute: `run_paper_experiments.py --experiment routing` (already done in Task 2)

The routing experiment already runs GA, PSO, ACO, and GA_EQTSP. The baseline comparison data is in `results/paper_experiments/routing/`.

- [ ] **Step 1: Verify baseline results exist**

```bash
ls F:\Projects\Py\MultiUAV\results\paper_experiments\routing\
```

Expected: Files for each algorithm and UAV count.

- [ ] **Step 2: Generate baseline comparison figures**

```bash
cd F:\Projects\Py\MultiUAV && F:\Anaconda3\envs\pytorch\python.exe generate_paper_figures.py --figure routing
```

Expected: `F:\Projects\Latex\MultiUAV\figs\routing_comparison.png`

- [ ] **Step 3: Commit figures**

```bash
git add F:\Projects\Latex\MultiUAV\figs\
git commit -m "feat: add baseline comparison figures"
```

---

### Task 6: Run Sensitivity Analysis

**Files:**
- Execute: `run_sensitivity_analysis.py`

- [ ] **Step 1: Run GA population sensitivity**

```bash
cd F:\Projects\Py\MultiUAV && F:\Anaconda3\envs\pytorch\python.exe run_sensitivity_analysis.py --analysis ga_population
```

Expected: `results/sensitivity/ga_population/ga_population_sensitivity.json`

- [ ] **Step 2: Run data volume sensitivity**

```bash
cd F:\Projects\Py\MultiUAV && F:\Anaconda3\envs\pytorch\python.exe run_sensitivity_analysis.py --analysis data_volume
```

Expected: `results/sensitivity/data_volume/data_volume_sensitivity.json`

- [ ] **Step 3: Generate sensitivity figures**

```bash
cd F:\Projects\Py\MultiUAV && F:\Anaconda3\envs\pytorch\python.exe run_sensitivity_analysis.py --analysis generate_figures
```

Expected: `results/sensitivity/figures/ga_population_sensitivity.png` and `data_volume_sensitivity.png`

- [ ] **Step 4: Commit sensitivity results**

```bash
git add results/sensitivity/
git commit -m "feat: add sensitivity analysis results and figures"
```

---

### Task 7: Generate All Paper Figures

**Files:**
- Execute: `generate_paper_figures.py --figure all`

- [ ] **Step 1: Generate all figures**

```bash
cd F:\Projects\Py\MultiUAV && F:\Anaconda3\envs\pytorch\python.exe generate_paper_figures.py --figure all
```

Expected figures in `F:\Projects\Latex\MultiUAV\figs\`:
- `clustering_comparison.png`
- `load_balance_comparison.png`
- `routing_comparison.png`
- `training_convergence.png`
- `energy_breakdown.png`
- `ga_convergence.png`
- `adaptive_weight_evolution.png`
- `parameter_sensitivity.png`
- `ga_population_sensitivity.png`
- `data_volume_sensitivity.png`
- `trajectory_comparison_{100,200,300}mb.png`
- `system_architecture.png`

- [ ] **Step 2: Verify all figures exist**

```bash
ls F:\Projects\Latex\MultiUAV\figs\*.png | wc -l
```

Expected: At least 15 PNG files.

- [ ] **Step 3: Commit figures**

```bash
git add F:\Projects\Latex\MultiUAV\figs\
git commit -m "feat: generate all paper figures"
```

---

### Task 8: Update Paper Tables with Experiment Data

**Files:**
- Modify: `F:\Projects\Latex\MultiUAV\main.tex`

- [ ] **Step 1: Update Table I (Load Balancing)**

Read `results/paper_experiments/clustering/clustering_results.json` and update Table I in `main.tex`.

- [ ] **Step 2: Update Table II (Routing Performance)**

Read `results/paper_experiments/routing/routing_summary_uav2.json` and update Table II in `main.tex`.

- [ ] **Step 3: Update Table III (Communication Performance)**

Use routing results to compute communication metrics and update Table III.

- [ ] **Step 4: Update Table IV (Algorithm Runtime)**

Read timing data from routing results and update Table IV.

- [ ] **Step 5: Commit paper updates**

```bash
cd F:\Projects\Latex\MultiUAV && git add main.tex && git commit -m "feat: update paper tables with experiment data"
```

---

## Execution Order

1. **Task 1**: Clustering (5 minutes)
2. **Task 2**: Routing (30-60 minutes)
3. **Task 3**: MA-TD3 N=2 training (1-3 hours)
4. **Task 4**: MA-TD3 N=3,4 training (2-6 hours)
5. **Task 5**: Baseline figures (5 minutes)
6. **Task 6**: Sensitivity analysis (30-60 minutes)
7. **Task 7**: Generate all figures (5 minutes)
8. **Task 8**: Update paper tables (30 minutes)

**Total estimated time: 5-10 hours** (dominated by MA-TD3 training)

---

## Troubleshooting

### Common Issues

1. **`ModuleNotFoundError: No module named 'radio_map_G2A'`**
   - Solution: Run from project root directory `F:\Projects\Py\MultiUAV\`

2. **`NameError: name 'uav' is not defined`**
   - Solution: Already fixed in commit `c429b13`

3. **`AttributeError: 'MultiUAVWorld' object has no attribute 'max_x'`**
   - Solution: Already fixed in commit `4a0777d`

4. **Training not converging**
   - Try increasing `warmup` to 120-200
   - Try increasing `guided_action_prob_start` to 0.3-0.4
   - Check TensorBoard for reward trends

5. **GA_EQTSP very slow**
   - This is expected due to radio map queries
   - Reduce `iteration` to 100 for faster results
