# GitHub Data Availability Execution Plan (PhyCL-Net / `template/1.md`)

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** 将 `template/1.md` 中涉及的实验结果、图表、复现脚本与训练得到的模型权重（LOSO 各折 best checkpoints）整理为一个可公开、可核验、可复现的 GitHub 归档，用于期刊 Data Availability Statement。

**Architecture:** 采用“代码与轻量结果进 Git 仓库 + 大体积权重走 GitHub Release 资产（zip 分卷）”的方式，规避 GitHub 单文件 100MB 限制与 Git LFS 配额风险，同时保留可复现所需的每折权重与逐折结果 JSON/CSV。

**Tech Stack:** Git + GitHub Releases（推荐）/ Git LFS（备选），PowerShell（Windows），`conda activate SCI666`（本项目运行规范），`Get-FileHash`（SHA256 校验）。

---

## Scope：必须覆盖的“与 `template/1.md` / `template/last2.tex` 相关”内容

### A. 论文数据汇总/口径文件（必须）
- `template/1.md`
- `docs/experiments/1.md`（同内容的存档/镜像，建议一并公开）
- `docs/REPRODUCIBILITY.md`
- `docs/REPRODUCIBILITY_MANIFEST.json`
- `README.md`（需补充 Data Availability 入口与 Release 链接）

### B. 图表（必须：`template/1.md` 明确引用的）
- `figures/fine_grained/age_stratification.png`
- `figures/fine_grained/confusion_matrix_34class.png`
- `figures/fine_grained/top_confused_pairs.png`
- `figures/fine_grained/per_class_metrics.csv`
- `figures/fine_grained/analysis_summary.json`
- `figures/fig1_accuracy_vs_params.pdf`
- `figures/noise_robustness_curve.pdf`
- `outputs/robustness_final/Robustness_to_Sensor_Noise.png`
- `outputs/robustness_final/noise_robustness_results.csv`

### C. 实验结果与元数据（必须：支撑 `template/1.md` 中表格/结论）
对每个实验输出目录，至少包含：
- `summary_results.json`
- `loso_results_seed*.json`
- `split_stats_seed*.json`
- `experiment_config.yaml`
- `experiment.log`
- `errors_seed*_loso_SA*.csv`（可选但推荐，便于误差审计）

**`template/1.md`/`template/last2.tex` 涉及的关键输出目录（发布到 GitHub 时建议使用的“论文命名”）：**
- `outputs/mspa_faa_pdk_baseline/`（Spectral baseline；seed 42/123；`ckpt_best*` 共 24；本地来源：baseline 输出目录（建议先重命名为 `outputs/mspa_faa_pdk_baseline/`））
- `outputs/phycl_net/`（`phycl_net`；LOSO 12 折；**五个 seeds (42, 123, 456, 789, 1024)**；建议在公开仓库中整理为单一目录；本地来源可能分散在 `outputs/ablation_no_mspa/` 与历史 run 目录中）
- `outputs/ablation_no_tfcl/`（w/o TFCL）
- `outputs/ablation_no_dks/`（w/o DKS）
- `outputs/ablation_no_faa/`（w/o FAA）
- `outputs/ablation_time_only/`（Time-only）
- `outputs/ablation_freq_only/`（Freq-only）
- `outputs/stage1_inceptiontime_final/`（InceptionTime baseline）
- `outputs/stage1_transformer_final/`（Transformer baseline）
- `outputs/stage1_lstm_final/`（LSTM baseline）
- `outputs/stage1_resnet_final/`（ResNet baseline）
- `outputs/stage1_tcn_final/`（TCN baseline）
- `outputs/lite_amsnet_sa01/`（`template/1.md` 引用的 Lite 模型权重示例）

> 说明：`template/1.md` 中还出现了 `outputs/phycl_net_main`、`outputs/mspa_faa_pdk_baseline`、`outputs/baseline_*` 等“逻辑目录名”。建议在公开仓库中提供一份 **Path Mapping 表**（见下文），将这些逻辑名映射到公开仓库的“论文命名目录”（并注明本地来源目录）。

### D. 复现代码（必须：能从 outputs 复核指标/曲线）
本仓库实际主代码目录为 `code1/`（README/AGENTS.md 里写的 `code/` 在当前工作区不存在）：
- `code1/DMC_Net_experiments.py`
- `code1/models/`、`code1/losses/`
- `code1/scripts/`（至少包括：`eval_noise_robustness.py`、`fine_grained_analysis.py`、`paired_ttest_from_markdown.py`、`train_baselines.py`、`pack_sci_submission.py`、`eval_lite_amsnet_noise.py`）

---

## Recommended Publishing Strategy（GitHub 可落地、审稿人可下载）

### Strategy 1（推荐）：Repo + GitHub Releases（权重走 Release 资产）
**优点**
- 不依赖 Git LFS 配额（本项目权重总量明显 > 1GB）。
- 审稿人可直接下载单个 zip（或分卷）并用 SHA256 验证。

**做法**
- Git 仓库内追踪：代码 + `template/1.md` + `docs/` + `figures/` + 轻量结果 JSON/CSV/LOG。
- 大体积：各实验目录的 `ckpt_best*.pth` 打包为 zip（必要时分卷），上传到 GitHub Release（例如 `paper-v1.0-artifacts`）。

### Strategy 2（备选）：Git LFS（不推荐作为默认）
仅当你确认拥有足够 LFS 存储/带宽配额时使用。否则建议仍走 Releases。

---

## Target GitHub Layout（建议结构）

```
PhyCL-Net-data-availability/
  README.md
  template/1.md
  docs/experiments/1.md
  docs/REPRODUCIBILITY.md
  docs/REPRODUCIBILITY_MANIFEST.json
  figures/
  outputs/
    <run_dir>/
      summary_results.json
      loso_results_seed*.json
      split_stats_seed*.json
      experiment_config.yaml
      experiment.log
      errors_seed*_loso_SA*.csv
    robustness_final/
      noise_robustness_results.csv
      Robustness_to_Sensor_Noise.png
  code/   (可选：将 code1 重命名为 code，保持与 README/AGENTS 一致)
  code1/  (可选：若不想重命名，则保留 code1 并在 README 写清楚)
  manifests/
    PATH_MAPPING.md
    ARTIFACT_MANIFEST.json
    SHA256SUMS.txt
```

---

## Path Mapping（必须写进 `manifests/PATH_MAPPING.md`）

| `template/1.md` / `template/last2.tex` 中的逻辑路径 | 公开仓库建议路径（论文命名） | 本地来源目录（当前工作区） |
|---|---|
| `outputs/mspa_faa_pdk_baseline/...` | `outputs/mspa_faa_pdk_baseline/...` | baseline 输出目录（建议先重命名为 `outputs/mspa_faa_pdk_baseline/`） |
| `outputs/phycl_net_main/...` / `outputs/tdfnet_5seeds/...` | `outputs/phycl_net/...`（统一五个 seeds：42/123/456/789/1024） | `outputs/ablation_no_mspa/...` +（如有）其它历史 run 目录 |
| `outputs/baseline_inceptiontime/...` | `outputs/baseline_inceptiontime/...` | `outputs/stage1_inceptiontime_final/...` |
| `outputs/baseline_transformer/...` | `outputs/baseline_transformer/...` | `outputs/stage1_transformer_final/...` |
| `outputs/baseline_lstm/...` | `outputs/baseline_lstm/...` | `outputs/stage1_lstm_final/...` |
| `outputs/baseline_resnet/...` | `outputs/baseline_resnet/...` | `outputs/stage1_resnet_final/...` |
| `outputs/baseline_tcn/...` | `outputs/baseline_tcn/...` | `outputs/stage1_tcn_final/...` |
| `code/...` | `code/...`（对外统一使用 `code/`；内部可由 `code1/` 同步/重命名） | `code1/...` |

---

## Execution Tasks（可直接照做）

### Task 1: 冻结当前“投稿版本”并记录版本信息
**Files:**
- Modify: `README.md`
- Create: `manifests/ARTIFACT_MANIFEST.json`

**Steps:**
1. 记录当前 commit hash（用于论文/补充材料引用）。
2. 在 `README.md` 增加 “Data Availability” 小节：包含 GitHub Repo URL 与 Release URL（待创建）。
3. 生成 `manifests/ARTIFACT_MANIFEST.json`：列出每个实验目录、包含文件、seeds、folds（12 subjects）与文件校验信息（见 Task 4）。

### Task 2: 生成并核对 “需要公开的文件清单”（以 `template/1.md` 为准）
**Create:** `manifests/PATH_MAPPING.md`

**Steps:**
1. 将上面的 Path Mapping 表写入 `manifests/PATH_MAPPING.md`。
2. 对 `template/1.md` 中出现但当前缺失的文件（例如 `docs/figures/paper/*`、`docs/tables/*`），明确标注为：
   - “未在仓库中生成/不作为公开归档的一部分”，或
   - “由 `code1/scripts/...` 生成（给出确切命令与输出路径）”。

### Task 3: 权重归档策略（只公开 best checkpoints）
**Decision:**
- 默认仅公开 `ckpt_best_seed*_loso_SA*.pth`（与 `template/1.md` 的引用一致）。
- `ckpt_last*` 作为可选项：仅当你需要“严格逐 epoch 复现”或审稿人要求时再公开。

### Task 4: 打包权重到 GitHub Release（并生成 SHA256）
**Create:** `manifests/SHA256SUMS.txt`

**Steps (PowerShell, Windows):**
1. 在公开仓库根目录创建打包目录：
   - `New-Item -ItemType Directory -Force artifacts, manifests | Out-Null`
2. 为每个 run_dir 创建一个 zip（只打包 `ckpt_best*.pth`）：
   - `outputs/mspa_faa_pdk_baseline/` → `ckpts_mspa_faa_pdk_baseline_best.zip`
   - `outputs/phycl_net/` → `ckpts_phycl_net_best.zip`（建议按 seed 分卷）
   - 各 baseline/ablation 目录同理（如需公开）
3. 推荐的最小可复用命令（以 `mspa_faa_pdk_baseline` 为例）：
   ```powershell
   $files = Get-ChildItem outputs/mspa_faa_pdk_baseline -File -Filter 'ckpt_best*.pth'
   Compress-Archive -Path $files.FullName -DestinationPath artifacts/ckpts_mspa_faa_pdk_baseline_best.zip -Force
   ```
4. 若需按 seed 分卷（更便于控制 zip 体积；`phycl_net` 建议默认分卷）：
   ```powershell
   $seeds = @(42, 123)
   foreach ($s in $seeds) {
     $files = Get-ChildItem outputs/mspa_faa_pdk_baseline -File -Filter ("ckpt_best_seed{0}_loso_*.pth" -f $s)
     Compress-Archive -Path $files.FullName -DestinationPath ("artifacts/ckpts_mspa_faa_pdk_baseline_seed{0}.zip" -f $s) -Force
   }
   ```
2. 若单个 zip 超过 GitHub Release 2GB 限制，按 seed 分卷打包（例如 `seed42.zip`、`seed123.zip`）。
3. 对每个 zip 运行 `Get-FileHash -Algorithm SHA256`，写入 `manifests/SHA256SUMS.txt`。
   ```powershell
   Get-ChildItem artifacts -File -Filter '*.zip' |
     ForEach-Object { $h = Get-FileHash $_.FullName -Algorithm SHA256; "{0}  {1}" -f $h.Hash, $_.Name } |
     Out-File manifests/SHA256SUMS.txt -Encoding ascii
   ```
4. 打包后检查 zip 体积，避免上传失败：
   ```powershell
   Get-ChildItem artifacts -File -Filter '*.zip' |
     Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,2)}}
   ```

### Task 5: 创建 GitHub 仓库并上传
**Steps:**
1. 本次目标仓库：`https://github.com/AlexZander-666/1`（默认按独立数据可用性仓库管理）。
2. 提交并推送：代码/文档/图表/轻量结果（JSON/CSV/log）。
3. 创建 GitHub Release：`paper-v1.0-artifacts`，上传 Task 4 的 zip 权重与 `SHA256SUMS.txt`。
4. 打 tag（可选）：`v1.0-paper`，并在 Release 描述中注明对应论文版本与 commit hash。

### Task 6: 复核（最重要：避免投稿后链接不可用/文件不全）
**Steps:**
1. 在另一台机器或空目录 `git clone` 该公开仓库。
2. 下载 Release 权重 zip，使用 `SHA256SUMS.txt` 逐一校验。
3. 至少跑一次“只读验证”：用 `code1/scripts/eval_noise_robustness.py` 或 `code1/scripts/fine_grained_analysis.py` 在一个 checkpoint 上生成输出，确认脚本可运行、路径正确。

---

## Data Availability Statement（可直接粘贴的英文模版）

> The source code, experimental results, figures, and trained model checkpoints supporting the findings of this study are publicly available on GitHub at: **https://github.com/AlexZander-666/1**. The trained checkpoints for all LOSO folds are provided as GitHub Release assets (tag: **paper-v1.0-artifacts**) together with SHA-256 checksums for integrity verification.  
>  
> The datasets used in this study (e.g., SisFall, MobiFall v2.0, UniMiB SHAR, and KFall) are publicly available from their respective owners and are therefore not redistributed in this repository; instructions and links for obtaining the datasets are provided in the repository documentation.
