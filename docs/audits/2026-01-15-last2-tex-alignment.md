# `paper/jec/last2.tex` Alignment Audit (GitHub Data Availability)

This repository treats `paper/jec/last2.tex` as the single source of truth. This note records what is already aligned, and what still needs to be added to make the GitHub archive fully consistent with the paper.

## Model Naming (Paper ↔ Code)
- **PhyCL-Net (ours)**: `--model phycl_net` (forces `mspa=False`)
- **MSPA-FAA-PDK (spectral baseline)**: `--model mspa_faa_pdk` (forces `mspa=True`)
- Legacy internal name: `--model amsv2` (kept for backward compatibility with older logs/configs; avoid for paper-facing commands)

## Paper Figure Files
`paper/jec/last2.tex` references the following paths, all of which are present:
- `figures/fig01_architecture_and_block.pdf`
- `figures/fig02_pdk_module.pdf`
- `figures/fig03_cross_gate_fusion.pdf`
- `figures/fig04_pareto_tradeoff.pdf`
- `figures/fig05_fold_stability.pdf`
- `figures/fig06_radar.pdf`
- `figures/fig07_tsne.pdf`
- `figures/fig08_attention.pdf`
- `figures/fig09_confusion.pdf`

## Output Directories (Paper-facing)
- **PhyCL-Net results**: `outputs/phycl_net/`
- **MSPA-FAA-PDK results**: `outputs/mspa_faa_pdk_baseline/`

Each directory should contain at minimum:
- `summary_results.json`
- `loso_results_seed*.json`
- `split_stats_seed*.json`
- `experiment_config.yaml`
- `experiment.log`
- `errors_seed*_loso_SA*.csv` (recommended)

## Known Gaps vs `last2.tex` (Must Be Resolved Before Final Submission)
1. **PhyCL-Net five-seed packaging**
   - `last2.tex` reports PhyCL-Net aggregated across **five seeds (42, 123, 456, 789, 1024)**.
   - The repository currently contains LOSO artifacts for PhyCL-Net under `outputs/phycl_net/` for seeds **456/789/1024**.
   - Action: add the missing LOSO-derived artifacts and **best-fold checkpoints** for **seed42** and **seed123** into `outputs/phycl_net/` (or provide them as Release assets and list them in `manifests/PATH_MAPPING.md` + `manifests/SHA256SUMS.txt`).

2. **Efficiency benchmarking protocol artifacts**
   - `last2.tex` reports CPU latency (p50/p95), Params, and FLOPs under a fixed protocol.
   - Action: run `code1/scripts/benchmark_efficiency_cpu.py` on the target environment (Windows 10, PyTorch 2.5.1, single-thread CPU) and archive the resulting JSON next to the Release assets (recommended) so reviewers can verify the reported numbers.

## Reference Plan
- Execution plan: `docs/plans/2026-01-15-github-data-availability.md`
