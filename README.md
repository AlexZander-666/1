# PhyCL-Net Data Availability Archive
Data availability archive for the JEC submission “Physics-Guided Contrastive Learning for Low-Latency On-Device Wearable Fall Detection”. This repository provides the source code, derived experimental results, figures, and reproducibility instructions; trained checkpoints are distributed via GitHub Releases.

## Environment
- Activate env before any install/run: `conda activate SCI666`
- Use Tsinghua PyPI mirror: `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`
- Install deps: `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
- GPU check: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`

## Quick Start
- Smoke test: `python code1/PhyCL-Net_experiments.py --dataset dryrun --epochs 2 --batch-size 4 --profile`
- Full SisFall LOSO example (PhyCL-Net): `python code1/PhyCL-Net_experiments.py --dataset sisfall --data-root ./data --model phycl_net --eval-mode loso --seeds 42 --epochs 50 --batch-size 256 --lr 0.004 --warmup-epochs 10 --weighted-loss --amp --use-tfcl`
- Baselines: `python code1/scripts/train_baselines.py --data-root ./data --epochs 50`
- Noise robustness check: `python code1/scripts/eval_noise_robustness.py --ckpt outputs/phycl_net/ckpt_best_seed456_loso_SA01.pth --data-root ./data --figure-dir ./figures/demo`

## Data Availability
- Repo: `https://github.com/AlexZander-666/1`
- Checkpoints: publish `ckpt_best_seed*_loso_SA*.pth` as GitHub Release assets with `SHA256SUMS.txt` (see `docs/plans/2026-01-15-github-data-availability.md`).

### Suggested GitHub repo settings (reviewer-friendly)
- Repository name: `phycl-net-data-availability`
- Description: `Data availability archive for PhyCL-Net: code, derived results, figures, and reproducibility instructions; checkpoints via GitHub Releases.`

## Project Layout
- `code1/` - training entry (`PhyCL-Net_experiments.py`; wrapper `phycl_net_experiments.py`), models, losses, and analysis scripts.
- `data/` - local datasets (SisFall/KFall/UniMiB_SHAR/MobiFall); read-only, not versioned.
- `outputs/`, `figures/` - derived results and figures; model weights are excluded by `.gitignore` (see `docs/plans/2026-01-15-github-data-availability.md`).
- `docs/` - reproducibility manifest, submission checklist, analysis plans, and experiment log (`docs/experiments/1.md`).
- `paper/jec/` - LaTeX manuscript (`paper/jec/last2.tex`) and bib.
- `automation/` - queue helpers for training sweeps; `scripts/` - misc utilities.
- `tools/` - auxiliary MCP servers/tools (see their READMEs).

## Reproducibility Notes
- Keep `data/`, `outputs/`, `figures/`, and checkpoints intact; they are ignored by git but required to reproduce reported numbers.
- Use seeds/config paths from `docs/REPRODUCIBILITY_MANIFEST.json`.
- Step-by-step guide: `docs/REPRODUCIBILITY.md`.
- Prefer `python code1/PhyCL-Net_experiments.py ...` for paper-facing commands; `python code1/phycl_net_experiments.py ...` remains as a thin wrapper.
- For manuscript assets, rely on `paper/jec/last2.tex` and `figures/`.

For detailed ground rules, see `AGENTS.md`.
