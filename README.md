# PhyCL-Net Data Availability Archive
Data availability archive for the JEC submission “Physics-Guided Contrastive Learning for Low-Latency On-Device Wearable Fall Detection”. This repository provides the source code, derived experimental results, figures, and reproducibility instructions; trained checkpoints are distributed via GitHub Releases.

## Environment
- Activate env before any install/run: `conda activate SCI666`
- Use Tsinghua PyPI mirror: `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`
- Install deps: `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
- GPU check: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`

## Quick Start
- Smoke test: `python code/PhyCL-Net_experiments.py --dataset dryrun --epochs 2 --batch-size 4 --profile`
- Full SisFall LOSO example (PhyCL-Net): `python code/PhyCL-Net_experiments.py --dataset sisfall --data-root ./data --model phycl_net --eval-mode loso --seeds 42 --epochs 50 --batch-size 256 --lr 0.004 --warmup-epochs 10 --weighted-loss --amp --use-tfcl`
- Baselines: `python code/scripts/train_baselines.py --data-root ./data --epochs 50`
- Noise robustness check: `python code/scripts/eval_noise_robustness.py --ckpt outputs/phycl_net/ckpt_best_seed456_loso_SA01.pth --data-root ./data --figure-dir ./figures/demo`

## Data Availability
- Repo: `https://github.com/AlexZander-666/1`
- Checkpoints: publish `ckpt_best_seed*_loso_SA*.pth` as GitHub Release assets with SHA-256 checksums for integrity verification (see `SHA256SUMS.txt`).

### Suggested GitHub repo settings (reviewer-friendly)
- Repository name: `phycl-net-data-availability`
- Description: `Data availability archive for PhyCL-Net: code, derived results, figures, and reproducibility instructions; checkpoints via GitHub Releases.`

## Project Layout
- `code/` - training entry (`PhyCL-Net_experiments.py`; wrapper `phycl_net_experiments.py`), models, losses, and analysis scripts.
- `data/` - SisFall (public) placed under `data/SisFall`.
- `outputs/` - derived results and model checkpoints for the paper.
- `docs/` - manuscript (`docs/main.tex`), bibliography (`docs/main.bib`), and reproducibility docs.
- `logs/` - auxiliary run logs (optional).

## Reproducibility Notes
- Keep `data/` and `outputs/` intact; they are required to reproduce reported numbers.
- Use seeds/config paths from `docs/REPRODUCIBILITY_MANIFEST.json`.
- Step-by-step guide: `docs/REPRODUCIBILITY.md`.
- Prefer `python code/PhyCL-Net_experiments.py ...` for paper-facing commands; `python code/phycl_net_experiments.py ...` remains as a thin wrapper.
- For manuscript assets, rely on `docs/main.tex`.

For detailed ground rules, see `AGENTS.md`.
