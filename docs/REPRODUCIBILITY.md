# Reproducibility (PhyCL-Net / `docs/main.tex`)

This repository treats `docs/main.tex` as the single source of truth for model naming, metrics, and reporting protocol.

## Environment (SCI666, GPU)

1. Activate conda env:
   - `conda activate SCI666`
2. Use the Tsinghua mirror:
   - `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`
3. Install deps:
   - `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
4. Verify GPU:
   - `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`

## Dataset

SisFall is public. Download it and place under `./data` (the training CLI uses `--data-root ./data`).

## Main Paper Commands

- **PhyCL-Net (ours, LOSO)**:
  - `python code/PhyCL-Net_experiments.py --dataset sisfall --data-root ./data --model phycl_net --eval-mode loso --seeds 42 123 456 789 1024 --epochs 50 --batch-size 256 --lr 0.004 --warmup-epochs 10 --weighted-loss --amp --use-tfcl --out-dir ./outputs/phycl_net`
- **MSPA-FAA-PDK (spectral baseline, LOSO)**:
  - `python code/PhyCL-Net_experiments.py --dataset sisfall --data-root ./data --model mspa_faa_pdk --eval-mode loso --seeds 42 123 --epochs 50 --batch-size 256 --lr 0.004 --warmup-epochs 10 --weighted-loss --amp --use-tfcl --out-dir ./outputs/mspa_faa_pdk_baseline`

Each run writes machine-readable artifacts under `outputs/<run_dir>/` including `experiment_config.yaml`, `experiment.log`, fold-level JSONs, and `ckpt_best_seed*_loso_SA*.pth` checkpoints.

## Figures / Analyses

- Noise robustness:
  - `python code/scripts/eval_noise_robustness.py --ckpt outputs/phycl_net/ckpt_best_seed456_loso_SA01.pth --data-root ./data --figure-dir ./figures/demo`
- Paper figure generation helper:
  - `python code/scripts/generate_paper_figures.py --phycl-net-ckpt outputs/phycl_net/ckpt_best_seed456_loso_SA01.pth --data-root ./data --output-dir ./figures/demo`

## Notes

- `code/phycl_net_experiments.py` is a thin wrapper that loads `code/PhyCL-Net_experiments.py` (hyphen-safe import).
- Large checkpoint sets can be published as GitHub Release assets together with SHA-256 checksums for integrity verification.
