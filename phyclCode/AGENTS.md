# Repository Guidelines

## Project Structure & Module Organization
- `app/`: GUI and CLI entry (`python -m app`), Qt widgets, data loading, and demo metrics.
- `models/`, `losses/`: PhyCL-Net architecture modules and custom objectives.
- `scripts/`: Training/eval helpers (`train_4090.*`, `train_baselines.py`, `eval_noise_robustness.py`), packaging (`build_exe.ps1`), and figure/paper utilities.
- `template/`: Paper assets (LaTeX, figures, bib) and reference audits; edit only for publications.
- `docs/`: Static figures for documentation or demos.
- `PhyCL_Net_experiments.py`: Main experiment runner invoked by scripts.
- Data/checkpoints stay local under `data/` and `outputs/`.

## Environment Requirements
- All project commands and scripts must run inside the `PhyCL` conda environment (`conda activate PhyCL` first or `conda run -n PhyCL <cmd>` for one-offs).
- Recreate `PhyCL` with `conda create -n PhyCL python=3.11 -y` before working if it is missing.

## Build, Test, and Development Commands
- Env: `python -m venv .venv`; `.venv\Scripts\activate`; `pip install -r requirements-demo.txt` (CPU torch).
- GUI demo: `python -m app --gui` to compare time-domain vs spectral checkpoints.
- CLI metric: `python -m app --sisfall-root ./data/SisFall --ckpt-time <pth>` for headless accuracy/latency.
- Training presets (Linux/WSL): `bash scripts/train_4090.sh dryrun|quick|full|ablation` -> logs/artifacts in `outputs/`.
- Baselines: `python scripts/train_baselines.py <args>`; noise tests: `python scripts/eval_noise_robustness.py ...`.
- Package EXE: `pwsh scripts/build_exe.ps1` produces `dist/PhyCLNetDemo/`.

## Coding Style & Naming Conventions
- Python, PEP 8, 4-space indent; keep functions small and typed (`@dataclass`, `Path`, explicit tensor shapes).
- Snake_case for files/args; descriptive names (e.g., `ckpt_time`, `sample_rate_hz`); prefer pure helpers.
- Preserve device/precision flags (`--amp`, CUDA env) and seed handling; keep defaults consistent between CLI and GUI.
- UI strings are bilingual; maintain phrasing and Qt signal wiring.

## Testing Guidelines
- No automated suite; validate GUI flow and CLI metrics after changes.
- For training changes, run `scripts/train_4090.sh dryrun` to confirm data loading/AMP; review `outputs/<run>/train.log`.
- Recheck latency/operating-point prints when windows lack labels; ensure checkpoints load for both time/spectral paths.
- Avoid committing datasets or checkpoints; keep large artifacts local.

## Commit & Pull Request Guidelines
- Commit messages: concise, imperative (e.g., "Guard SisFall selection", "Tune LOSO defaults").
- PRs: summarize scope, list commands/tests run, link issues/notes, add GUI screenshots when UI changes.
- Keep diffs focused; separate paper-template edits from code; call out breaking CLI arg or config changes explicitly.
