## PhyCL-Net GUI Demo (时间域主线 vs 谱域 baseline)

本演示只做推理/对比，贴合 `copy666.md` 的部署叙述：时间域主线（去掉 MSPA）与谱域 baseline 的精度–延迟权衡。

更完整的“软件使用说明”（适合发给非开发用户）见：
- `phyclCode/USER_GUIDE.md`

### 1. 依赖
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-demo.txt
```
默认使用 CPU 版 torch（方便单线程延迟对齐）；如需 GPU 可自行安装对应的 CUDA 版。

### 2. 运行 GUI
```bash
python -m app --gui
```
在 GUI 中：
- Data 标签：选择 SisFall 根目录（含 ADL/FALL）或 CSV；自动按 `window_size=512, stride=256, sample_rate=50Hz` 标准化+切窗。
- Models 标签：选择两份 checkpoint  
  - 主线：`ablation mspa:False`（论文主线）  
  - 谱域：`ablation mspa:True`（谱域 baseline）
  设备默认 CPU，如有 GPU 可下拉选择。
- Run/Compare 标签：选择阈值，一键运行单模型或双模型对比；若有标签（SisFall），会给出 Accuracy/Macro-F1、`FPR@TPR=95%`、`TPR@FPR=1%`，并尝试测 p50/p95 单线程延迟。

### 3. CLI 快速测试（可选）
```bash
python -m app --sisfall-root ./data/SisFall --ckpt-time ./outputs/ckpt_best_seed42_fold0.pth
```
仅跑主线模型，打印指标。

### 4. 打包 EXE
```powershell
.\scripts\build_exe.ps1
```
生成 `dist/PhyCLNetDemo/`（onedir）。体积较大属正常（PyTorch+Qt）。

### 5. 输入格式要点
- SisFall：复用训练时的通道选择与按通道 z-score 标准化；自动从文件名解析 subject。
- CSV：至少 3 列数值。可在 GUI 中填采样率；若与 50Hz 不同，会线性重采样后再标准化再切窗。默认无标签，只做概率/阈值统计。

### 6. 已知限制
- 推理仅支持二分类（fall vs non-fall），多分类未处理。
- TFCL 投影头在推理中不使用；checkpoint 若带有这些权重会被忽略。
- 延迟测试是轻量级 CPU 测量，如需严格对标论文可单独运行更长基准。
