# PhyCLNetDemo 软件使用文档（GUI 推理/对比版）

本软件用于 **PhyCL‑Net 论文主线（去 MSPA）** 与 **谱域 baseline（保留 MSPA）** 的推理演示与 trade‑off 对比。输入支持：
- **SisFall 原始 `.txt`**（带标签，可输出指标）
- **自定义 `.csv`**（默认无标签，输出概率/阈值统计）

> 说明：本软件为离线推理演示工具，不包含训练流程。训练请使用仓库中的训练脚本。

---

## 1. 你会得到哪些文件（打包产物）

打包产物位于：
- `dist/PhyCLNetDemo/PhyCLNetDemo.exe`

重要注意：
- 这是 `--onedir` 打包方式：**必须保留整个 `dist/PhyCLNetDemo/` 目录**，不要只拷贝 `PhyCLNetDemo.exe`。
- 如果需要拷贝给另一台机器，请把 `dist/PhyCLNetDemo/` 整个目录复制过去。

---

## 2. 运行环境要求（用户侧）

- Windows 10/11 64-bit
- 推荐：CPU 推理（更接近“single-thread CPU latency”叙述，且依赖最少）
- 可选：GPU 推理（仅当你构建时打包/包含了 CUDA 版本 torch 且目标机器 GPU/驱动/CUDA 兼容）

---

## 3. 快速开始（3 分钟上手）

你需要准备：
1) 一个数据输入（SisFall 根目录或一个 CSV 文件）
2) 至少一份 checkpoint（主线或谱域 baseline 的 `.pth/.pt`）

操作步骤：
1) 双击运行 `dist/PhyCLNetDemo/PhyCLNetDemo.exe`
2) 进入 `Data` 页，加载数据（SisFall 或 CSV）
3) 进入 `Models` 页，选择并加载模型 checkpoint（可只加载一个）
4) 进入 `Run/Compare` 页，点击“运行主线模型 / 运行谱域模型 / 对比两模型”

软件会在输出框中打印：
- 推理窗口数、阈值
- 若有标签（SisFall）：Accuracy、Macro‑F1、TP/FP/FN/TN、两种 operating point
- 轻量延迟：p50/p95（单线程快速测量）

---

## 4. 模型（checkpoint）准备与选择

你通常需要两份 checkpoint，对应两种结构：

### 4.1 主线模型（去 MSPA）
- 训练参数：`--ablation mspa:False`
- GUI：在 `Models` 页选择 “主线(去 MSPA) ckpt” 并加载

### 4.2 谱域 baseline（保留 MSPA）
- 训练参数：`--ablation mspa:True`
- GUI：在 `Models` 页选择 “谱域 baseline ckpt” 并加载

### 4.3 常见 checkpoint 文件名（示例）
- `outputs/ckpt_best_seed42_fold0.pth`
- `outputs/ckpt_best_seed42_loso_SA01.pth`

只要是 PyTorch 保存的权重文件（`.pth/.pt`）且结构匹配，均可加载。

### 4.4 结构不匹配会怎样
若 checkpoint 与当前选择的结构（是否 MSPA、输入通道数、类别数）不匹配，加载会失败并提示类似：
- “Checkpoint parameters do not match the model architecture …”

这通常意味着你：
- 选错了“主线/谱域”类型；或
- 该 checkpoint 不是本仓库/该版本训练出来的；或
- 输入通道/类别配置不同。

---

## 5. 数据输入：SisFall `.txt`（推荐用于演示指标）

### 5.1 目录结构要求

你选择的根目录需要包含：
- `ADL/`（非跌倒样本）
- `FALL/`（跌倒样本）

例如：
```
SisFall/
  ADL/*.txt
  FALL/*.txt
```

软件也会尝试自动识别 `root/SisFall/ADL` 与 `root/SisFall/FALL`。

### 5.2 预处理流程（与训练对齐）
对每个文件，程序会：
- 解析每行数值（允许尾部 `;`，分隔符支持 `,` 或 `;`）
- 选取通道：默认 `accel3`（前三通道）
- **按通道做 z-score 标准化**
- 以 `window_size=512`、`stride=256` 滑窗切片

### 5.3 Subject 过滤（便于单个被试对比）
在 `Data` 页加载 SisFall 后，会自动解析文件名中的 subject（如 `SA01`），并在下拉框提供：
- `ALL`：全部数据
- `SAxx`：按��试过滤

过滤后输出会显示：
- windows 数量
- FALL/ADL 数量

---

## 6. 数据输入：自定义 `.csv`（推荐用于无标签实时数据）

### 6.1 CSV 格式要求
CSV 至少需要 3 列数值（建议带表头）：
- `accel_x, accel_y, accel_z`

项目内置一个可直接测试的样例：
- `phyclCode/sample_input.csv`

### 6.2 采样率与重采样
在 `Data` 页选择 CSV 后，需要填写 CSV 的真实采样率（Hz）：
- 程序会线性重采样到 50Hz（默认训练采样率）
- 然后 z-score 标准化与切窗

### 6.3 CSV 无标签时输出什么
CSV 默认 **无标签**，因此不会输出 Accuracy/Macro‑F1，仅输出：
- 总窗口数
- 超过阈值的窗口数（“高于阈值的窗口”）

---

## 7. 设备选择（CPU / GPU）

在 `Models` 页可以选择设备：
- 默认 `cpu`
- 若运行环境支持 CUDA，会出现 `cuda:0 / cuda:1 ...`

建议：
- 做论文叙述里的 CPU 延迟展示：选择 `cpu`
- 做性能演示：可选择 `cuda:0`（前提是你的打包/环境支持 CUDA）

---

## 8. 推理与对比（Run/Compare 页）

### 8.1 阈值（fall prob threshold）
- `threshold` 表示将窗口判定为“fall”的概率阈值（默认 0.5）
- SisFall 有标签时可调整阈值观察 TPR/FPR trade‑off

### 8.2 单模型运行
按钮：
- “运行主线模型”
- “运行谱域模型”

输出内容：
- windows 数量与阈值
- （SisFall）Accuracy / Macro‑F1 / TP FP FN TN / 两个 operating point
- （CSV）高于阈值的窗口数量
- 轻量延迟：p50/p95（单线程）

### 8.3 双模型对比（顺序执行）
按钮：
- “对比两模型”

行为说明：
- 两个模型会 **按顺序** 运行并依次输出（便于对比）
- 推理会在后台线程执行，GUI 不会卡死
- 若当前已有推理任务在运行，软件会提示“已有推理任务在运行，请稍候…”

---

## 9. 性能与体验建议（更快、更稳）

- 初次启动更快：软件会尽量延迟加载 torch/模型；但第一次加载 ckpt 或首次推理仍会有“首次初始化”的额外耗时。
- 数据量很大时：建议先用子集目录或先选单一 subject（例如 `SA01`）验证流程。
- 推理更快：可以适当增大 `batch_size`（但会增加显存/内存占用）。
- 延迟测量更稳定：建议关闭其他占用 CPU 的程序；CPU 测量为轻量快速测量，非严格 benchmark。

---

## 10. 常见问题（FAQ / 排错）

### 10.1 双击没反应 / 闪退
建议在命令行中启动查看报错：
1) 打开 `dist/PhyCLNetDemo/` 目录
2) 在该目录执行：`PhyCLNetDemo.exe`

### 10.2 “为什么 EXE 很大”
正常：PyTorch + Qt GUI 推理版通常是几百 MB 甚至更大（onedir）。

### 10.3 “加载 checkpoint 失败”
常见原因：
- ckpt 文件损坏或路径包含特殊字符导致读取失败
- ckpt 与当前选择的“主线/谱域”结构不匹配
- ckpt 不是 PhyCL‑Net 的权重文件

### 10.4 “没有输出 Accuracy/Macro‑F1”
你加载的是 CSV（默认无标签），这是正常行为。需要指标请加载 SisFall（ADL/FALL）。

### 10.5 推理很慢
建议：
- 优先选 CPU（更少环境依赖）
- 先用较小数据集/单 subject
- 检查是否选了不兼容的 CUDA 设备（可能导致隐式回退或初始化卡顿）

---

## 11. CLI（可选，仅用于快速验证）

如果你用源码方式运行（非 exe），可以用 CLI 做快速验证：
```bash
python -m app --sisfall-root ./data/SisFall --ckpt-time ./outputs/ckpt_best_seed42_fold0.pth
```

CLI 输出会打印：
- basic metrics（Accuracy/Macro‑F1 等）
- operating points

---

## 12. 开发者：重新打包 EXE（可选）

在 `phyclCode/` 下（并确保环境里有 pyinstaller）：
```powershell
pwsh scripts/build_exe.ps1
```

高级用法：直接用 spec 打包（便于调参瘦身）：
```powershell
pyinstaller --noconfirm PhyCLNetDemo.spec
```

输出目录：
- `phyclCode/dist/PhyCLNetDemo/`

