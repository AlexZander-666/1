# PhyCLNetDemo.exe 使用说明（GUI 推理/对比版）

本软件用于 **PhyCL‑Net 论文主线（去 MSPA） vs 谱域 baseline（保留 MSPA）** 的推理演示与 trade‑off 对比，输入支持 **SisFall 原始 `.txt`** 与 **自定义 `.csv`**。

更完整的使用手册（包含 GUI 分页说明、输入格式细节、性能建议与排错）见：
- `phyclCode/USER_GUIDE.md`

---

## 1. 你会得到哪些文件

打包产物位于：
- `dist/PhyCLNetDemo/PhyCLNetDemo.exe`

注意：这是 `--onedir` 打包方式，运行时需要整个目录（不要只拷贝 exe）。

---

## 2. 运行环境要求

- Windows 10/11 64-bit
- 建议 CPU 推理（更接近 copy666.md 的“single-thread CPU latency”叙述）
- 如你在 GUI 中选择 CUDA 设备，则需要本机 PyTorch/CUDA 运行时兼容（仅当你打包时把 CUDA 版 torch 一起打入才一定可用；默认建议用 CPU）

---

## 3. 启动方式

双击运行：
- `dist/PhyCLNetDemo/PhyCLNetDemo.exe`

如果打不开：
- 尝试以命令行启动查看报错：在 `dist/PhyCLNetDemo/` 目录下执行 `PhyCLNetDemo.exe`

---

## 4. 模型（checkpoint）准备

你需要两份 checkpoint（`.pth/.pt`），分别对应两种结构：

1) **主线模型（去 MSPA）**
- 训练参数：`--ablation mspa:False`

2) **谱域 baseline（保留 MSPA）**
- 训练参数：`--ablation mspa:True`

GUI 的 `Models` 页分别加载它们即可。

提示：checkpoint 通常长这样（仅示例）：
- `outputs/ckpt_best_seed42_fold0.pth`
- `outputs/ckpt_best_seed42_fold0.pth`（另一套 mspa 配置训练出来的）

---

## 5. 数据输入：SisFall `.txt`

### 5.1 目录结构

选择的根目录需要包含：
- `ADL/`（非跌倒样本）
- `FALL/`（跌倒样本）

例如：
```
SisFall/
  ADL/*.txt
  FALL/*.txt
```

### 5.2 预处理（与训练对齐）

程序会对每个文件：
- 解析每行数值（允许以 `;` 结尾）
- 选取通道（默认 `accel3` → 取前三通道）
- **按通道做 z-score 标准化**（与训练脚本一致）
- 按 `window_size=512`、`stride=256` 切窗

选择 SisFall 后会带标签，因此可以计算：
- Accuracy / Macro‑F1
- `FPR@TPR=95%`
- `TPR@FPR=1%`

---

## 6. 数据输入：自定义 `.csv`

### 6.1 CSV 格式

最简单的 CSV：三列数值，加表头：
- `accel_x, accel_y, accel_z`

项目已提供一个可直接测试的样例：
- `sample_input.csv`

### 6.2 采样率

在 `Data` 页选择 CSV 后：
- 填入 CSV 的采样率（Hz）
- 程序会重采样到 50Hz（线性插值），再标准化与切窗

注意：CSV 默认 **无标签**，因此不会输出 Accuracy/Macro‑F1，只会输出：
- 高于阈值的窗口数量

---

## 7. 推理与对比（Run/Compare 页）

### 7.1 阈值（fall prob threshold）

- `threshold` 表示判定“fall”概率阈值（默认 0.5）
- SisFall 有标签时可以通过阈值调整观察 TPR/FPR 的变化趋势

### 7.2 单模型运行

点击：
- “运行主线模型” 或 “运行谱域模型”

会输出：
- 预测窗口数量
- （SisFall）Accuracy/Macro‑F1、混淆统计 TP/FP/FN/TN
- （SisFall）`FPR@TPR=95%` 与 `TPR@FPR=1%`
- 轻量延迟测试：p50/p95（单线程）

### 7.3 双模型对比

点击：
- “对比两模型”

会先后输出主线与谱域 baseline 的上述指标，用于展示 trade‑off。

---

## 8. 常见问题

### 8.1 “attempted relative import with no known parent package”

已修复：新版本入口改为绝对导入，并在无包上下文时自动加入仓库根路径。

### 8.2 为什么 EXE 很大

正常：PyTorch + Qt 的 GUI 推理版通常是几百 MB 级别（onedir）。

### 8.3 运行很慢

建议：
- 优先选 CPU（并保持单线程测延迟）
- 尽量不要一次性加载过大的数据目录（先用子集验证流程）

---

## 9. 重新打包（开发者）

在 `PhyCL` conda 环境中：
```powershell
conda run -n phyCL pyinstaller --noconfirm --noconsole --onedir --name PhyCLNetDemo --collect-all torch --collect-all PySide6 --add-data "template;template" app/main.py
```

输出在：
- `dist/PhyCLNetDemo/`
