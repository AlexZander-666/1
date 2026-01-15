# PhyCL-FallDetect 跌倒检测系统 V1.0 用户手册

---

## 1. 整体界面介绍

软件启动后显示主界面，包含三个功能标签页：

1. **Data** — 数据加载标签页
2. **Models** — 模型加载标签页
3. **Run/Compare** — 推理与对比标签页

界面主要区域与编号含义如下：

- ① SisFall 数据集路径输入框
- ② "选择 SisFall 根目录" 按钮
- ③ "加载 SisFall" 按钮
- ④ CSV 文件路径输入框
- ⑤ "选择 CSV" 按钮
- ⑥ 采样率设置（Hz）
- ⑦ "加载 CSV" 按钮
- ⑧ Subject 过滤下拉框
- ⑨ 状态/日志显示区域

---

## 2. 系统环境要求

### 2.1 运行环境

- Windows 10/11 64-bit
- 建议使用 CPU 推理（更接近论文中的单线程延迟测试）
- 如需 GPU 推理，需本机 PyTorch/CUDA 运行时兼容

### 2.2 文件说明

打包目录结构如下：

```
PhyCL-FallDetect/
  ├── PhyCL-FallDetect/          ← 主程序目录
  │   ├── PhyCL-FallDetect.exe   ← 主程序
  │   └── _internal/             ← 运行时依赖
  ├── SisFall/                   ← SisFall 数据集
  │   ├── ADL/                   ← 非跌倒样本
  │   └── FALL/                  ← 跌倒样本
  ├── source_code/               ← 源代码
  │   ├── app/                   ← GUI 应用代码
  │   ├── models/                ← 模型定义
  │   ├── losses/                ← 损失函数
  │   ├── PhyCL_Net_experiments.py
  │   └── requirements.txt
  ├── sample_input.csv           ← 样例 CSV 文件
  └── 用户手册.md                ← 本手册
```

注意：运行时需要整个 `PhyCL-FallDetect/` 目录，不要只拷贝 exe

---

## 3. 功能：启动软件

操作步骤：

1. 进入 `PhyCL-FallDetect/` 目录。
2. 双击 `PhyCL-FallDetect.exe` 启动软件。
3. 软件启动后显示主界面。

如果打不开：

- 在 `PhyCL-FallDetect/` 目录下打开命令行，执行 `PhyCL-FallDetect.exe` 查看报错信息。

---

## 4. 功能：加载 SisFall 数据集（Data 标签页）

用途：加载 SisFall 跌倒检测数据集进行推理测试。

### 4.1 目录结构要求

选择的根目录需要包含：

```
SisFall/
  ADL/    ← 非跌倒样本（日常活动）
  FALL/   ← 跌倒样本
```

### 4.2 操作步骤

1. 点击 **Data** 标签页。
2. 点击"选择 SisFall 根目录 (含 ADL/FALL)"按钮。
3. 在弹出的文件夹选择对话框中，选择 SisFall 数据集根目录。
4. 点击"加载 SisFall"按钮。
5. 加载成功后，状态区域显示加载的 subjects 数量和窗口数量。

### 4.3 Subject 过滤

- 加载完成后，可在 Subject 过滤下拉框中选择特定受试者。
- 选择 "ALL" 表示使用全部数据。

---

## 5. 功能：加载自定义 CSV 数据（Data 标签页）

用途：加载自定义的加速度数据进行推理测试。

### 5.1 CSV 格式要求

最简单的 CSV 格式：三列数值，加表头：

```csv
accel_x,accel_y,accel_z
0.12,0.34,9.81
...
```

项目已提供可直接测试的样例文件：`sample_input.csv`

### 5.2 操作步骤

1. 点击 **Data** 标签页。
2. 点击"选择 CSV"按钮。
3. 在弹出的文件选择对话框中，选择 CSV 文件。
4. 在"采样率"输入框中填入 CSV 的采样率（Hz）。
5. 点击"加载 CSV"按钮。
6. 程序会自动重采样到 50Hz，再进行标准化与切窗。

注意：CSV 数据默认无标签，因此不会输出 Accuracy/Macro-F1，只会输出高于阈值的窗口数量。

---

## 6. 功能：加载模型（Models 标签页）

用途：加载训练好的 checkpoint 文件用于推理。

软件支持两种模型结构的对比：

| 模型类型 | 说明 | 训练参数 |
|---------|------|---------|
| 主线模型 | 去 MSPA（论文主线） | `--ablation mspa:False` |
| 谱域 baseline | 保留 MSPA | `--ablation mspa:True` |

### 6.1 操作步骤

1. 点击 **Models** 标签页。
2. 点击"选择 主线(去 MSPA) ckpt"按钮，选择主线模型的 checkpoint 文件（`.pth` 或 `.pt`）。
3. 点击"加载主线模型"按钮。
4. 点击"选择 谱域 baseline ckpt"按钮，选择谱域模型的 checkpoint 文件。
5. 点击"加载谱域模型"按钮。
6. 在"设备"下拉框中选择推理设备（默认 CPU，如有 GPU 可选择 CUDA）。

---

## 7. 功能：运行推理与对比（Run/Compare 标签页）

用途：对加载的数据进行跌倒检测推理，并对比两种模型的性能。

### 7.1 阈值设置

- "阈值 (fall prob)"：判定跌倒的概率阈值，默认 0.5。
- 可通过调整阈值观察 TPR/FPR 的变化趋势。

### 7.2 单模型运行

操作步骤：

1. 点击 **Run/Compare** 标签页。
2. 设置阈值（可选）。
3. 点击"运行主线模型"或"运行谱域模型"按钮。

输出结果：

- 预测窗口数量
- （SisFall）Accuracy / Macro-F1
- （SisFall）混淆矩阵统计：TP / FP / FN / TN
- （SisFall）`FPR@TPR=95%` 与 `TPR@FPR=1%`
- 延迟测试：p50 / p95（单线程 CPU）

### 7.3 双模型对比

操作步骤：

1. 确保已加载两份模型（主线和谱域）。
2. 点击"对比两模型"按钮。
3. 系统会依次运行两个模型，输出各自的指标用于对比。

---

## 8. 功能：推理结果示例

### 8.1 SisFall 数据集推理结果示例

```
[主线 (无 MSPA)] windows=1024, threshold=0.50
Accuracy=96.48%, Macro-F1=95.32%
TP=512 FP=8 FN=20 TN=484
FPR@TPR=95%: 2.15% | TPR@FPR=1%: 89.45%
Latency p50=1.23 ms, p95=1.56 ms (单线程)
```

### 8.2 CSV 数据推理结果示例

```
[主线 (无 MSPA)] windows=128, threshold=0.50
高于阈值的窗口: 3
Latency p50=1.18 ms, p95=1.42 ms (单线程)
```

---

## 9. 数据预处理说明

### 9.1 SisFall 数据预处理

程序会对每个文件：

- 解析每行数值（允许以 `;` 结尾）
- 选取通道（默认 `accel3` → 取前三通道）
- 按通道做 z-score 标准化（与训练脚本一致）
- 按 `window_size=512`、`stride=256` 切窗

### 9.2 CSV 数据预处理

- 若采样率与 50Hz 不同，会进行线性重采样
- 重采样后进行标准化与切窗

---

## 10. 常见问题

### 10.1 为什么 EXE 文件很大？

正常现象：PyTorch + Qt 的 GUI 推理版通常是几百 MB 级别（onedir 打包方式）。

### 10.2 运行很慢怎么办？

建议：

- 优先选择 CPU 设备（并保持单线程测延迟）
- 尽量不要一次性加载过大的数据目录（先用子集验证流程）

### 10.3 路径包含中文导致加载失败

建议：

- 将数据文件和软件放置在不含中文的路径下
- 例如：`D:/PhyCLNetDemo/` 而非 `D:/跌倒检测/`

### 10.4 模型加载失败

可能原因：

- checkpoint 文件与模型结构不匹配
- 确保主线模型使用 `mspa:False` 训练，谱域模型使用 `mspa:True` 训练

---

## 11. 注意事项

- 推理仅支持二分类（fall vs non-fall），多分类未处理
- TFCL 投影头在推理中不使用；checkpoint 若带有这些权重会被忽略
- 延迟测试是轻量级 CPU 测量，如需严格对标论文可单独运行更长基准
- 避免将数据集或 checkpoint 提交到版本控制，保持大文件本地存储

---
