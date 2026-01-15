"""PySide6 GUI + minimal CLI for PhyCL-Net demo (time-domain vs spectral)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtWidgets

# Support running both as a module (`python -m app`) and as a script entry (`app/main.py` from PyInstaller).
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    ROOT = _Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_CONFIG, DemoConfig
from app.data import load_csv_windows, load_sisfall_windows
from app.metrics import basic_metrics, operating_points

if TYPE_CHECKING:  # Heavy imports deferred to runtime to speed up GUI startup.
    import torch


@dataclass
class DemoState:
    config: DemoConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    sisfall_items: Optional[List[Tuple[np.ndarray, int, str]]] = None
    windows: List[np.ndarray] = field(default_factory=list)
    labels: Optional[List[int]] = None
    subjects: Optional[List[str]] = None
    sample_rate: float = 50.0
    model_time: Optional["torch.nn.Module"] = None
    model_spectral: Optional["torch.nn.Module"] = None
    ckpt_time: Optional[Path] = None
    ckpt_spectral: Optional[Path] = None


def _available_devices() -> List[str]:
    """
    Lazily query torch devices so the GUI can start without importing torch.
    """
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    except Exception:
        pass
    return devices


def _model_lib():
    # Deferred import to keep startup fast.
    from app import model as model_lib

    return model_lib


class DataTab(QtWidgets.QWidget):
    def __init__(self, state: DemoState, on_loaded):
        super().__init__()
        self.state = state
        self.on_loaded = on_loaded
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout()

        # SisFall loader
        sisfall_row = QtWidgets.QHBoxLayout()
        self.sisfall_path_edit = QtWidgets.QLineEdit()
        btn_sisfall = QtWidgets.QPushButton("选择 SisFall 根目录 (含 ADL/FALL)")
        btn_sisfall.clicked.connect(self._choose_sisfall)
        btn_load_sisfall = QtWidgets.QPushButton("加载 SisFall")
        btn_load_sisfall.clicked.connect(self._load_sisfall)
        sisfall_row.addWidget(self.sisfall_path_edit)
        sisfall_row.addWidget(btn_sisfall)
        sisfall_row.addWidget(btn_load_sisfall)
        layout.addLayout(sisfall_row)

        # CSV loader
        csv_row = QtWidgets.QHBoxLayout()
        self.csv_path_edit = QtWidgets.QLineEdit()
        btn_csv = QtWidgets.QPushButton("选择 CSV")
        btn_csv.clicked.connect(self._choose_csv)
        self.csv_sr = QtWidgets.QDoubleSpinBox()
        self.csv_sr.setRange(1.0, 500.0)
        self.csv_sr.setValue(self.state.sample_rate)
        self.csv_sr.setSuffix(" Hz")
        btn_load_csv = QtWidgets.QPushButton("加载 CSV")
        btn_load_csv.clicked.connect(self._load_csv)
        csv_row.addWidget(self.csv_path_edit)
        csv_row.addWidget(btn_csv)
        csv_row.addWidget(QtWidgets.QLabel("采样率"))
        csv_row.addWidget(self.csv_sr)
        csv_row.addWidget(btn_load_csv)
        layout.addLayout(csv_row)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("SisFall Subject 过滤"))
        self.subject_combo = QtWidgets.QComboBox()
        self.subject_combo.addItems(["ALL"])
        self.subject_combo.currentTextChanged.connect(self._apply_subject_filter)
        filter_row.addWidget(self.subject_combo)
        filter_row.addStretch(1)
        layout.addLayout(filter_row)

        self.status = QtWidgets.QPlainTextEdit()
        self.status.setReadOnly(True)
        layout.addWidget(self.status)
        self.setLayout(layout)

    def _choose_sisfall(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 SisFall 根目录")
        if directory:
            self.sisfall_path_edit.setText(directory)

    def _choose_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 CSV 文件", filter="CSV Files (*.csv);;All Files (*)")
        if path:
            self.csv_path_edit.setText(path)

    def _load_sisfall(self):
        path = self.sisfall_path_edit.text().strip()
        if not path:
            self._log("请先选择 SisFall 根目录")
            return
        try:
            items = load_sisfall_windows(Path(path), window_size=self.state.config.window_size, stride=self.state.config.stride, channels_used=self.state.config.channels_used)
        except Exception as e:
            self._log(f"SisFall 加载失败: {e}")
            return
        self.state.sisfall_items = items
        subjects = sorted({s for _, _, s in items})
        self.subject_combo.blockSignals(True)
        self.subject_combo.clear()
        self.subject_combo.addItems(["ALL"] + subjects)
        self.subject_combo.blockSignals(False)
        self._apply_subject_filter("ALL")
        self.state.sample_rate = self.state.config.sample_rate_hz
        self._log(f"SisFall 加载完成: subjects={len(subjects)}, windows={len(items)}")
        self.on_loaded()

    def _apply_subject_filter(self, subject: str):
        if not self.state.sisfall_items:
            return
        if subject == "ALL":
            filtered = self.state.sisfall_items
        else:
            filtered = [it for it in self.state.sisfall_items if it[2] == subject]
        self.state.windows = [w for w, _, _ in filtered]
        self.state.labels = [y for _, y, _ in filtered]
        self.state.subjects = [s for _, _, s in filtered]
        pos = int(sum(self.state.labels))
        total = len(self.state.labels)
        self._log(f"已应用过滤: {subject} | windows={total} | FALL={pos} ADL={total-pos}")

    def _load_csv(self):
        path = self.csv_path_edit.text().strip()
        if not path:
            self._log("请先选择 CSV 文件")
            return
        try:
            windows = load_csv_windows(Path(path), sample_rate_hz=float(self.csv_sr.value()), target_rate_hz=self.state.config.sample_rate_hz, window_size=self.state.config.window_size, stride=self.state.config.stride)
        except Exception as e:
            self._log(f"CSV 加载失败: {e}")
            return
        self.state.windows = list(windows)
        self.state.labels = None  # 无标签
        self.state.subjects = None
        self.state.sisfall_items = None
        self.state.sample_rate = self.state.config.sample_rate_hz
        self._log(f"CSV 加载完成: {len(self.state.windows)} 窗口 (未提供标签)")
        self.on_loaded()

    def _log(self, msg: str):
        self.status.appendPlainText(msg)


class ModelTab(QtWidgets.QWidget):
    def __init__(self, state: DemoState):
        super().__init__()
        self.state = state
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout()

        # MSPA off
        row1 = QtWidgets.QHBoxLayout()
        self.ckpt_time_edit = QtWidgets.QLineEdit()
        btn_time = QtWidgets.QPushButton("选择 主线(去 MSPA) ckpt")
        btn_time.clicked.connect(self._choose_time)
        btn_load_time = QtWidgets.QPushButton("加载主线模型")
        btn_load_time.clicked.connect(self._load_time)
        row1.addWidget(self.ckpt_time_edit)
        row1.addWidget(btn_time)
        row1.addWidget(btn_load_time)
        layout.addLayout(row1)

        # MSPA on
        row2 = QtWidgets.QHBoxLayout()
        self.ckpt_spec_edit = QtWidgets.QLineEdit()
        btn_spec = QtWidgets.QPushButton("选择 谱域 baseline ckpt")
        btn_spec.clicked.connect(self._choose_spec)
        btn_load_spec = QtWidgets.QPushButton("加载谱域模型")
        btn_load_spec.clicked.connect(self._load_spec)
        row2.addWidget(self.ckpt_spec_edit)
        row2.addWidget(btn_spec)
        row2.addWidget(btn_load_spec)
        layout.addLayout(row2)

        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(_available_devices())
        device_row = QtWidgets.QHBoxLayout()
        device_row.addWidget(QtWidgets.QLabel("设备"))
        device_row.addWidget(self.device_combo)
        layout.addLayout(device_row)

        self.status = QtWidgets.QPlainTextEdit()
        self.status.setReadOnly(True)
        layout.addWidget(self.status)
        self.setLayout(layout)

    def _choose_time(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择主线 ckpt", filter="PyTorch (*.pth *.pt);;All Files (*)")
        if path:
            self.ckpt_time_edit.setText(path)

    def _choose_spec(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择谱域 ckpt", filter="PyTorch (*.pth *.pt);;All Files (*)")
        if path:
            self.ckpt_spec_edit.setText(path)

    def _load_time(self):
        path = self.ckpt_time_edit.text().strip()
        if not path:
            self._log("请先选择 ckpt")
            return
        device = self.device_combo.currentText()
        try:
            model_lib = _model_lib()
            self.state.model_time = model_lib.load_phycl_net(
                Path(path),
                mspa_enabled=False,
                device=device,
                sample_rate=self.state.config.sample_rate_hz,
            )
            self.state.ckpt_time = Path(path)
            self._log(f"主线模型已加载: {path} @ {device}")
        except Exception as e:
            self._log(f"加载失败: {e}")

    def _load_spec(self):
        path = self.ckpt_spec_edit.text().strip()
        if not path:
            self._log("请先选择 ckpt")
            return
        device = self.device_combo.currentText()
        try:
            model_lib = _model_lib()
            self.state.model_spectral = model_lib.load_phycl_net(
                Path(path),
                mspa_enabled=True,
                device=device,
                sample_rate=self.state.config.sample_rate_hz,
            )
            self.state.ckpt_spectral = Path(path)
            self._log(f"谱域模型已加载: {path} @ {device}")
        except Exception as e:
            self._log(f"加载失败: {e}")

    def _log(self, msg: str):
        self.status.appendPlainText(msg)


class RunTab(QtWidgets.QWidget):
    def __init__(self, state: DemoState):
        super().__init__()
        self.state = state
        self.worker_thread: Optional[QtCore.QThread] = None
        self.worker: Optional["InferenceWorker"] = None
        self.compare_queue: List[Tuple[object, str]] = []
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout()
        self.thresh = QtWidgets.QDoubleSpinBox()
        self.thresh.setRange(0.0, 1.0)
        self.thresh.setSingleStep(0.01)
        self.thresh.setValue(self.state.config.threshold)
        t_row = QtWidgets.QHBoxLayout()
        t_row.addWidget(QtWidgets.QLabel("阈值 (fall prob)"))
        t_row.addWidget(self.thresh)
        layout.addLayout(t_row)

        btn_row = QtWidgets.QHBoxLayout()
        btn_run_time = QtWidgets.QPushButton("运行主线模型")
        btn_run_time.clicked.connect(self._run_time)
        btn_run_spec = QtWidgets.QPushButton("运行谱域模型")
        btn_run_spec.clicked.connect(self._run_spec)
        btn_compare = QtWidgets.QPushButton("对比两模型")
        btn_compare.clicked.connect(self._compare)
        btn_row.addWidget(btn_run_time)
        btn_row.addWidget(btn_run_spec)
        btn_row.addWidget(btn_compare)
        layout.addLayout(btn_row)

        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        layout.addWidget(self.out)
        self.setLayout(layout)

    def _need_data(self) -> bool:
        if not self.state.windows:
            self._log("请先加载数据 (SisFall 或 CSV)")
            return True
        return False

    def _run_time(self):
        if self._need_data():
            return
        if self.state.model_time is None:
            self._log("请先加载主线模型")
            return
        self.compare_queue.clear()
        self._start_worker(self.state.model_time, "主线 (无 MSPA)")

    def _run_spec(self):
        if self._need_data():
            return
        if self.state.model_spectral is None:
            self._log("请先加载谱域模型")
            return
        self.compare_queue.clear()
        self._start_worker(self.state.model_spectral, "谱域 baseline")

    def _compare(self):
        if self._need_data():
            return
        if self.state.model_time is None or self.state.model_spectral is None:
            self._log("请先加载两份模型")
            return
        self.compare_queue = [
            (self.state.model_time, "主线 (无 MSPA)"),
            (self.state.model_spectral, "谱域 baseline"),
        ]
        self._start_next_in_queue()

    def _start_next_in_queue(self):
        if not self.compare_queue:
            return
        model, name = self.compare_queue.pop(0)
        self._start_worker(model, name)

    def _start_worker(self, model, name: str):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self._log("已有推理任务在运行，请稍候...")
            return

        thresh = float(self.thresh.value())
        device = str(next(model.parameters()).device)
        windows = list(self.state.windows)
        labels = list(self.state.labels) if self.state.labels is not None else None

        self.worker_thread = QtCore.QThread()
        self.worker = InferenceWorker(
            model=model,
            windows=windows,
            labels=labels,
            name=name,
            threshold=thresh,
            device=device,
            config=self.state.config,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.message.connect(self._log)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker)
        self.worker_thread.start()
        self._log(f"[{name}] 已在后台启动推理 ({device})...")

    def _log(self, msg: str):
        self.out.appendPlainText(msg)

    def _clear_worker(self):
        self.worker_thread = None
        self.worker = None
        self._start_next_in_queue()


class InferenceWorker(QtCore.QObject):
    finished = QtCore.Signal()
    message = QtCore.Signal(str)

    def __init__(
        self,
        *,
        model,
        windows: Sequence[np.ndarray],
        labels: Optional[Sequence[int]],
        name: str,
        threshold: float,
        device: str,
        config: DemoConfig,
    ):
        super().__init__()
        self.model = model
        self.windows = windows
        self.labels = labels
        self.name = name
        self.threshold = threshold
        self.device = device
        self.config = config

    @QtCore.Slot()
    def run(self):
        try:
            model_lib = _model_lib()
            probs = model_lib.predict_probabilities(
                self.model,
                self.windows,
                batch_size=self.config.batch_size,
                device=self.device,
            )
            msg = [f"[{self.name}] windows={len(probs)}, threshold={self.threshold:.2f}"]
            if self.labels is not None:
                metrics = basic_metrics(self.labels, probs, threshold=self.threshold)
                op = operating_points(self.labels, probs)
                msg.append(f"Accuracy={metrics['accuracy']*100:.2f}%, Macro-F1={metrics['macro_f1']*100:.2f}%")
                msg.append(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
                msg.append(f"FPR@TPR=95%: {op['fpr_at_tpr95']*100:.2f}% | TPR@FPR=1%: {op['tpr_at_fpr1']*100:.2f}%")
            else:
                high = (probs >= self.threshold).sum()
                msg.append(f"高于阈值的窗口: {high}")

            try:
                p50, p95 = model_lib.measure_latency_ms(
                    self.model,
                    window_size=self.config.window_size,
                    runs=50,
                    warmup=10,
                    device=self.device,
                )
                msg.append(f"Latency p50={p50:.2f} ms, p95={p95:.2f} ms (单线程)")
            except Exception as e:
                msg.append(f"Latency 测试失败: {e}")

            self.message.emit("\n".join(msg))
        except Exception as e:
            self.message.emit(f"[{self.name}] 推理失败: {e}")
        finally:
            self.finished.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, state: DemoState):
        super().__init__()
        self.state = state
        self.setWindowTitle("PhyCL-Net Demo (时间域 vs 谱域)")
        tabs = QtWidgets.QTabWidget()
        self.data_tab = DataTab(self.state, self._on_data_loaded)
        self.model_tab = ModelTab(self.state)
        self.run_tab = RunTab(self.state)
        tabs.addTab(self.data_tab, "Data")
        tabs.addTab(self.model_tab, "Models")
        tabs.addTab(self.run_tab, "Run/Compare")
        self.setCentralWidget(tabs)

    def _on_data_loaded(self):
        # Placeholder hook; could refresh status across tabs.
        pass


def run_gui():
    app = QtWidgets.QApplication(sys.argv)
    state = DemoState()
    window = MainWindow(state)
    window.resize(900, 600)
    window.show()
    sys.exit(app.exec())


def run_cli(args: argparse.Namespace):
    state = DemoState()
    state.config = DemoConfig()
    windows = load_sisfall_windows(
        Path(args.sisfall_root),
        window_size=state.config.window_size,
        stride=state.config.stride,
        channels_used=state.config.channels_used,
    )
    state.windows = [w for w, _, _ in windows]
    state.labels = [y for _, y, _ in windows]
    model_lib = _model_lib()
    model_time = model_lib.load_phycl_net(
        Path(args.ckpt_time),
        mspa_enabled=False,
        device="cpu",
        sample_rate=state.config.sample_rate_hz,
    )
    probs = model_lib.predict_probabilities(
        model_time,
        state.windows,
        batch_size=state.config.batch_size,
        device="cpu",
    )
    m = basic_metrics(state.labels, probs, threshold=0.5)
    op = operating_points(state.labels, probs)
    print("主线模型 CLI 推理完成")
    print(m)
    print(op)


def main():
    parser = argparse.ArgumentParser(description="PhyCL-Net GUI/CLI demo")
    parser.add_argument("--gui", action="store_true", help="启动 GUI（默认）")
    parser.add_argument("--sisfall-root", type=str, help="CLI: SisFall 根目录")
    parser.add_argument("--ckpt-time", type=str, help="CLI: 主线 ckpt 路径")
    args = parser.parse_args()

    if args.gui or not args.sisfall_root:
        run_gui()
    else:
        if not args.ckpt_time:
            parser.error("--ckpt-time 必填 (CLI 模式)")
        run_cli(args)


if __name__ == "__main__":
    main()
