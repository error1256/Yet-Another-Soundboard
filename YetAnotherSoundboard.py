'''
TODO: add auto saving for auto_monitor
Its 2:40 AM and I dont want to sleep
'''
import os
import sys
import json
import time
import threading
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
import sounddevice as sd
import soundfile as sf
import re
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QPixmap, QFont, QCloseEvent, QIcon
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineProfile, QWebEnginePage
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QPushButton,
    QCheckBox,
    QSlider,
    QListWidget,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QLineEdit,
    QTabWidget,
    QListWidgetItem,
    QColorDialog,
    QGridLayout,
    QInputDialog,
)
APPDATA_DIR = os.path.join(os.getenv("APPDATA"), "errorC003C004", "Soundboard")
SETTINGS_PATH = os.path.join(APPDATA_DIR, "settings.json")
os.makedirs(APPDATA_DIR, exist_ok=True)

if not os.path.exists(SETTINGS_PATH):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump({"Current Preset": "Default"}, f, indent=2)

    PROFILE_NAME = "Default"
else:
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    PROFILE_NAME = data.get("Current Preset", "Default")

PROFILE_DIR = os.path.join(APPDATA_DIR, PROFILE_NAME)
MEDIA_DIR = os.path.join(PROFILE_DIR, "Media")
PRESET_PATH = os.path.join(PROFILE_DIR, "preset.json")
DEFAULT_IMG_PATH = os.path.join(APPDATA_DIR, "default.png")

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio

def create_auto_preset():
    default_folder = os.path.join(APPDATA_DIR, "Default")
    preset_path = os.path.join(default_folder, "preset.json")

    if not os.path.exists(preset_path):
        os.makedirs(default_folder, exist_ok=True)
        with open(preset_path, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)

def _load_auto_preset(w) -> None:
    if not os.path.exists(PRESET_PATH):
        create_auto_preset()
        return
    try:
        with open(PRESET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        w._apply_preset_to_ui(data)
    except Exception as e:
        print("Auto-load failed:", e)

def db_to_lin(db: float) -> float:
    return float(np.power(10.0, db / 20.0))

def soft_clip(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    return np.tanh(x * drive)

def one_pole_lowpass(x: np.ndarray, alpha: float, state: float) -> Tuple[np.ndarray, float]:
    y = np.empty_like(x)
    y_prev = state
    a = float(alpha)
    b = 1.0 - a
    for i in range(len(x)):
        y_prev = b * x[i] + a * y_prev
        y[i] = y_prev
    return y, float(y_prev)

def one_pole_highpass(
    x: np.ndarray, alpha: float, state_x: float, state_y: float
) -> Tuple[np.ndarray, float, float]:
    y = np.empty_like(x)
    x_prev = state_x
    y_prev = state_y
    a = float(alpha)
    for i in range(len(x)):
        y_prev = a * (y_prev + x[i] - x_prev)
        x_prev = x[i]
        y[i] = y_prev
    return y, float(x_prev), float(y_prev)

def resample_mono_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if audio is None or len(audio) == 0:
        return np.zeros((0,), dtype=np.float32)

    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)

    ratio = float(dst_sr) / float(src_sr)
    n_out = max(1, int(round(len(audio) * ratio)))

    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float32)

    out = np.interp(x_new, x_old, audio.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    return out

class AudioRingBuffer:
    """
    Single-producer/single-consumer ring buffer for float32 mono audio.
    Drops oldest audio on overflow, outputs zeros on underflow.
    """

    def __init__(self, capacity_frames: int) -> None:
        self._lock = threading.Lock()
        self._buf = np.zeros((max(1, int(capacity_frames)),), dtype=np.float32)
        self._cap = int(self._buf.size)
        self._r = 0
        self._w = 0
        self._size = 0

    def clear(self) -> None:
        with self._lock:
            self._r = 0
            self._w = 0
            self._size = 0

    def write(self, x: np.ndarray) -> None:
        if x.ndim != 1:
            x = x.reshape(-1)
        x = x.astype(np.float32, copy=False)
        n = int(x.size)
        if n <= 0:
            return

        with self._lock:
            if n >= self._cap:
                x = x[-self._cap :]
                n = self._cap
                self._r = 0
                self._w = 0
                self._size = 0

            overflow = max(0, (self._size + n) - self._cap)
            if overflow:
                self._r = (self._r + overflow) % self._cap
                self._size -= overflow

            first = min(n, self._cap - self._w)
            self._buf[self._w : self._w + first] = x[:first]
            rem = n - first
            if rem:
                self._buf[0:rem] = x[first:first + rem]

            self._w = (self._w + n) % self._cap
            self._size += n

    def read(self, frames: int) -> np.ndarray:
        frames = max(0, int(frames))
        if frames == 0:
            return np.zeros((0,), dtype=np.float32)

        out = np.zeros((frames,), dtype=np.float32)
        with self._lock:
            n = min(frames, self._size)
            if n <= 0:
                return out

            first = min(n, self._cap - self._r)
            out[:first] = self._buf[self._r : self._r + first]
            rem = n - first
            if rem:
                out[first:first + rem] = self._buf[0:rem]

            self._r = (self._r + n) % self._cap
            self._size -= n

        return out

@dataclass
class Preset:
    muted: bool = False
    match_output_volumes: bool = False
    mic_gain: float = 1.0

    playback_gain: float = 1.0  # primary output gain (virtual mic)
    headphones_gain: float = 1.0  # Headphones output gain

    gate_enabled: bool = True
    gate_threshold_db: float = -45.0  # dBFS
    gate_release_ms: float = 80.0

    comp_enabled: bool = True
    comp_threshold_db: float = -18.0
    comp_ratio: float = 4.0
    comp_makeup_db: float = 3.0

    eq_enabled: bool = True
    eq_low_db: float = 0.0 
    eq_high_db: float = 0.0  

    input: int = 0
    headphones: int = 0
    soundboard: int = 0

    auto_monitor_enabled: bool = True

class AudioEngine:
    def __init__(self) -> None:
        self._loading_ui = False
        self.stream: Optional[sd.Stream] = None
        self.headphones_stream: Optional[sd.OutputStream] = None

        self.preset = Preset()
        self.sample_rate = 44100
        self.blocksize = 512
        self.channels = 1

        self._gate_env = 0.0

        self._lp_state = 0.0
        self._hp_x_state = 0.0
        self._hp_y_state = 0.0

        self.last_rms = 0.0

        self._sb_lock = threading.Lock()
        self._sb_queue: Deque[np.ndarray] = deque()
        self._sb_current: Optional[np.ndarray] = None
        self._sb_pos = 0

        self._rec_lock = threading.Lock()
        self._recording = False
        self._rec_chunks: List[np.ndarray] = []

        self._headphones_lock = threading.Lock()
        self._headphones_device: Optional[int] = None
        self._headphones_rb: Optional[AudioRingBuffer] = None

    @property
    def is_monitoring(self) -> bool:
        return self.stream is not None

    def has_recording(self) -> bool:
        with self._rec_lock:
            return bool(self._rec_chunks)

    def start(
        self,
        input_device: int,
        output_device: int,
        headphones_device: Optional[int] = None,
        sample_rate: int = 44100,
        blocksize: int = 512,
    ) -> None:
        self.stop()

        self.sample_rate = int(sample_rate)
        self.blocksize = int(blocksize)

        self._gate_env = 0.0
        self._lp_state = 0.0
        self._hp_x_state = 0.0
        self._hp_y_state = 0.0
        self.last_rms = 0.0

        with self._headphones_lock:
            self._headphones_device = int(headphones_device) if headphones_device is not None else None
            self._headphones_rb = AudioRingBuffer(capacity_frames=int(self.sample_rate * 4))

        self.stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype="float32",
            channels=self.channels,
            callback=self._callback,
            device=(input_device, output_device),
        )
        self.stream.start()

        if headphones_device is not None:
            self.headphones_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                dtype="float32",
                channels=self.channels,
                callback=self._headphones_callback,
                device=int(headphones_device),
            )
            self.headphones_stream.start()

    def stop(self) -> None:
        if self.headphones_stream is not None:
            try:
                self.headphones_stream.stop()
            finally:
                self.headphones_stream.close()
                self.headphones_stream = None

        if self.stream is not None:
            try:
                self.stream.stop()
            finally:
     

Preview trimmed to first 10K characters
