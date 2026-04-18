'''
TODO:
* add auto saving for auto_monitor
* Icon not loading when downloading from tuna
* Restarting makes Icon disappear
* Deleting Folder just makes things reset to default but not save
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
    hear_self_enabled: bool = True

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
                self.stream.close()
                self.stream = None

        with self._headphones_lock:
            if self._headphones_rb is not None:
                self._headphones_rb.clear()

        self.stop_soundboard()
        self.stop_recording()

    def enqueue_sound(self, audio: np.ndarray) -> None:
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        audio = audio.astype(np.float32, copy=False)

        with self._sb_lock:
            self._sb_queue.append(audio)

    def stop_soundboard(self) -> None:
        with self._sb_lock:
            self._sb_queue.clear()
            self._sb_current = None
            self._sb_pos = 0

    def _pull_soundboard(self, frames: int) -> np.ndarray:
        out = np.zeros((frames,), dtype=np.float32)
        with self._sb_lock:
            written = 0
            while written < frames:
                if self._sb_current is None or self._sb_pos >= len(self._sb_current):
                    self._sb_current = self._sb_queue.popleft() if self._sb_queue else None
                    self._sb_pos = 0
                    if self._sb_current is None:
                        break

                remaining = frames - written
                chunk = self._sb_current[self._sb_pos : self._sb_pos + remaining]
                n = len(chunk)
                if n <= 0:
                    self._sb_current = None
                    self._sb_pos = 0
                    continue
                out[written : written + n] += chunk
                self._sb_pos += n
                written += n
        return out

    def start_recording(self) -> None:
        with self._rec_lock:
            self._rec_chunks.clear()
            self._recording = True

    def stop_recording(self) -> None:
        with self._rec_lock:
            self._recording = False

    def save_recording(self, path: str) -> None:
        with self._rec_lock:
            if not self._rec_chunks:
                raise RuntimeError("No audio recorded.")
            audio = np.concatenate(self._rec_chunks).astype(np.float32, copy=False)
        sf.write(path, audio, self.sample_rate)

    def _apply_gate(self, x: np.ndarray) -> np.ndarray:
        if not self.preset.gate_enabled:
            return x

        thr = db_to_lin(float(self.preset.gate_threshold_db))
        rel_ms = max(1.0, float(self.preset.gate_release_ms))
        release_coeff = float(np.exp(-((len(x) / self.sample_rate) / (rel_ms / 1000.0))))

        level = float(np.sqrt(np.mean(x * x) + 1e-12))
        if level >= thr:
            self._gate_env = 1.0
        else:
            self._gate_env *= release_coeff

        return x * self._gate_env

    def _apply_compressor(self, x: np.ndarray) -> np.ndarray:
        if not self.preset.comp_enabled:
            return x

        thr_db = float(self.preset.comp_threshold_db)
        ratio = max(1.0, float(self.preset.comp_ratio))
        makeup = db_to_lin(float(self.preset.comp_makeup_db))

        eps = 1e-12
        x_db = 20.0 * np.log10(np.maximum(np.abs(x), eps))

        over = x_db - thr_db
        gain_db = np.where(over > 0.0, -over * (1.0 - (1.0 / ratio)), 0.0)
        gain = np.power(10.0, gain_db / 20.0).astype(np.float32, copy=False)

        return x * gain * makeup

    def _apply_eq_tone(self, x: np.ndarray) -> np.ndarray:
        if not self.preset.eq_enabled:
            return x

        low_gain = db_to_lin(float(self.preset.eq_low_db))
        high_gain = db_to_lin(float(self.preset.eq_high_db))

        fc_low = 200.0
        fc_high = 3000.0
        alpha_low = float(np.exp(-2.0 * np.pi * fc_low / self.sample_rate))
        alpha_high = float(np.exp(-2.0 * np.pi * fc_high / self.sample_rate))

        low, self._lp_state = one_pole_lowpass(x, alpha_low, self._lp_state)
        high, self._hp_x_state, self._hp_y_state = one_pole_highpass(x, alpha_high, self._hp_x_state, self._hp_y_state)

        mid = x - low - high
        return (low * low_gain) + mid + (high * high_gain)

    def _callback(self, indata, outdata, frames, time_info, status) -> None:
        if status:
            print(status)

        mic = indata[:, 0].astype(np.float32, copy=False)
        self.last_rms = float(np.sqrt(np.mean(mic * mic) + 1e-12))

        if self.preset.muted:
            mic_proc = np.zeros_like(mic)
        else:
            mic_proc = mic * float(self.preset.mic_gain)
            mic_proc = self._apply_gate(mic_proc)
            mic_proc = self._apply_compressor(mic_proc)
            mic_proc = self._apply_eq_tone(mic_proc)

        if self.preset.match_output_volumes:
            # Make headphones follow primary playback gain INSIDE the preset
            self.preset.headphones_gain = self.preset.playback_gain


        sb = self._pull_soundboard(frames)
        if self.preset.hear_self_enabled:
            base = mic_proc + sb
        else:
            base = sb

        primary = soft_clip(base * float(self.preset.playback_gain), drive=1.2).astype(np.float32, copy=False)
        outdata[:, 0] = primary

        with self._rec_lock:
            if self._recording:
                self._rec_chunks.append(primary.copy())

        with self._headphones_lock:
            if self._headphones_device is not None and self._headphones_rb is not None:
                headphones = soft_clip(
                    base * float(self.preset.headphones_gain),
                    drive=1.2,
                ).astype(np.float32, copy=False)
                self._headphones_rb.write(headphones)

    def _headphones_callback(self, outdata, frames, time_info, status) -> None:
        if status:
            print(status)

        with self._headphones_lock:
            rb = self._headphones_rb
        if rb is None:
            outdata[:, 0] = 0.0
            return

        outdata[:, 0] = rb.read(frames)

class VoiceStudio(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Yet Another Soundboard - {PROFILE_NAME}")
        self.setMinimumWidth(980)
        self._monitor_started_by_auto = False

        self.engine = AudioEngine()
        self.sounds: Dict[str, Dict] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        tabs = QTabWidget()
        root.addWidget(tabs)

        mic_tab = QWidget()
        mic_layout = QVBoxLayout(mic_tab)
        mic_layout.setContentsMargins(12, 18, 12, 12)
        mic_layout.setSpacing(12)

        sb_tab = QWidget()
        sb_root = QVBoxLayout(sb_tab)
        sb_root.setContentsMargins(12, 18, 12, 12)
        sb_root.setSpacing(12)

        fx_tab = QWidget()
        fx_layout = QVBoxLayout(fx_tab)
        fx_layout.setContentsMargins(12, 18, 12, 12)
        fx_layout.setSpacing(12)

        preset_tab = QWidget()
        preset_layout = QVBoxLayout(preset_tab)
        preset_layout.setContentsMargins(12, 18, 12, 12)
        preset_layout.setSpacing(12)

        view_tab = QWidget()
        view_layout = QVBoxLayout(view_tab)
        view_layout.setContentsMargins(12, 18, 12, 12)
        view_layout.setSpacing(12)

        tuna_tab = QWidget()
        tuna_layout = QVBoxLayout(tuna_tab)
        tuna_layout.setContentsMargins(0, 0, 0, 0)
        tuna_layout.setSpacing(4)


        tabs.addTab(mic_tab, "Mic")
        tabs.addTab(sb_tab, "Soundboard")
        tabs.addTab(fx_tab, "Effects")
        tabs.addTab(preset_tab, "Presets")
        tabs.addTab(view_tab, "View")
        tabs.addTab(tuna_tab, "Tuna")

        
        # --------------------------------------------------------
        # Audio Devices
        # --------------------------------------------------------

        dev_group = QGroupBox("Audio Devices")
        dev_layout = QVBoxLayout(dev_group)

        self.input_combo = QComboBox()
        self.soundboard_combo = QComboBox()
        self.headphones_combo = QComboBox()

        self.input_combo.setEditable(False)
        self.soundboard_combo.setEditable(False)
        self.headphones_combo.setEditable(False)

        dev_layout.addWidget(QLabel("Microphone (Input):"))
        dev_layout.addWidget(self.input_combo)
        dev_layout.addWidget(QLabel("Headphones (Output):"))
        dev_layout.addWidget(self.headphones_combo)
        dev_layout.addWidget(QLabel("Virtual Mic (Output):"))
        dev_layout.addWidget(self.soundboard_combo)

        mic_layout.addWidget(dev_group)

        mix_group = QGroupBox("Mixer")
        mix_layout = QVBoxLayout(mix_group)

        self.mute_cb = QCheckBox("Mute mic")
        self.match_volumes_cb = QCheckBox("Match Output Volumes")

        self.mic_slider = self._make_slider(0, 200, 100)
        self.play_slider = self._make_slider(0, 200, 100)
        self.headphones_play_slider = self._make_slider(0, 200, 100)

        mix_layout.addWidget(self.mute_cb)
        mix_layout.addWidget(self.match_volumes_cb)
        mix_layout.addWidget(QLabel("Mic Gain"))
        mix_layout.addWidget(self.mic_slider)
        mix_layout.addWidget(QLabel("Primary Volume (Virtual Mic)"))
        mix_layout.addWidget(self.play_slider)
        mix_layout.addWidget(QLabel("Headphones Volume (Headphones)"))
        mix_layout.addWidget(self.headphones_play_slider)

        vu_row = QHBoxLayout()
        vu_row.addWidget(QLabel("VU"))
        self.vu = QProgressBar()
        self.vu.setRange(0, 100)
        self.vu.setTextVisible(False)
        vu_row.addWidget(self.vu, 1)
        mix_layout.addLayout(vu_row)

        mic_layout.addWidget(mix_group)

        trans_group = QGroupBox("Transport")
        trans_layout = QHBoxLayout(trans_group)

        self.auto_monitor_toggle = QCheckBox("Auto-Monitor")
        self.hear_self_toggle = QCheckBox("Hear Yourself")
        self.monitor_btn = QPushButton("Start Monitoring")
        self.record_btn = QPushButton("Start Recording")
        self.save_rec_btn = QPushButton("Save Recording…")
        self.save_rec_btn.setEnabled(False)

        trans_layout.addWidget(self.auto_monitor_toggle)
        trans_layout.addWidget(self.hear_self_toggle)
        trans_layout.addWidget(self.monitor_btn)
        trans_layout.addWidget(self.record_btn)
        trans_layout.addWidget(self.save_rec_btn)

        mic_layout.addWidget(trans_group)
        mic_layout.addStretch(1)
        
        # --------------------------------------------------------
        # Soundboard
        # --------------------------------------------------------

        sb_group = QGroupBox("Soundboard")
        sb_layout = QVBoxLayout(sb_group)

        self.sound_list = QListWidget()

        sb_btns = QHBoxLayout()
        self.add_sound_btn = QPushButton("Add Sound")
        self.remove_sound_btn = QPushButton("Remove")
        sb_btns.addWidget(self.add_sound_btn)
        sb_btns.addWidget(self.remove_sound_btn)

        sb_play = QHBoxLayout()
        self.play_sound_btn = QPushButton("Play Selected")
        self.stop_sounds_btn = QPushButton("Stop Sounds")
        sb_play.addWidget(self.play_sound_btn)
        sb_play.addWidget(self.stop_sounds_btn)

        sb_btns_2 = QHBoxLayout()
        self.rename_sounds_button = QPushButton("Rename Selected", self)
        self.remove_all_sounds_btn = QPushButton("Delete All Sounds")
        sb_btns_2.addWidget(self.rename_sounds_button)
        sb_btns_2.addWidget(self.remove_all_sounds_btn)

        sb_layout.addWidget(self.sound_list)
        sb_layout.addLayout(sb_btns)
        sb_layout.addLayout(sb_play)
        sb_layout.addLayout(sb_btns_2)

        sb_root.addWidget(sb_group)
        sb_root.addStretch(1)

        # --------------------------------------------------------
        # Effects
        # --------------------------------------------------------

        self.gate_group = self._make_checkable_group("Noise Gate", checked=True)
        gate_inner = QVBoxLayout(self.gate_group)
        self.gate_thr = self._make_slider(-80, -10, -45)
        self.gate_rel = self._make_slider(10, 400, 80)
        gate_inner.addWidget(QLabel("Threshold (dB)"))
        gate_inner.addWidget(self.gate_thr)
        gate_inner.addWidget(QLabel("Release (ms)"))
        gate_inner.addWidget(self.gate_rel)

        self.comp_group = self._make_checkable_group("Compressor", checked=True)
        comp_inner = QVBoxLayout(self.comp_group)
        self.comp_thr = self._make_slider(-40, 0, -18)
        self.comp_ratio = self._make_slider(1, 12, 4)
        self.comp_makeup = self._make_slider(0, 18, 3)
        comp_inner.addWidget(QLabel("Threshold (dB)"))
        comp_inner.addWidget(self.comp_thr)
        comp_inner.addWidget(QLabel("Ratio"))
        comp_inner.addWidget(self.comp_ratio)
        comp_inner.addWidget(QLabel("Makeup Gain (dB)"))
        comp_inner.addWidget(self.comp_makeup)

        self.eq_group = self._make_checkable_group("Tone EQ (Low/High)", checked=True)
        eq_inner = QVBoxLayout(self.eq_group)
        self.eq_low = self._make_slider(-12, 12, 0)
        self.eq_high = self._make_slider(-12, 12, 0)
        eq_inner.addWidget(QLabel("Low (dB)"))
        eq_inner.addWidget(self.eq_low)
        eq_inner.addWidget(QLabel("High (dB)"))
        eq_inner.addWidget(self.eq_high)

        fx_layout.addWidget(self.gate_group)
        fx_layout.addWidget(self.comp_group)
        fx_layout.addWidget(self.eq_group)
        fx_layout.addStretch(1)

        # --------------------------------------------------------
        # Presets
        # --------------------------------------------------------

        preset_group = QGroupBox("Presets")
        group_layout = QVBoxLayout(preset_group)
        preset_layout.addWidget(preset_group)

        top = QHBoxLayout()
        self.preset_name = QLineEdit()
        self.preset_name.setPlaceholderText("Preset name (e.g. 'Clean Voice')")
        self.save_preset_btn = QPushButton("Save Preset")
        top.addWidget(self.preset_name, 1)
        top.addWidget(self.save_preset_btn)

        bottom = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.load_preset_btn = QPushButton("Load Preset")
        self.export_preset_btn = QPushButton("Save as Default")
        bottom.addWidget(self.preset_combo, 1)
        bottom.addWidget(self.load_preset_btn)
        bottom.addWidget(self.export_preset_btn)

        group_layout.addLayout(top)
        group_layout.addLayout(bottom)
        preset_layout.addStretch(1)
        
        # --------------------------------------------------------
        # View
        # --------------------------------------------------------

        view_group = QGroupBox("View")
        view_group_layout = QVBoxLayout(view_group)
        view_layout.addWidget(view_group)

        self.reset_button = QPushButton("Reset to Defaults")
        self.background_color = QPushButton("Background Color")
        self.text_color = QPushButton("Text Color")
        self.muted_text_color = QPushButton("Muted Text Color")

        self.panel_color = QPushButton("Panel Color")
        self.panel_active_color = QPushButton("Panel Active Color")

        self.border_color = QPushButton("Border Color")
        self.border_hover_color = QPushButton("Border Hover Color")
        self.border_selected_color = QPushButton("Border Selected Color")
        self.border_disabled_color = QPushButton("Border Disabled Color")

        self.button_hover_color = QPushButton("Button Hover Color")
        self.button_pressed_color = QPushButton("Button Pressed Color")

        self.accent_color = QPushButton("Accent Color")
        self.accent_hover_color = QPushButton("Accent Hover Color")

        self.selection_color = QPushButton("Selection Color")

        self.slider_bg_color = QPushButton("Slider BG Color")
        self.disabled_text_color = QPushButton("Disabled Text Color")

        # Add Functions

        self.background_color.clicked.connect(self.change_bg_color)
        self.text_color.clicked.connect(self.change_text_color)
        self.muted_text_color.clicked.connect(self.change_muted_text_color)

        self.panel_color.clicked.connect(self.change_panel_color)
        self.panel_active_color.clicked.connect(self.change_panel_active_color)

        self.border_color.clicked.connect(self.change_border_color)
        self.border_hover_color.clicked.connect(self.change_border_hover_color)
        self.border_selected_color.clicked.connect(self.change_border_selected_color)
        self.border_disabled_color.clicked.connect(self.change_border_disabled_color)

        self.button_hover_color.clicked.connect(self.change_button_hover_color)
        self.button_pressed_color.clicked.connect(self.change_button_pressed_color)

        self.accent_color.clicked.connect(self.change_accent_color)
        self.accent_hover_color.clicked.connect(self.change_accent_hover_color)

        self.selection_color.clicked.connect(self.change_selection_color)

        self.slider_bg_color.clicked.connect(self.change_slider_bg_color)
        self.disabled_text_color.clicked.connect(self.change_disabled_text_color)

        # Add Widgets


        grid = QGridLayout()
        grid.setSpacing(8)

        buttons = [
            self.reset_button,

            self.background_color, self.text_color, self.muted_text_color,
            self.panel_color, self.panel_active_color,

            self.border_color, self.border_hover_color, self.border_selected_color, self.border_disabled_color,

            self.button_hover_color, self.button_pressed_color,

            self.accent_color, self.accent_hover_color,

            self.selection_color,

            self.slider_bg_color, self.disabled_text_color,
        ]

        cols = 4 

        for i, btn in enumerate(buttons):
            row = i // cols
            col = i % cols
            grid.addWidget(btn, row, col)

        view_group_layout.addLayout(grid)
        view_layout.addStretch(1)

        # --------------------------------------------------------
        # Tuna
        # --------------------------------------------------------

        self.tuna_view = QWebEngineView()
        tuna_layout.addWidget(self.tuna_view)

        self.tuna_profile = QWebEngineProfile("tuna_profile", self)

        if hasattr(self.tuna_profile, "setDownloadPath"):
            try:
                self.tuna_profile.setDownloadPath(MEDIA_DIR)
            except:
                pass

        self.tuna_profile.downloadRequested.connect(self._handle_tuna_download)

        self.tuna_page = QWebEnginePage(self.tuna_profile, self.tuna_view)
        self.tuna_view.setPage(self.tuna_page)

        self.tuna_view.load(QUrl("https://tuna.voicemod.net/"))


        # --------------------------------------------------------
        # Connections
        # --------------------------------------------------------

        self._populate_devices()
        self.headphones_combo.currentIndexChanged.connect(self._update_headphones_controls)

        self.mute_cb.stateChanged.connect(self._apply_ui_to_engine)
        self.match_volumes_cb.stateChanged.connect(self._apply_ui_to_engine)
        self.match_volumes_cb.stateChanged.connect(self._update_match_volume_ui)
        self.mic_slider.valueChanged.connect(self._apply_ui_to_engine)
        self.play_slider.valueChanged.connect(self._apply_ui_to_engine)
        self.headphones_play_slider.valueChanged.connect(self._apply_ui_to_engine)

        self.gate_group.toggled.connect(self._apply_ui_to_engine)
        self.gate_thr.valueChanged.connect(self._apply_ui_to_engine)
        self.gate_rel.valueChanged.connect(self._apply_ui_to_engine)

        self.comp_group.toggled.connect(self._apply_ui_to_engine)
        self.comp_thr.valueChanged.connect(self._apply_ui_to_engine)
        self.comp_ratio.valueChanged.connect(self._apply_ui_to_engine)
        self.comp_makeup.valueChanged.connect(self._apply_ui_to_engine)

        self.eq_group.toggled.connect(self._apply_ui_to_engine)
        self.eq_low.valueChanged.connect(self._apply_ui_to_engine)
        self.eq_high.valueChanged.connect(self._apply_ui_to_engine)

        self.add_sound_btn.clicked.connect(self._add_sound)
        self.remove_sound_btn.clicked.connect(self._remove_sound)
        self.play_sound_btn.clicked.connect(self._play_selected)
        self.stop_sounds_btn.clicked.connect(self.engine.stop_soundboard)
        self.rename_sounds_button.clicked.connect(self._rename_sounds)
        self.remove_all_sounds_btn.clicked.connect(self.remove_all_sounds)

        self.auto_monitor_toggle.toggled.connect(self._toggle_auto_monitoring)
        self.hear_self_toggle.toggled.connect(self._apply_ui_to_engine)
        self.monitor_btn.clicked.connect(self._toggle_monitor)
        self.record_btn.clicked.connect(self._toggle_record)
        self.save_rec_btn.clicked.connect(self._save_recording)

        self.save_preset_btn.clicked.connect(self._save_preset_file)
        self.load_preset_btn.clicked.connect(self._load_preset_file)
        self.export_preset_btn.clicked.connect(self._export_current_preset)

        self.vu_timer = QTimer(self)
        self.vu_timer.setInterval(100)
        self.vu_timer.timeout.connect(self._update_vu)
        self.vu_timer.start()

        self._update_headphones_controls()
        self.input_combo.currentIndexChanged.connect(self._restart_monitoring_if_active)
        self.soundboard_combo.currentIndexChanged.connect(self._restart_monitoring_if_active)
        self.headphones_combo.currentIndexChanged.connect(self._restart_monitoring_if_active)

        self.input_combo.currentIndexChanged.connect(self._on_input_device_changed)
        self.soundboard_combo.currentIndexChanged.connect(self._on_soundboard_device_changed)
        self.headphones_combo.currentIndexChanged.connect(self._on_headphones_device_changed)

        self.reset_button.clicked.connect(self.reset_theme)


        self.current_theme = DEFAULT_THEME.copy()
        apply_theme(QApplication.instance(), {"theme": self.current_theme})

        self._refresh_preset_dropdown()
        self.auto_monitor_toggle.setChecked(self.engine.preset.auto_monitor_enabled)
        if self.engine.preset.auto_monitor_enabled:
            self._monitor_started_by_auto = True
            self._start_monitoring()

    def _toggle_auto_monitoring(self, checked: bool) -> None:
        # If turning ON: start monitoring if not already running
        if checked:
            if not self.engine.is_monitoring:
                self._monitor_started_by_auto = True
                self._start_monitoring()
            return

        # If turning OFF: only stop if auto-monitor started it
        if self.engine.is_monitoring and self._monitor_started_by_auto:
            self._stop_monitoring()
        self._monitor_started_by_auto = False
    
        try:
            with open(PRESET_PATH, "w", encoding="utf-8") as f:
                json.dump(self._current_preset(), f, indent=2)
        except Exception as e:
            print("Failed to save auto-monitor state:", e)

    def _rename_sounds(self):
        file = self.sound_list.currentItem()
        name = file.text()
        if not file:
            return
        text, ok = QInputDialog.getText(self, "Rename sound", "Enter new name:")
        if ok:
            self.rename_sound(name, text)

    def rename_sound(self, name, new_name):
        if name not in self.sounds:
            return False

        # Sanitize new name (avoid invalid filenames)
        new_name = new_name.strip()
        if not new_name:
            return False

        if new_name in self.sounds:
            QMessageBox.warning(self, "Rename failed", f"A sound named '{new_name}' already exists.")
            return False

        snd = self.sounds[name]
        old_path = snd.get("path")

        if not old_path or not os.path.exists(old_path):
            QMessageBox.warning(self, "Rename failed", "Original sound file not found.")
            return False

        old_dir = os.path.dirname(old_path)
        old_ext = os.path.splitext(old_path)[1]
        new_path = os.path.join(old_dir, new_name + old_ext)

        try:
            # Rename main audio file
            os.rename(old_path, new_path)

            # Rename any matching image file
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif"):
                old_img = Path(old_path).with_suffix(ext)
                if old_img.exists():
                    new_img = Path(new_path).with_suffix(ext)
                    os.rename(old_img, new_img)

            # Update internal dictionary
            self.sounds[new_name] = {
                "path": new_path,
                "audio": snd["audio"],
                "sr": snd["sr"],
            }
            del self.sounds[name]

            # Update UI list item
            for i in range(self.sound_list.count()):
                item = self.sound_list.item(i)
                if item.text() == name:
                    item.setText(new_name)
                    break

            # Save updated preset.json
            with open(PRESET_PATH, "w", encoding="utf-8") as f:
                json.dump(self._current_preset(), f, indent=2)

            return True

        except Exception as e:
            QMessageBox.critical(self, "Rename failed", f"Could not rename sound:\n{e}")
            return False

    def remove_all_sounds(self):
        #ask user if they want to remove all sounds
        reply = QMessageBox.question(self, "Remove all sounds", "Are you sure you want to remove all sounds?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return
        for file in os.listdir(MEDIA_DIR):
            os.remove(os.path.join(MEDIA_DIR, file))
        self.sounds.clear()
        self.sound_list.clear()

    def add_custom_sound(self, name, path):
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=False)

            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            audio = normalize_audio(audio)

            self.sounds[name] = {
                "path": path,
                "audio": audio.astype(np.float32, copy=False),
                "sr": int(sr),
            }

            self.sound_list.addItem(name)
            try:
                with open(PRESET_PATH, "w", encoding="utf-8") as f:
                    json.dump(self._current_preset(), f, indent=2)
            except Exception as e:
                print("Failed to save preset after add_custom_sound:", e)

        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"Could not load downloaded sound:\n{e}")

    def _handle_tuna_download(self, download):
        import re

        def normalize(name):
            name = re.sub(r"\s\(\d+\)(\.[^.]+)$", r"\1", name)
            return name

        incoming = download.downloadFileName()
        filename = normalize(incoming)

        os.makedirs(MEDIA_DIR, exist_ok=True)
        target_path = os.path.join(MEDIA_DIR, filename)

        # Prevent duplicates
        if os.path.exists(target_path):
            print(f"Already downloaded: {target_path}")
            try:
                download.cancel()
            except:
                pass
            return
        fake_path = re.sub(r'(?i)\b-made-with-Voicemod\b[-_\s]*', '', filename)
        fake_name = normalize(fake_path)
        print(f'Downloading: "{fake_name}"')

        # Force exact path (works on more Qt builds)
        if hasattr(download, "setPath"):
            download.setPath(target_path)
        else:
            if hasattr(download, "setDownloadDirectory"):
                download.setDownloadDirectory(MEDIA_DIR)
            if hasattr(download, "setDownloadFileName"):
                download.setDownloadFileName(filename)

        download.isFinishedChanged.connect(
            lambda *args, d=download, p=target_path: self._on_tuna_download_finished(d, p)
        )

        download.accept()

    def _on_tuna_download_finished(self, download, path):
        try:
            reason = download.interruptReason()
        except:
            reason = None

        if reason is None or str(reason).endswith("NoReason"):
            filename = download.downloadFileName()
            path = re.sub(r'(?i)\b-made-with-Voicemod\b[-_\s]*', '', filename)
            os.rename(os.path.join(MEDIA_DIR, filename), os.path.join(MEDIA_DIR, path))

            print(f"Downloaded: {path}")
            # add sound
            full_path = os.path.join(MEDIA_DIR, path)
            self.add_custom_sound(Path(path).stem, full_path)

        else:
            print(f"Download failed: {reason}")

    def change_bg_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            set_theme_value(self, "bg", color.name())

    def change_text_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            set_theme_value(self, "text", color.name())

    def change_muted_text_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            set_theme_value(self, "muted_text", color.name())

    def change_panel_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            set_theme_value(self, "panel", color.name())

    def change_panel_active_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            set_theme_value(self, "panel_active", color.name())

    def change_border_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            set_theme_value(self, "border", color.name())

    def change_border_hover_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            set_theme_value(self, "border_hover", color.name())

    def change_border_selected_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            set_theme_value(self, "border_selected", color.name())

    def change_border_disabled_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            set_theme_value(self, "border_disabled", color.name())

    def change_button_hover_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            set_theme_value(self, "button_hover", color.name())

    def change_button_pressed_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            set_theme_value(self, "button_pressed", color.name())

    def change_accent_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            set_theme_value(self, "accent", color.name())

    def change_accent_hover_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            set_theme_value(self, "accent_hover", color.name())

    def change_selection_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            set_theme_value(self, "selection", color.name())

    def change_slider_bg_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            set_theme_value(self, "slider_bg", color.name())

    def change_disabled_text_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            set_theme_value(self, "disabled_text", color.name())

    def reset_theme(app: QApplication):
        for i in range(len(DEFAULT_THEME)):
            set_theme_value(app, list(DEFAULT_THEME.keys())[i], list(DEFAULT_THEME.values())[i])

    def _restart_monitoring_if_active(self) -> None:
        if self.engine.is_monitoring:
            self._stop_monitoring()
            self._start_monitoring()

    def _make_slider(self, mn: int, mx: int, val: int) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(mn, mx)
        s.setValue(val)
        return s

    def _make_checkable_group(self, title: str, checked: bool = True) -> QGroupBox:
        g = QGroupBox(title)
        g.setCheckable(True)
        g.setChecked(checked)
        return g

    def _update_headphones_controls(self) -> None:
        enabled = self.headphones_combo.currentIndex() != 0
        self.headphones_play_slider.setEnabled(enabled)

    def _update_headphones_controls(self) -> None:
        enabled = self.headphones_combo.currentIndex() != 0
        self.headphones_play_slider.setEnabled(enabled)

    def _update_match_volume_ui(self):
        """Visually lock the headphones slider when matching volumes."""
        if self.match_volumes_cb.isChecked():
            # Disable + show lock hint
            self.headphones_play_slider.setEnabled(False)
            self.headphones_play_slider.setToolTip("🔒 Locked: Matching Primary Volume")

            # Make the handle look 'locked'
            self.headphones_play_slider.setStyleSheet("""
                QSlider::handle:horizontal {
                    background: #5a5a5a;
                    border: 1px solid #444;
                }
                QSlider::sub-page:horizontal {
                    background: #5a5a5a;
                }
            """)
        else:
            # Restore normal behavior
            self.headphones_play_slider.setEnabled(
                self.headphones_combo.currentIndex() != 0
            )
            self.headphones_play_slider.setToolTip("")

            # Restore normal handle styling (uses your theme)
            self.headphones_play_slider.setStyleSheet("")

    def _populate_devices(self) -> None:
        self._loading_ui = True
        # Block signals so changing indexes does not overwrite preset
        self.input_combo.blockSignals(True)
        self.soundboard_combo.blockSignals(True)
        self.headphones_combo.blockSignals(True)

        self.input_combo.clear()
        self.soundboard_combo.clear()
        self.headphones_combo.clear()

        self._input_ids = []
        self._output_ids = []
        self._headphones_output_ids = [None]

        # Headphones can be disabled
        self.headphones_combo.addItem("None")

        devices = sd.query_devices()

        for i, dev in enumerate(devices):
            in_ch = int(dev.get("max_input_channels", 0) or 0)
            out_ch = int(dev.get("max_output_channels", 0) or 0)
            name = f"{i}: {dev['name']}"

            if in_ch >= 1:
                self.input_combo.addItem(name)
                self._input_ids.append(i)

            if out_ch >= 1:
                self.soundboard_combo.addItem(name)
                self._output_ids.append(i)

                self.headphones_combo.addItem(name)
                self._headphones_output_ids.append(i)

        # Set OS defaults ONLY if preset is still at default values
        try:
            default_in, default_out = sd.default.device

            if self.engine.preset.input == 0 and default_in in self._input_ids:
                self.input_combo.setCurrentIndex(self._input_ids.index(default_in))

            if self.engine.preset.soundboard == 0 and default_out in self._output_ids:
                self.soundboard_combo.setCurrentIndex(self._output_ids.index(default_out))

        except Exception:
            pass

        self._update_headphones_controls()

        # Re-enable signals
        self.input_combo.blockSignals(False)
        self.soundboard_combo.blockSignals(False)
        self.headphones_combo.blockSignals(False)
        self._loading_ui = False

    def _start_monitoring(self) -> None:
        if not self._input_ids or not self._output_ids:
            QMessageBox.critical(self, "No devices", "No input/output devices found.")
            return

        in_dev = self._input_ids[self.input_combo.currentIndex()]
        out_dev = self._output_ids[self.soundboard_combo.currentIndex()]

        headphones_idx = self.headphones_combo.currentIndex()
        headphones_dev = self._headphones_output_ids[headphones_idx] if 0 <= headphones_idx < len(self._headphones_output_ids) else None

        try:
            self.engine.start(in_dev, out_dev, headphones_device=headphones_dev, sample_rate=44100, blocksize=512)
        except Exception as e:
            QMessageBox.critical(self, "Audio start failed", str(e))
            return

        self.monitor_btn.setText("Stop Monitoring")

    def _stop_monitoring(self) -> None:
        self.engine.stop()
        self.monitor_btn.setText("Start Monitoring")
        self.record_btn.setText("Start Recording")
        self.save_rec_btn.setEnabled(self.engine.has_recording())

    def _apply_ui_to_engine(self) -> None:
        if getattr(self, "_loading_ui", False):
            return

        p = self.engine.preset

        p.muted = self.mute_cb.isChecked()
        p.match_output_volumes = self.match_volumes_cb.isChecked()
        p.mic_gain = self.mic_slider.value() / 100.0
        p.playback_gain = self.play_slider.value() / 100.0
        p.headphones_gain = self.headphones_play_slider.value() / 100.0

        if p.match_output_volumes:
            p.headphones_gain = p.playback_gain
            self.headphones_play_slider.setValue(int(round(p.headphones_gain * 100)))

        p.gate_enabled = self.gate_group.isChecked()
        p.gate_threshold_db = float(self.gate_thr.value())
        p.gate_release_ms = float(self.gate_rel.value())

        p.comp_enabled = self.comp_group.isChecked()
        p.comp_threshold_db = float(self.comp_thr.value())
        p.comp_ratio = float(self.comp_ratio.value())
        p.comp_makeup_db = float(self.comp_makeup.value())

        p.eq_enabled = self.eq_group.isChecked()
        p.eq_low_db = float(self.eq_low.value())
        p.eq_high_db = float(self.eq_high.value())

        p.auto_monitor_enabled = self.auto_monitor_toggle.isChecked()
        p.hear_self_enabled = self.hear_self_toggle.isChecked()

    def _on_input_device_changed(self, idx: int):
        self.engine.preset.input = idx

    def _on_soundboard_device_changed(self, idx: int):
        self.engine.preset.soundboard = idx

    def _on_headphones_device_changed(self, idx: int):
        self.engine.preset.headphones = idx

    def _add_sound(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Add sound (WAV/FLAC/OGG supported)",
            "",
            "Audio Files (*.wav *.flac *.ogg *.mp3 *.m4a *.mp4 );;All Files (*.*)",
        )
        if not path:
            return
        try:
            os.makedirs(MEDIA_DIR, exist_ok=True)
            media_path = os.path.join(MEDIA_DIR, os.path.basename(path))
            if path != media_path:
                try:
                    import shutil
                    shutil.copy(path, media_path)
                    path = media_path
                except Exception as e:
                    print("Failed to copy media file:", e)

            audio, sr = sf.read(path, dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            audio = normalize_audio(audio)

            name = Path(path).stem
            base = name
            k = 2
            while name in self.sounds:
                name = f"{base} ({k})"
                k += 1

            self.sounds[name] = {"path": path, "audio": audio.astype(np.float32, copy=False), "sr": int(sr)}

            img_path = Path(path).with_suffix(".png")
            if not img_path.exists():
                img_path = Path(path).with_suffix(".jpg")
            if not img_path.exists():
                img_path = Path(path).with_suffix(".jpeg")
            if not img_path.exists():
                img_path = Path(path).with_suffix(".bmp")
            if not img_path.exists():
                img_path = Path(path).with_suffix(".gif")
            if not img_path.exists():
                img_path = DEFAULT_IMG_PATH

            item = QListWidgetItem(name)
            if os.path.exists(img_path):
                item.setIcon(QIcon(QPixmap(str(img_path)).scaled(48, 48, Qt.KeepAspectRatio)))
            self.sound_list.addItem(item)

        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))

    def _remove_sound(self) -> None:
        item = self.sound_list.currentItem()
        if not item:
            return

        name = item.text()
        snd = self.sounds.get(name)
        if not snd:
            self.sound_list.takeItem(self.sound_list.currentRow())
            return

        reply = QMessageBox.question(
            self,
            "Remove sound",
            f"Delete '{name}' permanently (including the audio file)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.No:
            return

        try:
            path = snd.get("path")
            if path and os.path.exists(path):
                os.remove(path)
        except Exception as e:
            QMessageBox.warning(self, "File delete failed", f"Could not delete file:\n{e}")

        self.sounds.pop(name, None)
        self.sound_list.takeItem(self.sound_list.currentRow())

        try:
            with open(PRESET_PATH, "w", encoding="utf-8") as f:
                json.dump(self._current_preset(), f, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Preset save failed", f"Sound removed, but preset not saved:\n{e}")

        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif"):
            img_path = Path(path).with_suffix(ext)
            try:
                if img_path.exists():
                    os.remove(img_path)
            except Exception:
                pass

    def _play_selected(self) -> None:
        item = self.sound_list.currentItem()
        if not item:
            return
        name = item.text()
        snd = self.sounds.get(name)
        if not snd:
            return

        if not self.engine.is_monitoring:
            QMessageBox.information(self, "Not monitoring", "Start Monitoring first.")
            return

        audio = snd.get("audio")
        sr = int(snd.get("sr", 0))

        if audio is None or sr <= 0:
            QMessageBox.warning(self, "Invalid sound", "This sound is not loaded correctly. Try re-adding it.")
            return

        audio = resample_mono_linear(audio, sr, self.engine.sample_rate)

        if len(audio) == 0:
            return

        self.engine.enqueue_sound(audio)

    def _toggle_monitor(self) -> None:
        if not self.engine.is_monitoring:
            # manual start
            self._monitor_started_by_auto = False
            self._start_monitoring()
        else:
            self._stop_monitoring()
            self._monitor_started_by_auto = False

    def _toggle_record(self) -> None:
        if not self.engine.is_monitoring:
            QMessageBox.information(
                self,
                "Not monitoring",
                "Start Monitoring first (recording captures the monitored mix).",
            )
            return

        if self.record_btn.text().startswith("Start"):
            self.engine.start_recording()
            self.save_rec_btn.setEnabled(False)
            self.record_btn.setText("Stop Recording")
        else:
            self.engine.stop_recording()
            self.record_btn.setText("Start Recording")
            self.save_rec_btn.setEnabled(self.engine.has_recording())

    def _save_recording(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save recording",
            f"recording_{int(time.time())}.wav",
            "WAV (*.wav)",
        )
        if not path:
            return
        try:
            self.engine.save_recording(path)
            QMessageBox.information(self, "Saved", f"Saved recording:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _current_preset(self) -> Dict:
        sounds = {name: data["path"] for name, data in self.sounds.items()}

        return {
            "preset": asdict(self.engine.preset),
            "sounds": sounds,
            "theme": self.current_theme.copy(),
        }

    def _apply_preset_to_ui(self, data: Dict) -> None:
        self._loading_ui = True
        theme = data.get("theme", {})
        self.current_theme = DEFAULT_THEME.copy()
        self.current_theme.update(theme)

        apply_theme(QApplication.instance(), {"theme": self.current_theme})
        #print("Loaded theme:", data.get("theme"))

        preset_dict = dict(data["preset"])
        if "headphones_gain" not in preset_dict:
            preset_dict["headphones_gain"] = 1.0

        p = Preset(**preset_dict)
        self.engine.preset = p
        
        self.input_combo.blockSignals(True)
        self.soundboard_combo.blockSignals(True)
        self.headphones_combo.blockSignals(True)

        self.auto_monitor_toggle.setChecked(p.auto_monitor_enabled)
        self.hear_self_toggle.setChecked(p.hear_self_enabled)

        self.sounds.clear()
        self.sound_list.clear()
        for name, path in data.get("sounds", {}).items():
            try:
                audio, sr = sf.read(path, dtype="float32", always_2d=False)
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                self.sounds[name] = {"path": path, "audio": audio.astype(np.float32, copy=False), "sr": int(sr)}
                self.sound_list.addItem(name)
            except Exception as e:
                print(f"Could not load sound {name} from {path}:", e)

        # ---- Restore UI values ----
        self.mute_cb.setChecked(p.muted)
        self.match_volumes_cb.setChecked(p.match_output_volumes)
        self._update_match_volume_ui()
        self.mic_slider.setValue(int(round(p.mic_gain * 100)))
        self.play_slider.setValue(int(round(p.playback_gain * 100)))
        self.headphones_play_slider.setValue(int(round(p.headphones_gain * 100)))

        self.gate_group.setChecked(p.gate_enabled)
        self.gate_thr.setValue(int(round(p.gate_threshold_db)))
        self.gate_rel.setValue(int(round(p.gate_release_ms)))

        self.comp_group.setChecked(p.comp_enabled)
        self.comp_thr.setValue(int(round(p.comp_threshold_db)))
        self.comp_ratio.setValue(int(round(p.comp_ratio)))
        self.comp_makeup.setValue(int(round(p.comp_makeup_db)))

        self.eq_group.setChecked(p.eq_enabled)
        self.eq_low.setValue(int(round(p.eq_low_db)))
        self.eq_high.setValue(int(round(p.eq_high_db)))

        # ---- Restore device selections (SAFE) ----
        input_idx = int(p.input)
        soundboard_idx = int(p.soundboard)
        headphones_idx = int(p.headphones)

        if 0 <= input_idx < self.input_combo.count():
            self.input_combo.setCurrentIndex(input_idx)

        if 0 <= soundboard_idx < self.soundboard_combo.count():
            self.soundboard_combo.setCurrentIndex(soundboard_idx)

        if 0 <= headphones_idx < self.headphones_combo.count():
            self.headphones_combo.setCurrentIndex(headphones_idx)
        else:
            self.headphones_combo.setCurrentIndex(0)

        self._update_headphones_controls()

        # Re-enable signals
        self.input_combo.blockSignals(False)
        self.soundboard_combo.blockSignals(False)
        self.headphones_combo.blockSignals(False)

        # Now sync UI -> engine ONCE
        self._apply_ui_to_engine()
        self._loading_ui = False

    def _preset_folder(self, name: str) -> str:
        return os.path.join(APPDATA_DIR, name)

    def _preset_path(self, name: str) -> str:
        return os.path.join(self._preset_folder(name), "preset.json")

    def _settings_path(self) -> str:
        return os.path.join(APPDATA_DIR, "settings.json")

    def _get_current_preset_name(self) -> str:
        try:
            if not os.path.isfile(self._settings_path()):
                return "Default"
            with open(self._settings_path(), "r", encoding="utf-8") as f:
                return json.load(f).get("Current Preset", "Default")
        except Exception:
            return "Default"

    def _set_current_preset_name(self, name: str) -> None:
        os.makedirs(APPDATA_DIR, exist_ok=True)
        with open(self._settings_path(), "w", encoding="utf-8") as f:
            json.dump({"Current Preset": name}, f, indent=2)

    def _refresh_preset_dropdown(self) -> None:
        os.makedirs(APPDATA_DIR, exist_ok=True)

        items = []
        for entry in os.listdir(APPDATA_DIR):
            folder = os.path.join(APPDATA_DIR, entry)
            if os.path.isdir(folder) and os.path.isfile(os.path.join(folder, "preset.json")):
                items.append(entry)

        items.sort(key=str.lower)

        current = self._get_current_preset_name()

        self.preset_combo.clear()
        self.preset_combo.addItems(items)

        if current in items:
            self.preset_combo.setCurrentText(current)

    def _save_preset_file(self) -> None:
        name = self.preset_name.text().strip() or "Default"

        try:
            folder = self._preset_folder(name)
            os.makedirs(folder, exist_ok=True)

            path = self._preset_path(name)

            p = self._current_preset()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(p, f, indent=2)

            self._set_current_preset_name(name)
            self._refresh_preset_dropdown()
            self.preset_combo.setCurrentText(name)

            QMessageBox.information(self, "Preset saved", "Preset saved successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Preset save failed", str(e))

    def _load_preset_file(self) -> None:
        name = self.preset_combo.currentText().strip()
        if not name:
            return

        path = self._preset_path(name)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._apply_preset_to_ui(data)
            self._set_current_preset_name(name)

        except Exception as e:
            QMessageBox.critical(self, "Preset load failed", str(e))
        self.setWindowTitle(f"Yet Another Soundboard - {self._get_current_preset_name()}")

    def _export_current_preset(self) -> None:
        json.dumps(self._current_preset(), indent=2)
        preset_name = self._get_current_preset_name()
        QMessageBox.information(self, f"Default Preset: {preset_name}", "Preset saved successfully.")

    def _update_vu(self) -> None:
        try:
            val = int(min(max(self.engine.last_rms * 100.0, 0.0), 100.0))
            self.vu.setValue(val)
        except RuntimeError:
            self.vu_timer.stop()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.vu_timer.stop()
        self.engine.stop()
        try:
            with open(PRESET_PATH, "w", encoding="utf-8") as f:
                json.dump(self._current_preset(), f, indent=2)
                #print("SAVING PRESET:", self.engine.preset)
        except Exception as e:
            print("Failed to save preset on exit:", e)
        super().closeEvent(event)

QSS_TEMPLATE = """
QWidget {{
    background-color: {bg};
    color: {text};
    font-family: "Inter","Segoe UI",system-ui;
    font-size: 13px;
}}

/* Tabs */
QTabWidget::pane {{
    border: 0;
    background: transparent;
}}
QTabBar::tab {{
    background: {panel};
    color: {muted_text};
    border: 1px solid {border};
    border-bottom: 0;
    padding: 10px 14px;
    margin-right: 6px;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
}}
QTabBar::tab:hover {{
    color: {text};
}}
QTabBar::tab:selected {{
    background: {panel_active};
    color: {text};
    border-color: {border_selected};
}}

/* Grouping */
QGroupBox {{
    background-color: {panel};
    border: 1px solid {border};
    border-radius: 12px;
    margin-top: 18px;
    padding: 12px;
    padding-top: 18px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 0px;
    padding: 0 8px;
    color: {muted_text};
    font-weight: 600;
    background-color: {panel};
}}

/* Inputs */
QLineEdit, QComboBox, QListWidget {{
    background-color: {panel};
    border: 1px solid {border};
    border-radius: 10px;
    padding: 8px 10px;
    selection-background-color: {selection};
}}
QLineEdit:focus, QComboBox:focus, QListWidget:focus {{
    border: 1px solid {accent};
    outline: 0;
}}
QComboBox::drop-down {{
    border: 0;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background: {panel};
    border: 1px solid {border};
    outline: 0;
    selection-background-color: {selection};
    padding: 6px;
}}

QListWidget::item {{
    padding: 8px 10px;
    border-radius: 8px;
}}
QListWidget::item:hover {{
    background: {panel_active};
}}
QListWidget::item:selected {{
    background: {selection};
}}

/* Buttons */
QPushButton {{
    background-color: {panel_active};
    border: 1px solid {border};
    border-radius: 10px;
    padding: 8px 12px;
}}
QPushButton:hover {{
    background-color: {button_hover};
    border-color: {border_hover};
}}
QPushButton:pressed {{
    background-color: {button_pressed};
}}
QPushButton:disabled {{
    color: {disabled_text};
    background-color: {panel};
    border-color: {border_disabled};
}}

/* Checkbox */
QCheckBox {{
    spacing: 10px;
    background-color: transparent;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 6px;
    border: 1px solid {border};
    background: {panel};
}}
QCheckBox::indicator:checked {{
    background: {accent};
    border-color: {accent};
}}
QCheckBox::indicator:checked:hover {{
    background: {accent_hover};
    border-color: {accent_hover};
}}

/* Slider */
QSlider::groove:horizontal {{
    height: 6px;
    background: {slider_bg};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {accent};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 18px;
    margin: -7px 0;
    border-radius: 9px;
    background: {accent};
    border: 1px solid {accent};
}}
QSlider::handle:horizontal:hover {{
    background: {accent_hover};
    border-color: {accent_hover};
}}

/* Progress */
QProgressBar {{
    height: 10px;
    background: {slider_bg};
    border: 1px solid {border};
    border-radius: 6px;
}}
QProgressBar::chunk {{
    background: {accent};
    border-radius: 6px;
}}

/* Scrollbars */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {border};
    border-radius: 5px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {border_hover};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}

QLabel {{
    color: {text};
    background-color: transparent;
}}
"""

DEFAULT_THEME = {
    "bg": "#0f1115",
    "text": "#e6e9ef",
    "muted_text": "#9aa4b2",

    "panel": "#151823",
    "panel_active": "#1f2430",

    "border": "#2a2f3a",
    "border_hover": "#3a4152",
    "border_selected": "#3a4152",
    "border_disabled": "#1b1f27",

    "button_hover": "#242a37",
    "button_pressed": "#1a1f29",

    "accent": "#7c5cff",
    "accent_hover": "#8b73ff",

    "selection": "#2b3350",

    "slider_bg": "#1b2030",

    "disabled_text": "#677284",
}

def apply_theme(app: QApplication, settings: dict):
    app.setStyle("Fusion")
    app.setFont(QFont("Inter", 10))

    theme = DEFAULT_THEME.copy()
    if settings and "theme" in settings:
        theme.update(settings["theme"])

    qss = QSS_TEMPLATE.format(**theme)
    app.setStyleSheet(qss)
    for w in app.allWidgets():
        w.style().unpolish(w)
        w.style().polish(w)
        w.update()

def set_theme_value(self, key: str, value: str):
    self.current_theme[key] = value
    apply_theme(QApplication.instance(), {"theme": self.current_theme})

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = VoiceStudio()
    _load_auto_preset(w)
    w.show()
    sys.exit(app.exec())
