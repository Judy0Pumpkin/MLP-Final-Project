"""
ui.py

python -m pip install flask
python -m pip install flask-cors

啟動方式：python ui.py

負責：
    1. Flask state server（/state、/toggle、/shutdown）
    2. 背景執行緒跑推理迴圈（呼叫 inference.py）
    3. 讀取 monitor.html 啟動 PyQt5 / pywebview 視窗

inference.py、model.py、constants.py 不需要修改。
"""

import sys, os, threading, time
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import torch

from constants import SAMPLE_RATE, N_MELS, HOP_LENGTH, FIXED_FRAMES, WINDOW_SECONDS
from inference_new_arch import audio_to_mel, run_beat_tcn, smooth_bpm
import sounddevice as sd

# ── 預載所有可用模型 ──────────────────────────────────────
_loaded_models = {}   # key: 'v1' | 'v2'

def _load_all_models(device):
    base = os.path.dirname(os.path.abspath(__file__))
    ckpts = {
        'v1': os.path.join(base, 'checkpoints', 'best_model_F=0.6457.pt'),
        'v2': os.path.join(base, 'checkpoints', 'best_model_new_arch.pt'),
    }
    for name, path in ckpts.items():
        if not os.path.exists(path):
            print(f"  [{name}] checkpoint not found: {path}")
            continue
        try:
            if name == 'v1':
                import model as _m; cls = _m.BeatTCN
            else:
                import model_new_arch as _m2; cls = _m2.BeatTCN
            m = cls().to(device)
            ckpt = torch.load(path, map_location=device)
            m.load_state_dict(ckpt['model_state'])
            m.eval()
            _loaded_models[name] = m
            print(f"  [{name}] loaded  epoch={ckpt.get('epoch','?')}  val_F={ckpt.get('val_f',0):.4f}")
        except Exception as e:
            print(f"  [{name}] load failed: {e}")
    avail = ['librosa'] + list(_loaded_models.keys())
    with state_lock:
        state['available_models'] = avail
    print(f"  Available: {avail}")

# ────────────────────────────────────────────────────────
# 1. Flask State Server
# ────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

state = {
    "is_live": False, "loop": 0,
    "bpm": None, "prev_bpm": None,
    "beat_count": 0, "beat_times": [],
    "amplitude_max": 0.0, "amplitude_mean": 0.0,
    "mel_shape": [1, N_MELS, FIXED_FRAMES],
    "buffer_fill": 0.0, "silent": False,
    "window_seconds": WINDOW_SECONDS,
    "window_size": SAMPLE_RATE * WINDOW_SECONDS,
    "sample_rate": SAMPLE_RATE,
    "waveform": [], "beat_act": [],
    "downbeat_times": [], "tempo_bpm": None,
    "model": "–", "streaming": False,
    "selected_model": "v2",
    "available_models": ["librosa"],
}
state_lock = threading.Lock()

@app.route("/state")
def get_state():
    with state_lock:
        return jsonify(dict(state))

@app.route("/ping")
def ping():
    return "pong"

@app.route("/toggle", methods=["POST"])
def toggle_streaming():
    with state_lock:
        state["streaming"] = not state["streaming"]
        current = state["streaming"]
    print(f"  [UI] streaming {'開啟 ▶' if current else '暫停 ■'}")
    return jsonify({"streaming": current})

@app.route("/set_model", methods=["POST"])
def set_model():
    name = (request.get_json() or {}).get("model", "")
    with state_lock:
        avail = state["available_models"]
    if name not in avail:
        return jsonify({"error": f"model '{name}' not available", "available": avail}), 400
    with state_lock:
        state["selected_model"] = name
        state["model"] = name
    print(f"  [UI] model switched → {name}")
    return jsonify({"ok": True, "model": name})

@app.route("/shutdown", methods=["POST"])
def shutdown():
    def _exit():
        time.sleep(0.3)
        os._exit(0)
    threading.Thread(target=_exit, daemon=True).start()
    return jsonify({"ok": True})

def run_server(port=5050):
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


# ────────────────────────────────────────────────────────
# 2. 推理迴圈
# ────────────────────────────────────────────────────────
SILENCE_RMS = 0.01   # RMS 低於此值視為無音訊，跳過推理

def downsample(arr, target=512):
    idxs = np.linspace(0, len(arr) - 1, target).astype(int)
    return arr[idxs].tolist()


def get_audio_with_update():
    """邊錄音邊即時更新 state waveform"""
    total_samples = int(SAMPLE_RATE * WINDOW_SECONDS)
    collected = []
    filled_count = 0

    def callback(indata, _frames, _time_info, _status):
        nonlocal filled_count
        chunk = indata[:, 0].copy()
        collected.append(chunk)
        filled_count += len(chunk)
        idxs = np.linspace(0, len(chunk) - 1, min(512, len(chunk))).astype(int)
        with state_lock:
            state["buffer_fill"] = min(1.0, filled_count / total_samples)
            state["waveform"]    = chunk[idxs].tolist()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=512, dtype="float32",
                        callback=callback):
        time.sleep(WINDOW_SECONDS)

    audio = np.concatenate(collected)
    if len(audio) < total_samples:
        audio = np.pad(audio, (0, total_samples - len(audio)))
    else:
        audio = audio[:total_samples]
    with state_lock:
        state["buffer_fill"] = 1.0
    return audio


def inference_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Device: {device} ===")
    _load_all_models(device)

    with state_lock:
        state["is_live"] = True
        sel = state["selected_model"]
        state["model"] = sel

    prev_bpm, loop = None, 0

    while True:
        with state_lock:
            streaming = state["streaming"]
            sel = state["selected_model"]
        if not streaming:
            time.sleep(0.2)
            continue

        loop += 1
        with state_lock:
            state["loop"]        = loop
            state["buffer_fill"] = 0.0
            state["silent"]      = False
            state["model"]       = sel
        print(f"── Round {loop} [{sel}] ──")

        y = get_audio_with_update()

        rms = float(np.sqrt(np.mean(y ** 2)))
        if rms < SILENCE_RMS:
            print(f"  [無音訊，跳過]  RMS={rms:.4f} < {SILENCE_RMS}")
            with state_lock:
                state["silent"] = True
            continue

        mel_tensor = audio_to_mel(y)

        if sel in _loaded_models:
            beat_times, bpm, beat_act, downbeat_times, tempo_bpm = run_beat_tcn(
                _loaded_models[sel], mel_tensor, device)
        else:
            # librosa fallback
            import librosa
            tempo, beat_frames = librosa.beat.beat_track(
                y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
            beat_times     = librosa.frames_to_time(beat_frames, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
            bpm            = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
            beat_act       = None
            downbeat_times = np.array([])
            tempo_bpm      = bpm

        bpm = smooth_bpm(bpm)
        print(f"  BPM={bpm:.1f}  beats={len(beat_times)}  downbeats={len(downbeat_times)}")

        with state_lock:
            state["prev_bpm"]        = prev_bpm
            state["bpm"]             = round(bpm, 1)
            state["beat_count"]      = int(len(beat_times))
            state["beat_times"]      = [round(float(t), 3) for t in beat_times]
            state["downbeat_times"]  = [round(float(t), 3) for t in downbeat_times]
            state["tempo_bpm"]       = round(float(tempo_bpm), 1) if tempo_bpm else None
            state["amplitude_max"]   = round(float(y.max()), 4)
            state["amplitude_mean"]  = round(float(np.abs(y).mean()), 5)
            state["mel_shape"]       = list(mel_tensor.shape)
            state["silent"]          = False
            state["beat_act"]        = downsample(beat_act, 256) if beat_act is not None else []

        prev_bpm = round(bpm, 1)


# ────────────────────────────────────────────────────────
# 3. 視窗
# ────────────────────────────────────────────────────────

def get_html():
    html_path = os.path.join(os.path.dirname(__file__), "monitor.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return f.read()
    raise FileNotFoundError(f"找不到 monitor.html：{html_path}")

def launch_window(port=5050):
    html = get_html()

    try:
        import webview
        print("  [GUI] pywebview")
        time.sleep(1.2)
        webview.create_window("TCN · BPM Monitor", html=html,
                              width=780, height=840,
                              min_size=(600, 600), resizable=True)
        webview.start()
        return
    except ImportError:
        pass

    print("\n請安裝 GUI 套件：\n  pip install pywebview\n  或\n  pip install PyQt5 PyQtWebEngine")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass


# ────────────────────────────────────────────────────────
# 4. 入口
# ────────────────────────────────────────────────────────

def main():
    PORT = 5050
    threading.Thread(target=lambda: run_server(PORT), daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()
    print(f"=== Flask server @ http://127.0.0.1:{PORT} ===")
    launch_window(PORT)

if __name__ == "__main__":
    main()