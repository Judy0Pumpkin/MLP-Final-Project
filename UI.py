"""
ui.py

啟動方式：python ui.py

負責：
    1. Flask state server（/state、/toggle、/shutdown）
    2. 背景執行緒跑推理迴圈（呼叫 inference.py）
    3. 讀取 monitor.html 啟動 PyQt5 / pywebview 視窗

inference.py、model.py、constants.py 不需要修改。
"""

import sys, os, threading, time
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
import torch

from constants import SAMPLE_RATE, N_MELS, HOP_LENGTH, FIXED_FRAMES, WINDOW_SECONDS
from inference import  audio_to_mel, run_beat_tcn, smooth_bpm
import sounddevice as sd

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
    "model": "–", "streaming": False,
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

CHECKPOINT = r"C:\Program Files (x86)\pythonlearning\Machine_learning_Project\checkpoints\best_model.pt"

def downsample(arr, target=512):
    idxs = np.linspace(0, len(arr) - 1, target).astype(int)
    return arr[idxs].tolist()



def get_audio_with_update():
    """邊錄音邊即時更新 state waveform"""
    total_samples = int(SAMPLE_RATE * WINDOW_SECONDS)
    collected = []

    def callback(indata, frames, time_info, status):
        collected.append(indata[:, 0].copy())
        filled = sum(len(c) for c in collected)
        current = np.concatenate(collected)
        idxs = np.linspace(0, len(current) - 1, 512).astype(int)
        with state_lock:
            state["buffer_fill"] = min(1.0, filled / total_samples)
            state["waveform"]    = current[idxs].tolist()

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

    use_tcn, tcn_model = False, None
    if os.path.exists(CHECKPOINT):
        try:
            from model import BeatTCN
            m = BeatTCN().to(device)
            ckpt = torch.load(CHECKPOINT, map_location=device)
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            tcn_model = m
            use_tcn = True
            print(f"  [TCN] epoch={ckpt.get('epoch','?')}  val_F={ckpt.get('val_f', float('nan')):.4f}")
            print(f"=== BeatTCN (device: {device}) ===")
        except Exception as e:
            print(f"  [警告] 載入失敗：{e} → 改用 librosa")
    else:
        print("  [提示] 找不到 checkpoint，使用 librosa baseline")

    with state_lock:
        state["is_live"] = True
        state["model"]   = "BeatTCN" if use_tcn else "librosa"

    prev_bpm, loop = None, 0

    while True:
        with state_lock:
            streaming = state["streaming"]
        if not streaming:
            time.sleep(0.2)
            continue

        loop += 1
        print(f"── Round {loop} [{state['model']}] ──")
        with state_lock:
            state["loop"]        = loop
            state["buffer_fill"] = 0.0
            state["silent"]      = False

        y = get_audio_with_update()   # ← 改這裡，同時更新波形

        if np.abs(y).mean() < 1e-4:
            print("  [靜音，跳過]")
            with state_lock:
                state["silent"] = True
            continue

        mel_tensor = audio_to_mel(y)

        if use_tcn:
            beat_times, bpm, beat_act = run_beat_tcn(tcn_model, mel_tensor, device)
        else:
            import librosa
            tempo, beat_frames = librosa.beat.beat_track(
                y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
            beat_times = librosa.frames_to_time(
                beat_frames, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
            bpm = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
            beat_act = None
        bpm = smooth_bpm(bpm, prev_bpm)
        print(f"  BPM={bpm:.1f}  beats={len(beat_times)}")

        with state_lock:
            state["prev_bpm"]       = prev_bpm
            state["bpm"]            = round(bpm, 1)
            state["beat_count"]     = int(len(beat_times))
            state["beat_times"]     = [round(float(t), 3) for t in beat_times]
            state["amplitude_max"]  = round(float(y.max()), 4)
            state["amplitude_mean"] = round(float(np.abs(y).mean()), 5)
            state["mel_shape"]      = list(mel_tensor.shape)
            state["silent"]         = False
            state["beat_act"]       = downsample(beat_act, 256) if beat_act is not None else []

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