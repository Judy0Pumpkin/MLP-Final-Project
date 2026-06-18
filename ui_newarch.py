"""
ui_newarch.py
BeatTCN v2 即時推理介面（Flask + 瀏覽器）

執行：python ui_newarch.py
會自動在 http://127.0.0.1:5051 開啟瀏覽器視窗
"""

import os, threading, time
import numpy as np
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import torch
import sounddevice as sd

from constants import SAMPLE_RATE, HOP_LENGTH, N_MELS, FIXED_FRAMES, WINDOW_SECONDS, HOP_SAMPLES
from inference_new_arch import audio_to_mel, run_beat_tcn, smooth_bpm

PORT        = 5051
CHECKPOINT  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "checkpoints", "best_model_new_arch_f07.pt")
SILENCE_RMS = 0.003   # RMS 低於此值視為無音訊，跳過推理（調高 = 更嚴格）

# ── Flask ─────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

state = {
    "is_live": False, "loop": 0, 
    "bpm": None, "beat_count": 0, "beat_times": [],
    "downbeat_times": [], "tempo_bpm": None,
    "amplitude_max": 0.0, "amplitude_mean": 0.0,
    "waveform": [], "beat_act": [],
    "buffer_fill": 0.0, "silent": False, "streaming": False,
    "model": "BeatTCN v2",
    "window_seconds": WINDOW_SECONDS,
    "sample_rate": SAMPLE_RATE,
}
state_lock = threading.Lock()


@app.route("/")
def index():
    """讀取 monitor.html，把 API URL 換成本機 port。"""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor.html")
    with open(html_path, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("http://127.0.0.1:5050", f"http://127.0.0.1:{PORT}")
    return Response(html, mimetype="text/html")


@app.route("/state")
def get_state():
    with state_lock:
        return jsonify(dict(state))


@app.route("/toggle", methods=["POST"])
def toggle_streaming():
    with state_lock:
        state["streaming"] = not state["streaming"]
        current = state["streaming"]
    print(f"  streaming {'▶ 開啟' if current else '■ 暫停'}")
    return jsonify({"streaming": current})


@app.route("/shutdown", methods=["POST"])
def shutdown():
    def _exit():
        time.sleep(0.3)
        os._exit(0)
    threading.Thread(target=_exit, daemon=True).start()
    return jsonify({"ok": True})


# ── 推理迴圈 ──────────────────────────────────────────────────

def _downsample(arr, n=512):
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return arr[idx].tolist()


_WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
_ring_buffer    = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)
_ring_lock      = threading.Lock()

def _start_audio_thread():
    """獨立 thread：持續錄音，不停把新 chunk 滑入 ring buffer。"""
    _cbuf = []

    def cb(indata, _frames, _time, _status):
        _cbuf.extend(indata[:, 0])
        while len(_cbuf) >= HOP_SAMPLES:
            chunk = np.array(_cbuf[:HOP_SAMPLES], dtype=np.float32)
            del _cbuf[:HOP_SAMPLES]
            with _ring_lock:
                _ring_buffer[:-HOP_SAMPLES] = _ring_buffer[HOP_SAMPLES:]
                _ring_buffer[-HOP_SAMPLES:] = chunk

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=512, dtype="float32", callback=cb):
        while True:
            time.sleep(1)   # 只是讓 stream 保持開著


def inference_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 載入模型
    model = None
    if os.path.exists(CHECKPOINT):
        try:
            from model_new_arch import BeatTCN
            m    = BeatTCN().to(device)
            ckpt = torch.load(CHECKPOINT, map_location=device)
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            model = m
            print(f"[模型] loaded  epoch={ckpt.get('epoch','?')}  "
                  f"val_F={ckpt.get('val_f', 0):.4f}")
        except Exception as e:
            print(f"[警告] 模型載入失敗：{e}  → 改用 librosa")
    else:
        print(f"[提示] 找不到 {CHECKPOINT}，使用 librosa")

    with state_lock:
        state["is_live"] = True
        state["model"]   = "BeatTCN v2" if model else "librosa"

    TOTAL_CHUNKS  = _WINDOW_SAMPLES // HOP_SAMPLES
    chunks_filled = 0

    # 啟動錄音 thread，持續更新 _ring_buffer
    threading.Thread(target=_start_audio_thread, daemon=True).start()

    loop = 0
    while True:
        with state_lock:
            streaming = state["streaming"]
        if not streaming:
            time.sleep(0.2)
            continue

        time.sleep(1.0)   # 每 1 秒跑一次 inference

        loop += 1
        chunks_filled = min(chunks_filled + 2, TOTAL_CHUNKS)  # 1s = 2 chunks

        with _ring_lock:
            y = _ring_buffer.copy()

        with state_lock:
            state["loop"]        = loop
            state["buffer_fill"] = chunks_filled / TOTAL_CHUNKS
            state["waveform"]    = _downsample(y, 512)
            state["silent"]      = False

        # 只看最近 1 秒的 RMS，避免 buffer 剛啟動時大量 zeros 觸發 silence
        recent = y[-SAMPLE_RATE:]
        rms = float(np.sqrt(np.mean(recent ** 2)))
        with state_lock:
            state["amplitude_max"]  = round(float(np.abs(y).max()), 4)
            state["amplitude_mean"] = round(rms, 5)
        if rms < SILENCE_RMS:
            with state_lock:
                state["silent"] = True
            continue

        mel = audio_to_mel(y)

        if model:
            beat_times, bpm, beat_act, downbeat_times, tempo_bpm = \
                run_beat_tcn(model, mel, device)
        else:
            import librosa
            tempo, frames = librosa.beat.beat_track(
                y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
            beat_times     = librosa.frames_to_time(frames, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
            bpm            = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
            beat_act       = np.zeros(FIXED_FRAMES, dtype=np.float32)
            downbeat_times = np.array([])
            tempo_bpm      = bpm

        bpm = smooth_bpm(bpm)
        print(f"  BPM={bpm:.1f}  beats={len(beat_times)}  downbeats={len(downbeat_times)}")

        with state_lock:
            state["bpm"]            = round(bpm, 1)
            state["beat_count"]     = int(len(beat_times))
            state["beat_times"]     = [round(float(t), 3) for t in beat_times]
            state["downbeat_times"] = [round(float(t), 3) for t in downbeat_times]
            state["tempo_bpm"]      = round(float(tempo_bpm), 1) if tempo_bpm else None
            state["amplitude_max"]  = round(float(y.max()), 4)
            state["amplitude_mean"] = round(float(np.abs(y).mean()), 5)
            state["beat_act"]       = _downsample(beat_act, 256)
            state["silent"]         = False


# ── 入口 ──────────────────────────────────────────────────────

def main():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    threading.Thread(target=inference_loop, daemon=True).start()

    # 等 Flask 啟動後開瀏覽器
    def _open_browser():
        time.sleep(1.2)
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{PORT}")
    threading.Thread(target=_open_browser, daemon=True).start()

    print(f"=== BeatTCN v2 Monitor  →  http://127.0.0.1:{PORT} ===")
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
