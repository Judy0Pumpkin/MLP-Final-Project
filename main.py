import numpy as np
import sounddevice as sd
from collections import deque
import threading
import time

# ── 參數設定 ──────────────────────────────────────────────
SAMPLE_RATE = 22050       # Hz，librosa 預設 sr，之後 Mel Spectrogram 統一用這個
CHANNELS = 1              # 單聲道
CHUNK_SIZE = 512          # 每次 callback 收到的樣本數（越小延遲越低，但 CPU 負擔越重）
WINDOW_SECONDS = 3        # sliding window 長度（秒）
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS  # window 總樣本數

# ── Sliding Window Buffer ─────────────────────────────────
audio_buffer = deque(maxlen=WINDOW_SIZE) # deque 設定 maxlen，超過就自動丟掉最舊的資料
buffer_lock = threading.Lock()  # 多執行緒安全


# ── Callback 函式（每次收到 CHUNK_SIZE 個樣本就觸發）────────
def audio_callback(indata, frames, time_info, status):
    """
    sounddevice 的 callback，在獨立的執行緒中執行。
    indata shape: (CHUNK_SIZE, CHANNELS) → 取第一個 channel 變成 1D
    """
    if status:
        print(f"[警告] {status}")

    # 取單聲道，flatten 成 1D array
    chunk = indata[:, 0].copy()
    print(indata.shape, chunk.shape)  # debug 用，確認 shape 正確
    # 推進 buffer
    with buffer_lock:
        audio_buffer.extend(chunk)

    # 簡單檢查：印出這個 chunk 的基本資訊
    print(
        f"chunk shape: {chunk.shape} | "
        f"max: {chunk.max():.4f} | "
        f"buffer 目前長度: {len(audio_buffer)}/{WINDOW_SIZE}"
    )


# ── 取得當前 Window（給之後 Mel Spectrogram 用）──────────────
def get_current_window():
    """
    從 buffer 取出目前所有資料，轉成 numpy array。
    buffer 未滿時回傳 None，避免送進模型的資料太短。
    """
    with buffer_lock:
        if len(audio_buffer) < WINDOW_SIZE:
            return None
        return np.array(audio_buffer, dtype=np.float32)


# ── 主程式 ────────────────────────────────────────────────
def main():
    print("=== Real-Time Audio Input ===")
    print(f"Sample rate : {SAMPLE_RATE} Hz")
    print(f"Chunk size  : {CHUNK_SIZE} samples ({CHUNK_SIZE/SAMPLE_RATE*1000:.1f} ms)")
    print(f"Window size : {WINDOW_SECONDS} 秒 ({WINDOW_SIZE} samples)")

    # 列出可用的音訊裝置，方便確認麥克風
    print("── 可用音訊裝置 ──")
    print(sd.query_devices())
    print()

    # 開啟串流
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=CHUNK_SIZE,
        dtype="float32",
        callback=audio_callback,
    ):
        try:
            while True:
                time.sleep(1)

                # 每秒嘗試取一次 window，之後這裡會接 Mel Spectrogram
                window = get_current_window()
                if window is not None:
                    print(f"\n[Window ready] shape: {window.shape}, "
                          f"mean amplitude: {np.abs(window).mean():.5f}\n")
                else:
                    print(f"[Buffer 填充中...] {len(audio_buffer)}/{WINDOW_SIZE} samples\n")

        except KeyboardInterrupt:
            print("\n停止。")


if __name__ == "__main__":
    main()