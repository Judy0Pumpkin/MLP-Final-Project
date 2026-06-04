# ── 音訊輸入 ──────────────────────────────────────────────
SAMPLE_RATE   = 22050
CHANNELS      = 1       # 單聲道
CHUNK_SIZE    = 512     # 每次 callback 收到的樣本數

# ── Sliding Window ────────────────────────────────────────
WINDOW_SECONDS = 6                        # 每次送進模型的音訊長度（秒）
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS      # = 132300 samples
HOP_SECONDS    = 0.5                      # 每隔多久滑動一次（秒），控制輸出頻率
HOP_SAMPLES    = int(SAMPLE_RATE * HOP_SECONDS)    # = 11025 samples

# ── Mel Spectrogram ───────────────────────────────────────
N_MELS      = 128    # Mel filterbank 數量
HOP_LENGTH  = 512    # STFT hop size
N_FFT       = 2048   # FFT 視窗大小
F_MIN       = 27.5   # 最低頻率，對應鋼琴最低音 A0
F_MAX       = 8000   # 最高頻率

FIXED_FRAMES = WINDOW_SAMPLES // HOP_LENGTH   # 每個 window 的 Mel spectrogram 時間軸長度（258 frames @ 6s）

# ── 模型輸出 ──────────────────────────────────────────────
TEMPO_MIN = 30    # BPM 下限（慢板）
TEMPO_MAX = 250   # BPM 上限（極快）

# ── 裝置 ──────────────────────────────────────────────────
DEVICE = "cpu"    # "cuda" if torch.cuda.is_available() else "cpu"