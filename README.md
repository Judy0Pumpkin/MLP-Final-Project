# BeatTCN v2 — Real-Time Beat Detection

使用 Temporal Convolutional Network (TCN) 對麥克風輸入做即時拍點（beat）與節拍（downbeat）偵測，並以 Flask 網頁即時顯示波形、BPM 與 beat activation。

---

## 安裝環境

> Python 3.9+ 建議使用虛擬環境

```bash
pip install torch torchaudio
pip install librosa sounddevice
pip install flask flask-cors
pip install numpy scipy
```

---

## 資料集
可以從期末報告繳交資料夾中dataset.zip中下載, 解壓在專案根目錄即可，`dataset.py` 會自動掃描。

使用 **Ballroom Dataset**。解壓後結構如下：

```
dataset/
├── Waltz/
│   ├── song1.wav
│   └── song1.beats
├── Quickstep/
│   └── ...
```


---

## 檔案說明與執行方式

| 檔案 | 功能 | 執行 |
|------|------|------|
| `constants.py` | 全域參數（取樣率、Mel 設定、window 長度等），不直接執行 | — |
| `dataset.py` | Ballroom Dataset loader，含 Mel 計算與資料增強（slow-stretch）| — |
| `model.py` | BeatTCN 模型定義（CausalConv1d + TCNBlock） | — |
| `inference.py` | `audio_to_mel`、`run_beat_tcn`、`smooth_bpm` 等推理工具函式 | — |
| `train.py` | 訓練腳本，產生 checkpoint 存到 `checkpoints/` | `python train.py` |
| `main.py` | 測試麥克風輸入是否正常、確認 sliding window buffer 運作 | `python main.py` |
| `UI.py` | Flask 即時推理介面，自動開瀏覽器顯示 monitor | `python UI.py` |
| `monitor.html` | 網頁前端（由 `UI.py` 自動讀取並回傳，不需手動開啟） | — |

---

## 快速開始

### 1. 訓練模型

```bash
python train.py
```

訓練完成後會在 `checkpoints/` 產生 `best_model_new_arch_f07.pt`。

### 2. 測試麥克風輸入

```bash
python main.py
```

確認麥克風可以收音、audio buffer 正常填充後 Ctrl+C 結束。

### 3. 啟動即時 Beat Monitor

```bash
python UI.py
```

會自動在瀏覽器開啟 `http://127.0.0.1:5051`。

- 按右上角 **▶ START** 開始推理，畫面即時顯示 BPM、波形、beat activation
- 按右上角 **■ STOP** 暫停

若找不到 checkpoint，會自動退回使用 librosa 做 BPM 估計。

---

## Checkpoints

| 檔案 | 說明 |
|------|------|
| `best_model.pt` | 初版模型 |
| `best_model_new_arch.pt` | 新架構（含 downbeat head） |
| `best_model_new_arch_f07.pt` | 新架構，val F-measure ≈ 0.7（UI 預設載入） |

---

## 參數調整

主要超參數在 `constants.py`（音訊設定）與 `train.py` 頂部（訓練設定）：

- `WINDOW_SECONDS`：每次送進模型的音訊長度
- `BEAT_POS_WEIGHT`（`train.py`）：beat frame 的 loss 權重
