# WEEK01 Outline — DLinear & NLinear

## One-sentence focus

**When does Decomposition beat Normalization?** — so sánh DLinear vs NLinear qua đặc trưng dữ liệu (trend / seasonal strength, non-stationarity), không chỉ "model nào MSE thấp hơn trên 1 dataset".

## Problem

- Paper LTSF-Linear (Zeng et al.) cho thấy baseline **linear** (Linear, NLinear, DLinear) mạnh và đơn giản hơn Transformer trên nhiều benchmark LTSF.
- Toner & Darlow: trên benchmark "sạch", DLinear ≈ Linear (cùng model class) — nhưng **chưa trả lời khi nào** Decomposition (DLinear) thắng Normalization (NLinear).
- Hầu hết so sánh chỉ báo "winner theo dataset × horizon". Thiếu **prescriptive guideline**: đặc trưng chuỗi nào dự đoán model nào thắng.
- Nhóm giữ **VIC** — mã cổ phiếu **Vingroup** (HOSE), chuỗi giá đóng cửa hàng ngày (`close`, dùng `log`) — như 1 điểm miền tài chính (nhiễu / không dừng) trong ma trận; đồng thời thêm vài benchmark LTSF chuẩn để có **variance** về trend / season / stationarity.

## Research questions

| ID | Câu hỏi | Deliverable gắn với RQ |
| -- | ------- | ---------------------- |
| **RQ1** | Đặc trưng dữ liệu nào (trend strength, seasonal strength, non-stationarity) dự đoán DLinear thắng hay NLinear thắng? | Scatter / heatmap: data-char vs performance gap; guideline 1 đoạn |
| **RQ2** | Độ nhạy lookback `L` của hai model khác nhau thế nào, và pattern đó có phụ thuộc data-char không? | Ablation lookback; so sánh với profile STL của từng dataset |

**Claim mục tiêu (falsifiable):** *DLinear thắng khi seasonal / trend strength cao; NLinear thắng khi level / distribution shift lớn (non-stationary hơn).* Sai cũng là kết quả — miễn đo đúng protocol.

## Output type

**Experiment + short research note** (không demo app):

1. **Tự implement** Linear, NLinear, DLinear (scratch → `src/models.py`); channel-independent + shared weights.
2. **Đo đặc trưng dữ liệu** (STL via `statsmodels`) trên mọi dataset trước khi train.
3. **Benchmark đa dataset** (ETTh1, ETTh2, Weather, Exchange-Rate, Electricity, **VIC = Vingroup daily close**) × nhiều lookback / horizon.
4. **Ablation** MA kernel size của DLinear + lookback sensitivity (RQ2).
5. **Research note:** bảng overall + scatter/heatmap + guideline; hạn chế + hướng tiếp.

*Không replicate 1:1 paper. **Transfer miền VIC** = đánh giá lại trên chuỗi cổ phiếu Vingroup (ngoài benchmark LTSF chuẩn), không phải fine-tune weights — cộng giải thích qua data characteristics.*

## Hướng triển khai theo tuần

| Week | Mục tiêu | Deliverable chính (Working_Files) |
| ---- | -------- | --------------------------------- |
| **W1** | Nền tảng + outline + papers | Outline này; `WEEK01_Reading_Note/`; Paper Tracker ≥5 |
| **W2** | Research-gap note + scratch models + data-char | `WEEK02 - Research Gap Analysis/`; `WEEK02_Scratch_Models.ipynb`; `src/models.py`; `src/data_chars.py` |
| **W3** | Multi-dataset baseline | `src/dataset.py`; `WEEK03_Baseline_Protocol.md`; `WEEK03_Benchmark.ipynb` + bảng overall |
| **W4** | Ablation + heatmap + wrap | `src/eval.py`; `WEEK04_Ablation_Analysis.ipynb`; grow `WEEK02 - Research Gap Analysis/` + `Research_Notebook.md` |

**W1 — nền tảng:** Đọc LTSF-Linear (+ slide DLinear–NLinear); paper liên quan (Autoformer, RevIN, Toner & Darlow, STL / FPP3). Cơ chế → `WEEK01_Reading_Note/`. Tracker ≥5. Chốt outline này. (Research-gap LaTeX chuyển sang W2 theo convention tên tuần của Topic Team.)

**W2 — research gap + implement + đo data-char:** Living paper `WEEK02 - Research Gap Analysis/` (Introduction + problem + hypothesis; English LaTeX). Linear (`L → T`), NLinear (re-center bằng last value), DLinear (MA + dual heads). Channel-independent + shared weights → `src/models.py`. Viết `src/data_chars.py`: STL → trend / seasonal / resid strength (công thức kiểu Hyndman / FPP3). Chạy 1 lần trên tất cả dataset → bảng profile (file CSV hoặc markdown).

**W3 — benchmark:** Dataset list chốt bên dưới; sliding window; chronological split (70/15/15 hoặc protocol paper cho ETTh*). Metrics MSE/MAE. Train **Linear + NLinear + DLinear** × lookbacks × horizons. Naive baselines (last-value / seasonal-naive nếu áp dụng). Heatmap trọng số Linear (tuỳ chọn). *Mục đọc lại:* có bảng overall để W4 gắn với data-char.

**W4 — ablation + contribution + wrap:** (1) Sweep MA kernel size DLinear. (2) Lookback sensitivity curves (RQ2). (3) Scatter/heatmap: trục X = data-char, trục Y = gap `MSE(DLinear) − MSE(NLinear)` (hoặc tương đương). (4) Viết guideline 1 đoạn vào living paper `WEEK02 - Research Gap Analysis/`. Kiểm tra claim RQ1 (đúng/sai đều OK). Seed hướng tiếp (RevIN chồng NLinear, regime VIC sâu hơn, …).

## Quyết định đã chốt

- [x] **Góc nghiên cứu:** Decomposition vs Normalization qua **data characteristics**, không chỉ leaderboard 1 dataset.
- [x] **Datasets:** ETTh1, ETTh2, Weather, Exchange-Rate, Electricity, plus **VIC** = giá đóng cửa hàng ngày cổ phiếu **Vingroup (HOSE)**, biến `close` → `log(close)`. VIC = điểm transfer / stress miền tài chính; không phải benchmark duy nhất.
- [x] **Metrics:** MSE / MAE (log-price + returns cho VIC theo protocol W3).
- [x] **Lookback / horizon (mặc định):** lookback ∈ `{96, 192, 336, 720}` cho LTSF chuẩn; VIC (Vingroup) riêng lookback ∈ `{5, 30, 120, 480}`, `pred_len = 5` (protocol finance ngắn). Horizon LTSF ∈ `{96, 192, 336, 720}` (có thể cắt bớt nếu thiếu thời gian — ghi rõ trong protocol).
- [x] **Data-char (W2):** STL → `F_T`, `F_S` (trend / seasonal strength); ghi thêm proxy non-stationarity (vd. ADF p-value hoặc train–test mean/std shift) nếu kịp.
- [x] **Ablation (W4):** MA kernel size DLinear; lookback sensitivity (RQ2).
- [x] **Hypothesis RQ1 (falsifiable):** trên các dataset có **seasonal / trend strength cao**, `MSE(DLinear) < MSE(NLinear)` (cùng protocol); trên dataset **level-shift / non-stationary mạnh** (vd. Exchange, VIC high-vol), NLinear tốt hơn hoặc gap hẹp. Sai cũng là kết quả.
- [x] **Paper Tracker:** local `Paper_Tracker/*.xlsx` (upload Drive/Sheet nếu TA cần link).

## Folder map

```
DLinear_NLinear/
├── EXECUTION_PLAN.md            ← plan chi tiết cho agent / người follow
├── Paper_Tracker/
│   ├── Paper Tracker - ….xlsx   ← sheet ≥5 papers
│   └── papers/                  ← PDF local (tham chiếu)
└── Working_Files/
    ├── WEEK01_Outline.md        ← this file (vấn đề + kế hoạch)
    ├── WEEK01_Reading_Note/     ← Cornell / cơ chế (main.pdf)
    ├── WEEK02 - Research Gap Analysis/  ← living LaTeX paper (intro/problem/hypothesis → +results later)
    ├── WEEK02_* / WEEK03_* / WEEK04_*
    ├── Research_Notebook.md
    └── src/
        ├── models.py
        ├── data_chars.py        ← STL / strength metrics
        ├── dataset.py
        └── eval.py
```

**Phân vai file:** Outline = *làm gì / vì sao*; Reading Note = *hiểu cơ chế*; Tracker xlsx = *từng paper*; `WEEK02 - Research Gap Analysis/` = *living paper gửi TA (gap slice → full results)*; `EXECUTION_PLAN.md` = *làm từng bước thế nào*.

## Tiến độ (theo hướng dẫn Topic Team)

- Leader cập nhật **Google Form ~mỗi 3 ngày** (tiến độ / kết quả + path / kế hoạch / khó khăn / hỏi TA).
- **Cuối mỗi tuần:** một phiên bản kết quả trong Working_Files để TA đối chiếu với mục tiêu trên.
- Chi tiết checklist từng bước: xem [`../EXECUTION_PLAN.md`](../EXECUTION_PLAN.md).
