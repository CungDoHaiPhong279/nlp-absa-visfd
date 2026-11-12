# Phân tích cảm xúc theo khía cạnh trên UIT-ViSFD – Implementation Plan (bản 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng pipeline NLP chạy lại được, từ dữ liệu thô UIT-ViSFD đến bảng so sánh, khảo sát có kiểm soát, phân tích lỗi và demo, cho hai bài toán: phát hiện khía cạnh và cảm xúc theo khía cạnh trong bình luận điện thoại tiếng Việt.

**Architecture:** Gói Python `absa` (thư mục `src/absa/`), mỗi module làm một bước: đọc nhãn → tiền xử lý (có cache) → đánh giá → mô hình (TF-IDF, transformer, Bi-LSTM) → bảng → phân tích lỗi → demo. Mọi mô hình nhận `list[str]` và trả về **ma trận nhãn** `(n, 10)`, mã 0=None, 1=Positive, 2=Neutral, 3=Negative; nhờ đó một hàm `evaluate` dùng chung cho tất cả. Thí nghiệm khai báo bằng YAML có kế thừa (`base:`). Mỗi lần chạy ghi một JSON vào `results/runs/`; `scripts/make_tables.py` gom thành bảng.

**Tech Stack:** Python 3.11 hoặc 3.12 (bản chính thức python.org hoặc Colab, **không** dùng Python của MSYS2), pandas, numpy, scikit-learn, underthesea, py_vncorenlp (cần Java), PyTorch, transformers, sentencepiece, gradio, matplotlib, PyYAML, tabulate, jupyter/nbconvert, pytest.

**Spec:** `docs/spec/phac_thao_du_an_ABSA_UIT-ViSFD.md` (bản phác thảo dự án). Yêu cầu môn học: file `Huong_dan_bai_tap_lon_Xu_ly_ngon_ngu_tu_nhien.docx` (rubric 100 điểm). Plan này thay thế bản `2026-11-12-absa-visfd.md`.

---

## Quyết định so với spec và bản plan cũ

Mỗi dòng dưới đây là một điểm plan **cố ý** khác spec hoặc plan cũ. Task 1 chép nguyên phần này vào `docs/decisions.md`; các quyết định phát sinh khi chạy (C tốt nhất, `max_length`, cấu hình chốt) được ghi tiếp vào file đó để báo cáo truy vết được.

- **D1 – Baseline TF-IDF theo đúng spec 3.2:** chế độ mặc định `mode: split`. Bước 1: 10 bộ phân loại nhị phân (có/không khía cạnh). Bước 2: mỗi khía cạnh một bộ 3 lớp, chỉ train trên các bình luận có khía cạnh đó. Chế độ `joint` (4 lớp cho mỗi khía cạnh) là một nhánh của khảo sát E6.
- **D2 – Số sao (E5) là bước tiền xử lý** (`preprocess.use_star`), không phải tham số mô hình, vì mô hình chỉ nhận văn bản. Tiền tố `"[{n_star} sao] "` được thêm **sau** khi tách từ.
- **D3 – Chạy được trên Windows:** mọi script viết bằng Python (`scripts/*.py`), không dùng `.sh`, `unzip` hay vòng `for` của bash.
- **D4 – Không chạy trùng:** YAML kế thừa cấu hình gốc qua `base:`. Cấu hình gốc khai `groups: [E3, E4, ...]` nên đóng vai một nhánh của mọi khảo sát; file ablation chỉ chứa nhánh thay đổi.
- **D5 – Thêm chỉ số `polarity_macro_f1_detected`:** macro-F1 3 lớp cảm xúc, chỉ tính trên các cặp (bình luận, khía cạnh) mà cả gold lẫn dự đoán đều có khía cạnh. Chỉ số này để có cơ hội so với cách tính của bài báo. Kết luận về cách tính F1 của bài báo ghi ở `docs/metric_notes.md` (Task 2).
- **D6 – Confusion matrix 4×4** (gồm cả None) thay cho 3×3 trong spec 4.2, vì nó thể hiện cả lỗi bỏ sót và lỗi thừa khía cạnh.
- **D7 – Cache kết quả trung gian** ở `data/processed/`. Hướng dẫn môn học yêu cầu điều này, và tách từ VnCoreNLP chậm.
- **D8 – Bi-LSTM dùng âm tiết** (không tách từ), vì vector fastText `cc.vi.300` chủ yếu theo âm tiết. Tỉ lệ phủ từ vựng đo được ghi vào `docs/decisions.md`. Báo cáo gọi đây là "tái hiện xấp xỉ", không gọi là "tái lập".
- **D9 – `max_length` chọn theo EDA:** lấy p99 độ dài token con (subword) trên Train, làm tròn lên bội số của 32, không vượt 256.
- **D10 – Dò tham số trên Dev (E0):** `C` ∈ {0.25, 0.5, 1, 2, 4} cho TF-IDF LR/SVM; ngưỡng phát hiện khía cạnh ∈ {0.30, 0.35, …, 0.70} cho transformer ở chế độ split.
- **D11 – Cấu trúc repo:** gói `src/absa/` và 3 notebook (`01_eda`, `02_colab_train`, `03_error_analysis`) thay cho 4 notebook và `src/*.py` rời ở spec mục 8. Lý do: code dùng chung, test được.
- **D12 – `--eval-test` chỉ nhận cấu hình nằm trong `configs/final/`** và không dùng chung với `--limit`; code chặn cứng, không chỉ dựa vào quy ước.

## Global Constraints

- Thứ tự khía cạnh cố định ở mọi nơi: `("SCREEN", "CAMERA", "FEATURES", "BATTERY", "PERFORMANCE", "STORAGE", "DESIGN", "PRICE", "GENERAL", "SER&ACC")`.
- Mã cảm xúc: `0=None, 1=Positive, 2=Neutral, 3=Negative`. `POLARITIES = ("Positive", "Neutral", "Negative")`.
- `{OTHERS}` không phải khía cạnh: bỏ khi tạo ma trận, chỉ lưu thành cờ `has_others`.
- Split gốc: Train 7.786 · Dev 1.112 · Test 2.224. Không trộn, không chia lại.
- Nguồn dữ liệu: `https://raw.githubusercontent.com/LuongPhan/UIT-ViSFD/main/UIT-ViSFD.zip`, SHA-256 `d3c068814d23b11473c6bf774a81ffe1916b304501ac10e0c99fcd15f11a16a8`. Trong zip có `Train.csv`, `Dev.csv`, `Test.csv` ở thư mục gốc; cột `index, comment, n_star, date_time, label`.
- Chỉ chọn tham số trên **Dev**. Chỉ đánh giá **Test** bằng `--eval-test` với cấu hình trong `configs/final/` (D12).
- Vectorizer, từ vựng, embedding và trọng số lớp chỉ fit trên Train.
- Seed mặc định 42; thí nghiệm nhiều seed dùng `[42, 43, 44]`.
- `data/`, `models/`, `results/predictions/` không commit (`results/runs/*.json` có commit, xem D13). Tải lại dữ liệu bằng `python scripts/download_data.py`.
- Model id: `vinai/phobert-base-v2` (đầu vào phải tách từ bằng VnCoreNLP), `uitnlp/visobert` (đầu vào không tách từ). `max_length` ≤ 256.
- Chỉ số chính: `aspect_macro_f1` và `sentiment_macro_f1`. Accuracy không bao giờ là chỉ số chính.
- Bảng xuất ra dùng thang %, 2 chữ số thập phân. Hàm tính toán bên trong không làm tròn.
- Số trong báo cáo phải truy được về một file trong `results/runs/`. Không điền tay.
- Commit message **không** chứa dòng `Co-Authored-By` hay bất kỳ dòng ghi công AI nào.
- Sau khi cài: `pip freeze > requirements.lock.txt` (máy local) và `requirements.lock.colab.txt` (Colab).

## Phân công

Theo kế hoạch 10 tuần ở spec mục 7. Người phụ trách tự commit phần của mình, vì rubric điều chỉnh điểm theo đóng góp có minh chứng. Người còn lại review trước khi merge.

| Task | Nội dung | Phụ trách | Tuần |
|---|---|---|---|
| 1 | Khung dự án, tải dữ liệu, đọc nhãn | A (B dựng môi trường) | 1 |
| 2 | Hàm đánh giá, ghi chú cách tính của bài báo | B | 2 |
| 3 | Tiền xử lý, cache | A | 3 |
| 4 | EDA | A | 2–3 |
| 5 | Cấu hình và bộ chạy thí nghiệm | B | 3 |
| 6 | Baseline TF-IDF, E0–E3, E6 (TF-IDF) | A | 4–5 |
| 7 | Bảng so sánh | A | 4 |
| 8 | Lõi transformer (head, loss, decode) | B | 3–4 |
| 9 | Huấn luyện transformer, chạy trên Colab, E3–E6 | B (A chạy E3, E5) | 4–6 |
| 10 | Bi-LSTM + fastText | A | 5–6 |
| 11 | Chốt cấu hình, 3 seed, test, phân tích lỗi | Cả hai | 6–7 |
| 12 | Demo Gradio | A | 9 |
| 13 | README, `run_all.py`, kiểm tra tái lập | B | 9 |
| 14 | Báo cáo PDF, slide | Cả hai | 8–10 |

## Cấu trúc file

```
nlp-absa-visfd/
├── pyproject.toml  requirements.txt  requirements.lock*.txt  .gitignore  README.md
├── docs/  spec/…  decisions.md  metric_notes.md  superpowers/plans/…
├── configs/
│   ├── tfidf_lr_base.yaml  tfidf_svm.yaml  phobert_joint.yaml  visobert_joint.yaml  bilstm_fasttext.yaml
│   ├── ablation/  E1_ngram_1.yaml E1_ngram_13.yaml E1_char_25.yaml E2_word.yaml
│   │              E3_tfidf_strip_emoji.yaml E6_tfidf_joint.yaml
│   │              E3_phobert_strip_emoji.yaml E4_phobert_weight.yaml E5_phobert_star.yaml E6_phobert_split.yaml
│   ├── smoke/  tfidf_smoke.yaml  tiny_transformer.yaml
│   └── final/  (tạo ở Task 11)
├── scripts/  download_data.py  run.py  make_tables.py  prepare_fasttext.py  run_all.py
├── src/absa/
│   ├── __init__.py
│   ├── data.py         # hằng số, tách nhãn, ma trận, load_split
│   ├── evaluate.py     # evaluate, confusion_by_aspect
│   ├── preprocess.py   # normalize, segment, preprocess_df (cache)
│   ├── eda.py          # bảng thống kê
│   ├── config.py       # load_config (kế thừa), apply_overrides
│   ├── registry.py     # Model Protocol, REGISTRY, build_model
│   ├── experiment.py   # run_experiment, assert_final_config
│   ├── baseline.py     # TfidfAbsa
│   ├── report.py       # collect_runs, các bảng
│   ├── tiny.py         # encoder + tokenizer tí hon, không cần mạng
│   ├── transformer.py  # AbsaHead, absa_loss, decode, class weights, best_threshold, TransformerAbsa
│   ├── training.py     # set_seed, fit_loop (dùng chung cho transformer và Bi-LSTM)
│   ├── bilstm.py       # build_vocab, BiLstmAbsa
│   ├── errors.py       # collect_errors, sample_errors, plot_confusions
│   └── inference.py    # Predictor, render_html, summarize_predictions
├── demo/app.py
├── notebooks/  01_eda.ipynb  02_colab_train.ipynb  03_error_analysis.ipynb
├── tests/  conftest.py  test_*.py
├── data/ (gitignore)  raw/  processed/
├── models/ (gitignore)
├── results/  tables/  figures/  error_analysis/  runs/ (commit, D13)  predictions/ (gitignore)
└── report/  NLP_BTL_MaNhom.pdf  slides.pdf
```

## Review Focus

1. **Bình luận chỉ có `{OTHERS}`** (115 train / 18 dev / 37 test): phải ra một hàng toàn 0, không lỗi; hàm đánh giá tính được cả khi hàng đó đúng lẫn khi sai. → `test_parse_others_only` (Task 1), `test_all_none_rows_and_empty_prediction_no_crash` (Task 2).
2. **Khía cạnh hiếm STORAGE** (91 train, 14 dev): một lô hoặc một tập con có thể không có mẫu dương nào. Macro-F1 không lỗi chia cho 0; bộ phân loại và trọng số lớp không lỗi khi thiếu hẳn một lớp. → Task 2 `test_macro_ignores_absent_aspect`, Task 6 `test_single_class_aspect` và `test_single_polarity_aspect`, Task 8 `test_class_weights_zero_column`, Task 8 `test_split_loss_no_present_aspect`.
3. **Bình luận dài hơn `max_length`** (dài nhất 205 âm tiết): phải bị cắt, không lỗi. → Task 9 `test_truncation_long_input`, Task 10 `test_bilstm_truncates`.
4. **Đầu vào xấu ở demo:** rỗng, chỉ emoji, có `<script>`, hoặc CSV thiếu cột `comment`. Phải trả `{}` kèm HTML hợp lệ, escape đúng, và báo lỗi rõ ràng cho CSV. → Task 12.
5. **Rò rỉ tập Test:** mô hình không bao giờ thấy văn bản Test khi fit; `--eval-test` bị chặn với cấu hình ngoài `configs/final/` và khi có `--limit`. → Task 5 `test_fit_never_sees_test`, `test_eval_test_requires_final_config`, `test_eval_test_with_limit_raises`.

Ngoài ra: `py_vncorenlp` đổi thư mục làm việc khi nạp model (cần kiểm tra lại khi cài). → Task 3 `test_vncorenlp_keeps_cwd`.

---

### Task 1: Khung dự án, tải dữ liệu, đọc và tách nhãn

**Phụ trách:** A (B dựng môi trường và kiểm tra trên Colab)

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `requirements.txt`, `docs/decisions.md`, `scripts/download_data.py`, `src/absa/__init__.py`, `src/absa/data.py`, `tests/conftest.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `ASPECTS: tuple[str, ...]`, `POLARITIES: tuple[str, ...]`, `POLARITY_CODE: dict[str, int] = {"Positive": 1, "Neutral": 2, "Negative": 3}`
  - `parse_label(label: str) -> dict[str, str]`: ánh xạ khía cạnh sang cảm xúc, bỏ `{OTHERS}`, chấp nhận khoảng trắng thừa và thiếu `;` cuối.
  - `has_others(label: str) -> bool`
  - `format_label(aspects: dict[str, str]) -> str`: ví dụ `"{CAMERA#Positive};{BATTERY#Negative};"`, theo thứ tự `ASPECTS`; dict rỗng cho `""`.
  - `labels_to_matrix(labels: list[dict[str, str]]) -> np.ndarray`: dtype `int8`, shape `(n, 10)`.
  - `matrix_to_labels(Y: np.ndarray) -> list[dict[str, str]]`
  - `load_split(name: Literal["train", "dev", "test"], data_dir: Path = Path("data/raw")) -> pd.DataFrame`: giữ các cột gốc, thêm `aspects: dict` và `has_others: bool`; `n_star` kiểu int.
  - Fixture `fake_data_dir(tmp_path) -> Path` trong `tests/conftest.py`: ghi `Train.csv` (12 dòng), `Dev.csv` (4), `Test.csv` (4) đúng định dạng gốc. Mẫu nhãn lặp vòng: `"pin trâu máy đẹp" → {BATTERY#Positive};{DESIGN#Positive};`, `"pin yếu giá cao" → {BATTERY#Negative};{PRICE#Negative};`, `"giao hàng nhanh" → {SER&ACC#Positive};`, `"hihi" → {OTHERS};`. Comment có tiền tố tên split và hậu tố số dòng (`"test pin yếu giá cao 2"`) để không trùng nhau. `n_star` lặp vòng 5, 1, 4, 3.

- [ ] **Step 1: Chuẩn bị môi trường (B).** Cài Python 3.11/3.12 từ python.org và Java 17 (Temurin) cho VnCoreNLP. Rồi chạy `python -m venv .venv` và `.venv\Scripts\activate`. Chạy `python -c "import sys; print(sys.executable)"`; kỳ vọng đường dẫn nằm trong `.venv`, không phải `msys64`. Chạy `git init`.

- [ ] **Step 2: Viết test (fail)** trong `tests/test_data.py`:

```python
import re
from pathlib import Path
import numpy as np, pytest
from absa.data import (
    ASPECTS,
    parse_label,
    has_others,
    format_label,
    labels_to_matrix,
    matrix_to_labels,
    load_split,
)


def test_aspect_order():
    assert ASPECTS == (
        "SCREEN",
        "CAMERA",
        "FEATURES",
        "BATTERY",
        "PERFORMANCE",
        "STORAGE",
        "DESIGN",
        "PRICE",
        "GENERAL",
        "SER&ACC",
    )


def test_parse_basic():
    assert parse_label("{CAMERA#Positive};{BATTERY#Negative};{OTHERS};") == {
        "CAMERA": "Positive",
        "BATTERY": "Negative",
    }


def test_parse_others_only():
    assert parse_label("{OTHERS};") == {}
    assert has_others("{OTHERS};") is True
    assert has_others("{CAMERA#Positive};") is False


def test_parse_whitespace_and_no_trailing_semicolon():
    assert parse_label(" {SER&ACC#Neutral} ; {PRICE#Positive}") == {
        "SER&ACC": "Neutral",
        "PRICE": "Positive",
    }


@pytest.mark.parametrize(
    "bad, token",
    [
        ("{CAMRA#Positive};", "CAMRA#Positive"),
        ("{CAMERA#Good};", "CAMERA#Good"),
        ("{CAMERA};", "CAMERA"),
        ("{CAMERA#Positive};{CAMERA#Negative};", "CAMERA"),
    ],
)
def test_parse_invalid_raises(bad, token):
    with pytest.raises(ValueError, match=re.escape(token)):
        parse_label(bad)


def test_format_label_order_and_empty():
    assert (
        format_label({"BATTERY": "Negative", "CAMERA": "Positive"})
        == "{CAMERA#Positive};{BATTERY#Negative};"
    )
    assert format_label({}) == ""


def test_matrix_roundtrip():
    labels = [{"CAMERA": "Positive", "STORAGE": "Negative"}, {}]
    Y = labels_to_matrix(labels)
    assert Y.shape == (2, 10) and Y.dtype == np.int8
    assert Y[0, ASPECTS.index("CAMERA")] == 1 and Y[0, ASPECTS.index("STORAGE")] == 3
    assert Y[1].sum() == 0
    assert matrix_to_labels(Y) == labels
    assert labels_to_matrix([]).shape == (0, 10)


def test_load_split_fake(fake_data_dir):
    df = load_split("train", fake_data_dir)
    assert len(df) == 12
    assert {"index", "comment", "n_star", "date_time", "label", "aspects", "has_others"} <= set(
        df.columns
    )
    assert df["n_star"].dtype.kind == "i"
    assert int(df["aspects"].map(len).eq(0).sum()) == 3


REAL = Path("data/raw/Train.csv")


@pytest.mark.skipif(not REAL.exists(), reason="chưa tải dữ liệu")
def test_load_split_real_sizes():
    sizes = {s: len(load_split(s)) for s in ("train", "dev", "test")}
    assert sizes == {"train": 7786, "dev": 1112, "test": 2224}
    only_others = {
        s: int(load_split(s)["aspects"].map(len).eq(0).sum()) for s in ("train", "dev", "test")
    }
    assert only_others == {"train": 115, "dev": 18, "test": 37}
    assert int(load_split("train")["aspects"].map(lambda d: "STORAGE" in d).sum()) == 91
```

Thông báo lỗi: `"Nhãn không hợp lệ: {token}"` cho khía cạnh hoặc cảm xúc lạ (token là phần trong ngoặc đã strip); `"Khía cạnh lặp: {aspect}"` khi một khía cạnh xuất hiện hai lần.

- [ ] **Step 3: Chạy `pytest tests/test_data.py -v`**, kỳ vọng FAIL (`ModuleNotFoundError: absa`).

- [ ] **Step 4: Tạo khung dự án.**
  - `pyproject.toml`: layout `src/`, package `absa`, `requires-python = ">=3.10"`, `[tool.pytest.ini_options] pythonpath = ["src"]`, `testpaths = ["tests"]`.
  - `.gitignore`: `data/`, `models/`, `results/predictions/`, `*.pt`, `.ipynb_checkpoints/`, `.venv/`, `__pycache__/`.
  - `requirements.txt` (cận dưới, không ghim): `pandas>=2.0, numpy>=1.24, scikit-learn>=1.3, underthesea>=6.8, py_vncorenlp>=0.1.4, torch>=2.1, transformers>=4.40, sentencepiece>=0.1.99, gradio>=4.0, matplotlib>=3.7, PyYAML>=6.0, tabulate>=0.9, jupyter, nbconvert, pytest>=7`. Ghi chú đầu file: trên Windows nên cài torch bản CPU bằng `pip install torch --index-url https://download.pytorch.org/whl/cpu` trước.
  - `docs/decisions.md`: tiêu đề, chép nguyên phần "Quyết định so với spec" ở trên, rồi thêm mục trống `## Nhật ký quyết định khi chạy`.

- [ ] **Step 5: Viết `scripts/download_data.py`.** Tải URL ở Global Constraints bằng `urllib.request`, kiểm tra SHA-256 (sai thì xóa file tạm, báo lỗi, exit 1), giải nén bằng `zipfile` vào `data/raw/`. Nếu cả 3 CSV đã có thì in `"Đã có dữ liệu"` và bỏ qua. Có tham số `--data-dir` (mặc định `data/raw`).

- [ ] **Step 6: Cài đặt `src/absa/data.py`.** Tách token bằng regex `\{([^}]*)\}`, strip từng token. Token phải là `OTHERS` hoặc `ASPECT#POLARITY` với ASPECT ∈ `ASPECTS` và POLARITY ∈ `POLARITIES`. Đọc CSV bằng `encoding="utf-8"`. `load_split` báo `ValueError` nếu thiếu cột bắt buộc; tên file là `{Name}.csv` (Train/Dev/Test).

- [ ] **Step 7: Chạy `pip install -e . && pip install -r requirements.txt && python scripts/download_data.py && pytest tests/test_data.py -v`**. Kỳ vọng PASS toàn bộ, kể cả `test_load_split_real_sizes`. Chạy `pip freeze > requirements.lock.txt`.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore requirements.txt requirements.lock.txt docs/ scripts/download_data.py src/absa/__init__.py src/absa/data.py tests/conftest.py tests/test_data.py
git commit -m "feat: project skeleton, data download and label parsing"
```

---

### Task 2: Hàm đánh giá và cách tính của bài báo

**Phụ trách:** B

**Files:**
- Create: `src/absa/evaluate.py`, `docs/metric_notes.md`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `ASPECTS`, quy ước mã 0–3 (Task 1)
- Produces:
  - `evaluate(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict`, các khóa:
    - `aspect_macro_f1`, `aspect_micro_f1`: 10 bài toán nhị phân `Y > 0`. Macro chỉ lấy các khía cạnh có ít nhất 1 mẫu dương trong `Y_true`; không có khía cạnh nào thì trả `0.0`.
    - `sentiment_macro_f1`, `sentiment_micro_f1`: 30 cột nhị phân `Y == k` (k = 1..3, theo khía cạnh); macro chỉ lấy các cột có mẫu dương trong gold; không có cột nào thì `0.0`.
    - `polarity_macro_f1_detected` (D5): lấy các ô `(i, a)` có `Y_true > 0` và `Y_pred > 0`, tính macro-F1 3 lớp trên các lớp có mặt trong gold của tập con đó; tập con rỗng thì `0.0`.
    - `exact_match`: tỉ lệ hàng đúng hoàn toàn.
    - `per_aspect`: `{aspect: {"precision", "recall", "f1", "support"}}` cho bài toán khía cạnh; `support` kiểu int.
  - Mọi giá trị là `float` Python (không phải `np.float64`, để ghi JSON được), nằm trong [0, 1], không làm tròn.
  - `confusion_by_aspect(Y_true: np.ndarray, Y_pred: np.ndarray, aspect: str) -> np.ndarray`: shape `(4, 4)`, hàng là gold, cột là dự đoán, nhãn 0..3.

- [ ] **Step 1: Viết test (fail)**

```python
import json
import numpy as np
from absa.evaluate import evaluate, confusion_by_aspect

C, B = 1, 3  # cột CAMERA, BATTERY


def make(rows):
    Y = np.zeros((len(rows), 10), dtype=np.int8)
    for i, r in enumerate(rows):
        for col, v in r.items():
            Y[i, col] = v
    return Y


def test_perfect():
    Y = make([{C: 1}, {B: 3}, {}])
    m = evaluate(Y, Y)
    assert m["aspect_macro_f1"] == 1.0 and m["sentiment_macro_f1"] == 1.0
    assert m["exact_match"] == 1.0 and m["polarity_macro_f1_detected"] == 1.0


def test_right_aspect_wrong_polarity():
    m = evaluate(make([{C: 1}]), make([{C: 3}]))
    assert m["aspect_macro_f1"] == 1.0
    assert m["sentiment_macro_f1"] == 0.0
    assert m["exact_match"] == 0.0


def test_all_none_rows_and_empty_prediction_no_crash():
    m = evaluate(make([{}, {}]), make([{}, {C: 1}]))
    assert m["exact_match"] == 0.5
    assert m["aspect_macro_f1"] == 0.0 and m["sentiment_macro_f1"] == 0.0
    assert m["polarity_macro_f1_detected"] == 0.0


def test_macro_ignores_absent_aspect():
    m = evaluate(make([{C: 1}, {B: 1}]), make([{C: 1}, {}]))
    assert m["aspect_macro_f1"] == 0.5
    assert m["per_aspect"]["STORAGE"]["support"] == 0


def test_polarity_detected():
    m = evaluate(make([{C: 1}, {B: 3}]), make([{C: 1}, {B: 1}]))
    assert abs(m["polarity_macro_f1_detected"] - 1 / 3) < 1e-9  # Positive F1=2/3, Negative F1=0


def test_json_serializable():
    json.dumps(evaluate(make([{C: 1}]), make([{C: 1}])))


def test_confusion_by_aspect():
    cm = confusion_by_aspect(make([{C: 1}, {C: 3}, {}]), make([{C: 1}, {C: 1}, {C: 2}]), "CAMERA")
    assert cm.shape == (4, 4) and cm.sum() == 3
    assert cm[1, 1] == 1 and cm[3, 1] == 1 and cm[0, 2] == 1
```

- [ ] **Step 2: Chạy `pytest tests/test_evaluate.py -v`**, kỳ vọng FAIL.
- [ ] **Step 3: Cài đặt** bằng `sklearn.metrics.precision_recall_fscore_support(average=None, zero_division=0)` trên ma trận nhị phân, rồi tự lấy trung bình trên các cột có support > 0. Micro dùng `average="micro"`. Confusion dùng `sklearn.metrics.confusion_matrix(labels=[0, 1, 2, 3])`.
- [ ] **Step 4: Chạy lại test**, kỳ vọng PASS.
- [ ] **Step 5: Viết `docs/metric_notes.md`.** Đọc phần Experiments của Phan et al. (2021) (bản PDF qua thư viện trường hoặc liên hệ tác giả) và ghi lại:
  - (a) F1 khía cạnh 84,48 là micro, macro hay weighted?
  - (b) F1 cảm xúc 63,06 tính trên toàn bộ cặp (khía cạnh, cảm xúc), hay chỉ trên khía cạnh đã phát hiện đúng?
  - (c) Họ đánh giá trên Dev hay Test?
  - (d) Kết luận: số 84,48 ứng với khóa nào của `evaluate` và số 63,06 ứng với khóa nào; nếu không khớp khóa nào thì ghi `"không cùng định nghĩa"`.

  Ghi kèm trích dẫn trang. Cuối file thêm một khối máy đọc được (Task 7 dùng):

  ````
  ```yaml
  published_metric: {aspect: aspect_macro_f1, sentiment: sentiment_macro_f1}   # hoặc: published_metric: null
  ```
  ````

  Không truy cập được bài báo thì ghi đúng như vậy và đặt `published_metric: null`; khi đó bảng dùng nhãn "không cùng định nghĩa".
- [ ] **Step 6: Commit** `feat: ABSA evaluation metrics and notes on paper metric` (gồm `src/absa/evaluate.py`, `tests/test_evaluate.py`, `docs/metric_notes.md`).

---

### Task 3: Tiền xử lý và cache

**Phụ trách:** A

**Files:**
- Create: `src/absa/preprocess.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Produces:
  - `PRE_DEFAULTS = {"lowercase": False, "strip_emoji": False, "strip_punct": False, "segment": None, "use_star": False}`
  - `normalize(text: str, *, lowercase: bool = False, strip_emoji: bool = False, strip_punct: bool = False) -> str`. Thứ tự xử lý: NFC → `underthesea.text_normalize` → bỏ emoji → bỏ dấu câu (ký tự loại Unicode `P*` thay bằng khoảng trắng) → chữ thường → gộp khoảng trắng và strip.
  - `segment(texts: list[str], backend: Literal["underthesea", "vncorenlp"] = "underthesea") -> list[str]`: từ ghép nối bằng `_`; chuỗi rỗng giữ nguyên là `""`.
  - `preprocess_df(df: pd.DataFrame, pre_cfg: dict, cache_dir: Path | None = None, cache_tag: str | None = None) -> list[str]`. Gộp `pre_cfg` với `PRE_DEFAULTS` (khóa lạ thì báo `ValueError`). Thứ tự: normalize → segment → tiền tố `"[{n_star} sao] "` nếu `use_star`. Chỉ cache khi có cả `cache_dir` lẫn `cache_tag`. File cache: `{cache_dir}/{cache_tag}__{sha1(json.dumps(cfg, sort_keys=True))[:10]}.json`, nội dung `{"n": int, "config": dict, "texts": list[str]}`. Nếu `n` khác `len(df)` thì tính lại.

- [ ] **Step 1: Viết test (fail)**

```python
import os, shutil, unicodedata
import pandas as pd, pytest
from absa import preprocess as P


def test_nfc_and_tone_consistent():
    out = P.normalize(unicodedata.normalize("NFD", "Pin hóa đá"))
    assert out == unicodedata.normalize("NFC", out)
    assert P.normalize("hoá đá") == P.normalize("hóa đá")


def test_strip_emoji():
    assert P.normalize("pin trâu 😍👍", strip_emoji=True) == "pin trâu"
    assert P.normalize("😍😍", strip_emoji=True) == ""
    assert P.normalize("pin trâu 😍").endswith("😍")  # mặc định giữ emoji


def test_strip_punct_lower_whitespace():
    assert P.normalize("pin trâu!!! máy ok...", strip_punct=True) == "pin trâu máy ok"
    assert P.normalize("Pin TRÂU", lowercase=True) == "pin trâu"
    assert P.normalize("  pin \n trâu ") == "pin trâu"
    assert P.normalize("") == ""


def test_segment_underthesea():
    out = P.segment(["điện thoại pin trâu", ""])
    assert "điện_thoại" in out[0] and out[1] == ""


@pytest.mark.skipif(shutil.which("java") is None, reason="không có Java")
def test_vncorenlp_keeps_cwd():
    cwd = os.getcwd()
    out = P.segment(["điện thoại pin trâu"], backend="vncorenlp")
    assert "điện_thoại" in out[0] and os.getcwd() == cwd


def test_use_star_prefix_after_segment():
    df = pd.DataFrame({"comment": ["điện thoại tốt"], "n_star": [5]})
    out = P.preprocess_df(df, {"segment": "underthesea", "use_star": True})
    assert out[0].startswith("[5 sao] ") and "điện_thoại" in out[0]


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="lowercse"):
        P.preprocess_df(pd.DataFrame({"comment": ["a"], "n_star": [1]}), {"lowercse": True})


def test_cache_hit(tmp_path, monkeypatch):
    df = pd.DataFrame({"comment": ["pin trâu", "máy đẹp"], "n_star": [5, 4]})
    first = P.preprocess_df(df, {"segment": "underthesea"}, tmp_path, "train")
    monkeypatch.setattr(
        P, "segment", lambda *a, **k: (_ for _ in ()).throw(AssertionError("không dùng cache"))
    )
    assert P.preprocess_df(df, {"segment": "underthesea"}, tmp_path, "train") == first
    assert len(list(tmp_path.glob("train__*.json"))) == 1
```

- [ ] **Step 2: Chạy `pytest tests/test_preprocess.py -v`**, kỳ vọng FAIL.
- [ ] **Step 3: Cài đặt.**
  - Regex emoji gồm các dải U+1F000–U+1FAFF, U+2600–U+27BF, U+2B00–U+2BFF, U+FE0F, U+200D, U+20E3. Emoticon dạng chữ như `:)` không xử lý; ghi điều này thành giới hạn trong `docs/decisions.md`.
  - Backend `underthesea`: `underthesea.word_tokenize(text, format="text")`.
  - Backend `vncorenlp`: nạp `py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(Path("models/vncorenlp").resolve()))` một lần (biến module), tải model bằng `py_vncorenlp.download_model` nếu chưa có. Lưu `os.getcwd()` trước khi nạp và `os.chdir` về sau, kể cả khi lỗi (`try/finally`). Mỗi văn bản: `" ".join(rdr.word_segment(text))`.
- [ ] **Step 4: Chạy test**, kỳ vọng PASS (`test_vncorenlp_keeps_cwd` có thể bị skip nếu máy không có Java; B chạy lại trên Colab để xác nhận PASS).
- [ ] **Step 5: Commit** `feat: Vietnamese normalization, segmentation and preprocessing cache`

---

### Task 4: Khảo sát dữ liệu (EDA)

**Phụ trách:** A

**Files:**
- Create: `src/absa/eda.py`, `notebooks/01_eda.ipynb`
- Test: `tests/test_eda.py`

**Interfaces:**
- Consumes: `load_split`, `parse_label` (Task 1), `normalize` (Task 3)
- Produces:
  - `aspect_polarity_table(df: pd.DataFrame) -> pd.DataFrame`: đọc cột `aspects` (dict). Chỉ mục là 10 khía cạnh theo thứ tự `ASPECTS` (kể cả khía cạnh có 0 mẫu), các cột `Positive, Neutral, Negative, Total` kiểu int.
  - `aspects_per_comment(df) -> pd.Series`: chỉ mục là số khía cạnh (0, 1, 2, …), giá trị là số bình luận.
  - `star_vs_polarity(df) -> pd.DataFrame`: chỉ mục `n_star` 1–5, cột là số cặp Positive/Neutral/Negative.
  - `length_stats(texts: list[str]) -> dict`: các khóa `chars_mean`, `syllables_mean`, `syllables_p95`, `syllables_p99`, `syllables_max`.
  - `duplicate_report(splits: dict[str, pd.DataFrame]) -> dict[str, int]`: các khóa `train_dup`, `train_dev_overlap`, `train_test_overlap`, `dev_test_overlap`. So sánh trên `normalize(comment, lowercase=True)`.
  - `emoji_stats(texts: list[str], top: int = 10) -> dict`: `share_with_emoji` (float) và `top_emojis` (`list[tuple[str, int]]`).
  - `TEENCODE: tuple[str, ...] = ("k", "ko", "kh", "hk", "khg", "dc", "đc", "ok", "oke", "sp", "mn", "bt", "vs", "j", "r", "mik", "mk", "đt", "dt", "cam", "fb", "nv", "trc", "ms")`, và `teencode_counts(texts, terms=TEENCODE) -> pd.Series`: số bình luận chứa mỗi từ (tách theo khoảng trắng, chữ thường), sắp giảm dần.

- [ ] **Step 1: Viết test (fail)** với DataFrame giả 3 dòng: nhãn `{CAMERA#Positive};{PRICE#Negative};`, `{OTHERS};`, `{CAMERA#Neutral};`; `n_star` lần lượt 5, 1, 3; cột `aspects` tạo bằng `parse_label`.
  - `aspect_polarity_table`: `.loc["CAMERA","Total"] == 2`, `.loc["PRICE","Negative"] == 1`, `.loc["STORAGE","Total"] == 0`, có đủ 10 hàng.
  - `aspects_per_comment(df).to_dict() == {0: 1, 1: 1, 2: 1}`.
  - `star_vs_polarity(df).loc[5, "Positive"] == 1` và `.loc[5, "Negative"] == 1`.
  - `length_stats(["a b c", "a"])`: `syllables_mean == 2.0`, `syllables_max == 3`.
  - `duplicate_report({"train": df_of(["a", "b", "A"]), "dev": df_of(["b"]), "test": df_of(["c"])}) == {"train_dup": 1, "train_dev_overlap": 1, "train_test_overlap": 0, "dev_test_overlap": 0}`.
  - `emoji_stats(["pin 😍", "ok", "😍👍"])`: `share_with_emoji` xấp xỉ `2/3`, `top_emojis[0] == ("😍", 2)`.
  - `teencode_counts(["ko có gì", "Ko ok"])["ko"] == 2`.
- [ ] **Step 2: Chạy test**, kỳ vọng FAIL.
- [ ] **Step 3: Cài đặt `eda.py`.**
- [ ] **Step 4: Viết `notebooks/01_eda.ipynb`.** Mỗi mục có một ô markdown nhận xét ngắn. Thứ tự: nạp dữ liệu → bảng → hình → nhận xét. Xuất ra:
  - `results/tables/eda_aspect_polarity_{train,dev,test}.csv`, `eda_aspects_per_comment.csv`, `eda_star_vs_polarity.csv`, `eda_length.csv`, `eda_duplicates.json`, `eda_emoji.json`, `eda_teencode.csv`.
  - `results/figures/eda_aspect_freq.png` (cột), `eda_aspect_polarity_stacked.png` (cột chồng), `eda_star_polarity_heatmap.png`, `eda_length_hist.png`.
  - Ô tùy chọn (chỉ chạy khi có `transformers` và mạng): đếm số token con (subword) của PhoBERT sau tách từ VnCoreNLP, và của ViSoBERT trên văn bản thô, trên Train. Ghi `results/tables/eda_subword_len.csv` (các cột mean, p95, p99, max).
- [ ] **Step 5: Chạy `pytest tests/test_eda.py -v`** (PASS), rồi `jupyter nbconvert --execute --to notebook --inplace notebooks/01_eda.ipynb` (không lỗi). Mở `eda_duplicates.json`: nếu `train_test_overlap > 0` thì ghi số đó vào `docs/decisions.md` (không xóa mẫu, vì phải giữ split gốc).
- [ ] **Step 6: Ghi `max_length` vào `docs/decisions.md`** theo D9, dùng `eda_subword_len.csv` (ô tùy chọn có thể chạy trên Colab). Kỳ vọng thô: p95 khoảng 78 âm tiết, nên `max_length` có lẽ là 128 hoặc 160; hãy dùng số đo thật.
- [ ] **Step 7: Commit** `feat: EDA module, notebook and figures` (gồm cả `results/tables/eda_*` và `results/figures/eda_*`).

---

### Task 5: Cấu hình và bộ chạy thí nghiệm

**Phụ trách:** B

**Files:**
- Create: `src/absa/config.py`, `src/absa/registry.py`, `src/absa/experiment.py`, `scripts/run.py`
- Test: `tests/test_config.py`, `tests/test_experiment.py`

**Interfaces:**
- Consumes: `load_split`, `labels_to_matrix`, `format_label`, `matrix_to_labels` (Task 1), `evaluate` (Task 2), `preprocess_df` (Task 3)
- Produces:
  - `load_config(path: Path) -> dict`: nếu có khóa `base` (đường dẫn tương đối so với file hiện tại) thì nạp đệ quy và gộp sâu (dict con ghi đè từng khóa). Hai khóa `name` và `groups` **không** kế thừa. Thiếu `name` thì báo `ValueError`. Kết quả không còn khóa `base`. Khóa `groups` mặc định `[]`.
  - `apply_overrides(cfg: dict, overrides: list[str]) -> dict`: mỗi phần tử có dạng `"model.C=2"`; giá trị parse bằng `yaml.safe_load`; khóa phải tồn tại sẵn (không thì `ValueError`). Mỗi override thêm hậu tố `"__C=2"` (khóa lá) vào `name` và đặt `groups = []`, vì một lần chạy có override không phải là nhánh ablation. Trả về dict mới, không sửa dict gốc.
  - `class Model(Protocol)`: `fit(texts: list[str], Y: np.ndarray, dev_texts: list[str] | None = None, Y_dev: np.ndarray | None = None) -> Model`; `predict(texts: list[str]) -> np.ndarray` shape `(n, 10)`, int8. Tùy chọn có `info: dict` và `save(path: Path) -> None`.
  - `REGISTRY: dict[str, str] = {"tfidf": "absa.baseline:TfidfAbsa", "transformer": "absa.transformer:TransformerAbsa", "bilstm": "absa.bilstm:BiLstmAbsa"}` (nạp lười, để TF-IDF không cần torch).
  - `build_model(model_cfg: dict, seed: int) -> Model`: gọi `Cls(**{k: v for k, v in model_cfg.items() if k != "type"}, seed=seed)`. `type` lạ thì `ValueError` liệt kê các loại hợp lệ. Giá trị trong REGISTRY có thể là chuỗi `"module:Class"` hoặc một callable (test dùng callable).
  - `assert_final_config(path: Path) -> None`: báo `ValueError` nếu `configs/final` không nằm trong `path.resolve().parents`.
  - `run_experiment(cfg: dict, *, eval_test: bool = False, limit: int | None = None, data_dir: Path = Path("data/raw"), out_dir: Path = Path("results"), cache_dir: Path | None = Path("data/processed")) -> dict`:
    1. `eval_test` cùng `limit` thì báo `ValueError("Không đánh giá test khi dùng --limit")`.
    2. Nạp các split; tiền xử lý **toàn bộ** split với `cache_tag=split`; rồi mới cắt `limit` dòng đầu của train và dev (test không bao giờ cắt).
    3. Gọi `set_seed(cfg["seed"])` nếu có torch (import trong `try`); `build_model`; đo `train_seconds` cho `fit(train, Y, dev, Y_dev)`.
    4. Dự đoán dev (và test nếu `eval_test`), gọi `evaluate`.
    5. Nếu `cfg.get("save")` và model có `save`, lưu vào `models/{name}__seed{seed}/`.
    6. Ghi `out_dir/runs/{name}__seed{seed}.json` với các khóa `name, seed, groups, config, dev, test (None nếu không có), train_seconds, limit, model_info, git_commit ("unknown" nếu lỗi), timestamp (ISO), versions ({gói: phiên bản} cho python, numpy, scikit-learn, torch, transformers nếu có)`.
    7. Ghi `out_dir/predictions/{name}__seed{seed}__{split}.csv` với các cột `index, comment, n_star, gold, pred` (dùng `format_label`; `comment` là văn bản gốc).
    8. Trả về dict đã ghi.
  - CLI: `python scripts/run.py --config A.yaml [B.yaml ...] [--seeds 42 43 44] [--eval-test] [--limit N] [--set k=v ...]`. Cấu hình chạy lần lượt, mỗi seed một lần. `--seeds` ghi đè `cfg["seed"]`. Với `--eval-test`, gọi `assert_final_config` cho từng file trước khi chạy bất cứ thứ gì. Sau mỗi lần chạy in một dòng `"{name} seed={seed} dev aspect={:.2%} sentiment={:.2%} ({train_seconds:.0f}s)"`.

- [ ] **Step 1: Viết `tests/test_config.py` (fail):**
  - `test_inherit_deep_merge`: base `{name: b, seed: 42, groups: [E1], preprocess: {lowercase: true}, model: {type: tfidf, C: 1}}`, con `{base: base.yaml, name: c, model: {C: 2}}`. Kỳ vọng `cfg["model"] == {"type": "tfidf", "C": 2}`, `cfg["preprocess"] == {"lowercase": True}`, `cfg["groups"] == []`, `"base" not in cfg`.
  - `test_missing_name_raises`.
  - `test_override_suffix_and_parse`: `apply_overrides(cfg, ["model.C=0.5", "preprocess.segment=null"])` cho `C == 0.5`, `segment is None`, `name == "b__C=0.5__segment=null"`, `groups == []`; dict gốc không đổi.
  - `test_override_unknown_key_raises`: `["model.Cc=1"]` báo `ValueError`.
- [ ] **Step 2: Viết `tests/test_experiment.py` (fail)**, dùng `fake_data_dir` và một `DummyModel` đăng ký bằng `monkeypatch.setitem(registry.REGISTRY, "dummy", DummyModel)`. `DummyModel.fit` lưu lại `texts` và `dev_texts`; `predict` trả ma trận 0.
  - `test_run_writes_json_and_predictions`: `run_experiment({"name": "d", "seed": 42, "preprocess": {}, "model": {"type": "dummy"}}, data_dir=fake, out_dir=tmp, cache_dir=None)`. Kỳ vọng có `tmp/runs/d__seed42.json` với `test is None` và `dev["exact_match"] == 0.25` (1/4 dòng dev là `{OTHERS}`); file `d__seed42__dev.csv` có 4 dòng, cột đúng.
  - `test_fit_never_sees_test`: với `eval_test=True`, không chuỗi nào trong `fit_texts + fit_dev_texts` bắt đầu bằng `"test "`; file JSON có `test` khác None.
  - `test_limit_slices_train_dev`: `limit=3` → fit nhận đúng 3 văn bản train và 3 văn bản dev.
  - `test_eval_test_with_limit_raises`.
  - `test_eval_test_requires_final_config`: `assert_final_config(Path("configs/tfidf_lr_base.yaml"))` báo lỗi; `assert_final_config(Path("configs/final/x.yaml"))` không báo lỗi.
  - `test_unknown_model_type_raises`: `build_model({"type": "abc"}, 42)` báo `ValueError` có chữ `tfidf`.
- [ ] **Step 3: Chạy `pytest tests/test_config.py tests/test_experiment.py -v`**, kỳ vọng FAIL.
- [ ] **Step 4: Cài đặt** `config.py`, `registry.py`, `experiment.py`, `scripts/run.py`. `set_seed` lấy từ `absa.training` (Task 9). Trước khi có Task 9, `run_experiment` chỉ đặt `random.seed` và `numpy.random.seed`, rồi `try: from absa.training import set_seed`.
- [ ] **Step 5: Chạy test**, kỳ vọng PASS.
- [ ] **Step 6: Commit** `feat: config inheritance, model registry and experiment runner`

---

### Task 6: Baseline TF-IDF và khảo sát E0–E3, E6 (TF-IDF)

**Phụ trách:** A

**Files:**
- Create: `src/absa/baseline.py`, `configs/tfidf_lr_base.yaml`, `configs/tfidf_svm.yaml`, `configs/ablation/E1_ngram_1.yaml`, `E1_ngram_13.yaml`, `E1_char_25.yaml`, `E2_word.yaml`, `E3_tfidf_strip_emoji.yaml`, `E6_tfidf_joint.yaml`, `configs/smoke/tfidf_smoke.yaml`
- Test: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `Model` Protocol, `REGISTRY["tfidf"]` (Task 5)
- Produces: `class TfidfAbsa(mode: Literal["split", "joint"] = "split", analyzer: Literal["word", "char_wb"] = "word", ngram_range: tuple[int, int] | list[int] = (1, 2), min_df: int = 2, classifier: Literal["lr", "svm"] = "lr", C: float = 1.0, seed: int = 42)`
  - `.vectorizer`: một `TfidfVectorizer(sublinear_tf=True, min_df=min_df, analyzer=analyzer, ngram_range=tuple(ngram_range))` duy nhất, fit trên Train.
  - `mode="split"`: theo D1. Bộ phát hiện khía cạnh train trên `Y[:, a] > 0` của toàn bộ Train. Bộ cảm xúc train trên các hàng có `Y[:, a] > 0`, nhãn 1..3. Lúc dự đoán: có khía cạnh thì lấy nhãn của bộ cảm xúc, không thì 0.
  - `mode="joint"`: mỗi khía cạnh một bộ 4 lớp (0–3).
  - Bộ phân loại: `LogisticRegression(C=C, max_iter=2000, random_state=seed)` hoặc `LinearSVC(C=C, random_state=seed)`. Một bộ (bất kỳ bước nào) mà dữ liệu train chỉ có một lớp thì dùng `DummyClassifier(strategy="most_frequent")`. Khía cạnh không có hàng nào để train bộ cảm xúc thì dự đoán 0 (không có khía cạnh).
  - `mode` hoặc `classifier` lạ: báo `ValueError` ngay trong `__init__`.
  - `fit` bỏ qua `dev_texts`/`Y_dev`; `predict` trả int8 `(n, 10)`.

- [ ] **Step 1: Viết test (fail)**, các test dùng `min_df=1`:
  - `test_predict_shape_dtype`: trên 4 văn bản đồ chơi, shape `(n, 10)`, dtype int8, giá trị ⊆ {0, 1, 2, 3}, cho cả `split` và `joint`.
  - `test_vectorizer_fit_on_train_only`: fit trên `["pin trâu", "pin yếu"]`; sau `predict(["camera đẹp"])` thì `vectorizer.vocabulary_` không đổi và không chứa `"camera"`.
  - `test_single_class_aspect`: cột STORAGE toàn 0 → fit không lỗi, dự đoán STORAGE toàn 0 (cả hai mode).
  - `test_single_polarity_aspect`: CAMERA luôn là 1 ở mọi hàng có CAMERA → các dự đoán CAMERA khác 0 đều bằng 1.
  - `test_learns_trivial_signal`: 10 lần `"pin tốt"` → BATTERY=1, 10 lần `"pin tệ"` → BATTERY=3, 10 lần `"giao nhanh"` → SER&ACC=1. Kỳ vọng `predict(["pin tốt", "pin tệ"])[:, 3].tolist() == [1, 3]`, cho cả hai mode và cả `lr`, `svm`.
  - `test_invalid_mode_raises`: `TfidfAbsa(mode="x")` báo `ValueError`.
  - `test_run_experiment_smoke_real` (skip nếu chưa có `data/raw/Train.csv`): `run_experiment(load_config(Path("configs/smoke/tfidf_smoke.yaml")), limit=200, out_dir=tmp_path)`; có JSON, `test is None`, `0 <= dev["aspect_macro_f1"] <= 1`.
- [ ] **Step 2: Chạy `pytest tests/test_baseline.py -v`**, kỳ vọng FAIL.
- [ ] **Step 3: Cài đặt `baseline.py`.**
- [ ] **Step 4: Viết các file YAML.** Cấu hình gốc:

```yaml
# configs/tfidf_lr_base.yaml
name: tfidf_lr
seed: 42
groups: [E1, E2, E3, E6_tfidf]
preprocess: {lowercase: true, strip_emoji: false, strip_punct: false, segment: null, use_star: false}
model: {type: tfidf, mode: split, analyzer: word, ngram_range: [1, 2], min_df: 2, classifier: lr, C: 1.0}
```

  Mỗi file ablation có dạng `base: ../tfidf_lr_base.yaml`, một `name` (trùng tên file, ví dụ `E1_ngram_1`), `groups: [E1]`, `[E2]`, `[E3]` hoặc `[E6_tfidf]` (khớp với `groups` của cấu hình gốc), và **đúng một** khóa thay đổi:
  - `E1_ngram_1`: `ngram_range: [1, 1]`
  - `E1_ngram_13`: `[1, 3]`
  - `E1_char_25`: `analyzer: char_wb, ngram_range: [2, 5]`. Đây là hai khóa của cùng một yếu tố "loại đặc trưng"; ghi chú điều này trong file.
  - `E2_word`: `preprocess.segment: underthesea`
  - `E3_tfidf_strip_emoji`: `preprocess.strip_emoji: true`
  - `E6_tfidf_joint`: `model.mode: joint`

  `tfidf_svm.yaml`: `base: tfidf_lr_base.yaml`, `name: tfidf_svm`, `model.classifier: svm`, không có groups. `smoke/tfidf_smoke.yaml`: `base: ../tfidf_lr_base.yaml`, `name: tfidf_smoke`.
- [ ] **Step 5: Chạy test**, kỳ vọng PASS.
- [ ] **Step 6: E0 – dò C trên Dev.** Chạy cả 5 giá trị qua `--set`, để tên run có hậu tố `__C=` và không trùng với run gốc:

```bash
python scripts/run.py --config configs/tfidf_lr_base.yaml --set model.C=0.25
python scripts/run.py --config configs/tfidf_lr_base.yaml --set model.C=0.5
python scripts/run.py --config configs/tfidf_lr_base.yaml --set model.C=1.0
python scripts/run.py --config configs/tfidf_lr_base.yaml --set model.C=2
python scripts/run.py --config configs/tfidf_lr_base.yaml --set model.C=4
```

  Lặp lại 5 lệnh trên với `configs/tfidf_svm.yaml`. Với mỗi bộ phân loại, chọn C có `dev.sentiment_macro_f1` cao nhất; bằng nhau thì chọn C nhỏ hơn. Ghi C của LR vào `tfidf_lr_base.yaml`; nếu C tốt nhất của SVM khác thì ghi đè `model.C` trong `tfidf_svm.yaml`. Ghi cả hai bảng 5 giá trị kèm lựa chọn vào `docs/decisions.md`. `sweep_table` lọc các run theo tiền tố `tfidf_lr__C=` và `tfidf_svm__C=`.
- [ ] **Step 7: Chạy cấu hình chính và ablation** (cùng C đã chốt): `python scripts/run.py --config configs/tfidf_lr_base.yaml configs/tfidf_svm.yaml configs/ablation/E1_ngram_1.yaml configs/ablation/E1_ngram_13.yaml configs/ablation/E1_char_25.yaml configs/ablation/E2_word.yaml configs/ablation/E3_tfidf_strip_emoji.yaml configs/ablation/E6_tfidf_joint.yaml`. Kỳ vọng: mỗi cấu hình in một dòng dev. Ghi thời gian chạy của `tfidf_lr` vào `docs/decisions.md` (để đưa vào README).
- [ ] **Step 8: Commit** `feat: TF-IDF baseline, E0 sweep and TF-IDF ablations` (gồm `configs/`, `docs/decisions.md`, `src/absa/baseline.py`, test).

---

### Task 7: Bảng so sánh tự động

**Phụ trách:** A

**Files:**
- Create: `src/absa/report.py`, `scripts/make_tables.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: định dạng JSON của run (Task 5), ghi chú `docs/metric_notes.md` (Task 2)
- Produces:
  - `MAIN_METRICS = ("aspect_macro_f1", "sentiment_macro_f1", "polarity_macro_f1_detected", "exact_match")`
  - `collect_runs(runs_dir: Path = Path("results/runs")) -> pd.DataFrame`: dạng dài, mỗi hàng là một `(name, seed, split)` với split ∈ {dev, test} và chỉ lấy split khác None. Các cột: `name, seed, split, groups, train_seconds`, các cột `MAIN_METRICS`, và `per_aspect` (dict).
  - `PUBLISHED = {"BiLSTM+fastText (Phan et al., 2021)": {"aspect": 84.48, "sentiment": 63.06}}`
  - `comparison_table(runs: pd.DataFrame, split: Literal["dev", "test"], names: list[str] | None = None, published_metric: dict[str, str] | None = None) -> pd.DataFrame`. Các cột: `name`, các `MAIN_METRICS`, `n_seeds`, `source`. Ô có dạng `"84.48 ± 0.31"` khi `n_seeds > 1` (std mẫu, ddof=1), dạng `"84.48"` khi `n_seeds == 1`. `source="ours"`. Hàng cuối là PUBLISHED với `source="published"`; số được đặt vào cột ghi trong `published_metric` (ví dụ `{"aspect": "aspect_macro_f1", "sentiment": "sentiment_macro_f1"}`, lấy từ `docs/metric_notes.md`). Nếu `published_metric is None` thì đặt vào cột `note` với chữ "không cùng định nghĩa: 84.48 / 63.06", các ô số để `"—"`.
  - `ablation_table(runs: pd.DataFrame, group: str, split: str = "dev") -> pd.DataFrame`: như `comparison_table` nhưng chỉ gồm các run có `group` trong `groups`, không có hàng published.
  - `sweep_table(runs: pd.DataFrame, prefix: str, split: str = "dev") -> pd.DataFrame`: các run có `name` bắt đầu bằng `prefix` (dùng cho E0).
  - `per_aspect_table(runs: pd.DataFrame, split: str, names: list[str]) -> pd.DataFrame`: hàng là khía cạnh, cột là `names` cộng `support`; giá trị là F1 khía cạnh trung bình theo seed, thang %, 2 chữ số.
  - `scripts/make_tables.py [--runs results/runs] [--out results/tables]` ghi các file sau, mỗi file cả `.csv` và `.md` (dùng `DataFrame.to_markdown(index=False)`):
    - `comparison_dev`, `comparison_test` (chỉ khi có run test)
    - `ablation_{group}_dev` cho mỗi group xuất hiện
    - `sweep_E0_{tfidf_lr,tfidf_svm}_dev` (tiền tố `tfidf_lr__C=` và `tfidf_svm__C=`)
    - `per_aspect_test` (các run có test).

    `published_metric` đọc từ một khối YAML trong `docs/metric_notes.md` có dạng ```` ```yaml published_metric: {...} ``` ````; không có khối đó thì dùng `None`.

- [ ] **Step 1: Viết test (fail).** Tạo JSON giả trong `tmp_path/runs`:
  - 2 run `phobert` với seed 42 và 43, `aspect_macro_f1` dev là 0.80 và 0.82; 1 run `tfidf` seed 42 có `groups: ["E1"]`.
  - Kỳ vọng ô `aspect_macro_f1` của `phobert` là `"81.00 ± 1.41"`, ô của `tfidf` là `"80.00"`, `n_seeds` lần lượt 2 và 1.
  - Hàng cuối có `source == "published"`. Khi `published_metric=None`, cột `note` chứa `"không cùng định nghĩa"`.
  - `comparison_table(runs, "test")` rỗng (trừ hàng published) khi không run nào có test.
  - `ablation_table(runs, "E1")` chỉ có `tfidf`.
  - `per_aspect_table` có 10 hàng.
- [ ] **Step 2: Chạy `pytest tests/test_report.py -v`**, kỳ vọng FAIL.
- [ ] **Step 3: Cài đặt.**
- [ ] **Step 4: Chạy test** (PASS), rồi `python scripts/make_tables.py`. Kỳ vọng có `results/tables/comparison_dev.md`, `ablation_E1_dev.md`, `ablation_E2_dev.md`, `ablation_E3_dev.md`, `ablation_E6_tfidf_dev.md`, `sweep_E0_tfidf_lr_dev.md`, mỗi bảng có hàng `tfidf_lr`.
- [ ] **Step 5: Commit** `feat: comparison, ablation and per-aspect tables` (gồm cả `results/tables/*.md|csv`).

---

### Task 8: Lõi transformer (head, loss, decode, trọng số lớp, ngưỡng)

**Phụ trách:** B

**Files:**
- Create: `src/absa/tiny.py`, `src/absa/transformer.py` (phần lõi; phần `TransformerAbsa` thêm ở Task 9)
- Test: `tests/test_transformer_core.py`

**Interfaces:**
- Consumes: `evaluate` (Task 2)
- Produces:
  - `build_tiny(texts: list[str], hidden: int = 32, layers: int = 1, heads: int = 2, max_positions: int = 300) -> tuple[RobertaModel, PreTrainedTokenizerFast]`. Tokenizer dùng `tokenizers` WordLevel, pre-tokenizer `Whitespace`, từ vựng lấy từ `texts`, các token đặc biệt `<s> <pad> </s> <unk>` (id 0–3), post-processor `TemplateProcessing("<s> $A </s>")`. Đặt `RobertaConfig.pad_token_id` bằng id của `<pad>`, `vocab_size = len(vocab)`. Không cần mạng.
  - `class AbsaHead(nn.Module)`, khởi tạo bằng `(encoder: PreTrainedModel, mode: Literal["joint", "split"], dropout: float = 0.1)`. `forward(input_ids, attention_mask) -> dict` dùng vector vị trí 0 của `last_hidden_state`, qua dropout:
    - `joint`: trả `{"logits": Tensor[B, 10, 4]}`, từ `Linear(h, 40)`.
    - `split`: trả `{"aspect_logits": Tensor[B, 10], "polarity_logits": Tensor[B, 10, 3]}`, từ `Linear(h, 10)` và `Linear(h, 30)`.
  - `absa_loss(outputs: dict, Y: Tensor[B, 10] (long), mode: str, class_weights: Tensor[10, 4] | None = None) -> Tensor`:
    - `joint`: trung bình theo 10 khía cạnh của `F.cross_entropy(logits[:, a], Y[:, a], weight=class_weights[a] if có else None)`.
    - `split`: `BCEWithLogits(aspect_logits, (Y > 0).float())` cộng CE của `polarity_logits` với nhãn `Y - 1`, **chỉ** trên các ô có `Y > 0`. Không có ô nào thì phần CE là `0`. Có `class_weights` ở mode split thì báo `ValueError("class_weights chỉ hỗ trợ mode joint")`.
  - `decode(outputs: dict, mode: str, threshold: float = 0.5) -> np.ndarray`, int8 `(B, 10)`. `joint`: argmax. `split`: `sigmoid(aspect_logits) > threshold` thì `argmax(polarity) + 1`, không thì 0.
  - `compute_class_weights(Y: np.ndarray) -> torch.Tensor`, shape `(10, 4)`, float32. Đếm theo khía cạnh và lớp, cộng 1 (làm trơn), lấy nghịch đảo, chuẩn hóa để trung bình mỗi hàng bằng 1, **rồi** kẹp vào [0.1, 10].
  - `THRESHOLD_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)`
  - `best_threshold(aspect_probs: np.ndarray, polarity_codes: np.ndarray, Y_true: np.ndarray, grid=THRESHOLD_GRID) -> float`: dự đoán là `np.where(aspect_probs > t, polarity_codes, 0)`; chọn `t` có `evaluate(...)["sentiment_macro_f1"]` cao nhất; bằng nhau thì chọn `t` gần 0.5 nhất.

- [ ] **Step 1: Viết test (fail).** Fixture `tiny = build_tiny(["pin trâu", "máy đẹp quá", "giá cao"])`.
  - `test_build_tiny_offline`: `tokenizer(["pin trâu"])["input_ids"][0][0] == tokenizer.convert_tokens_to_ids("<s>")`; `encoder.config.pad_token_id == tokenizer.pad_token_id`.
  - `test_joint_shapes`, `test_split_shapes`: với batch 2, shape như phần Interfaces.
  - `test_joint_loss_with_weights`: loss là scalar hữu hạn khi có `compute_class_weights(Y)`.
  - `test_split_loss_ignores_absent_aspects`: đổi `polarity_logits[:, absent]` thành giá trị ngẫu nhiên khác thì loss không đổi (`torch.allclose`).
  - `test_split_loss_no_present_aspect`: `Y` toàn 0 → loss hữu hạn và bằng phần BCE.
  - `test_split_class_weights_raises`.
  - `test_decode_joint_argmax`; `test_decode_split_threshold`: `aspect_logits` âm cho ra 0; dương và `polarity_logits` lớn nhất ở chỉ số 2 cho ra 3.
  - `test_class_weights_zero_column`: Y có cột STORAGE toàn 0 → shape `(10, 4)`, mọi giá trị hữu hạn và nằm trong [0.1, 10]; ở một cột có lớp 0 nhiều hơn hẳn lớp 1, `w[a, 0] < w[a, 1]`.
  - `test_best_threshold`: `Y_true` có 2 hàng với CAMERA=1; `aspect_probs` bằng 0.4 ở ô có CAMERA, 0.1 ở các ô khác; `polarity_codes` toàn 1. Kỳ vọng `best_threshold(...) == 0.35`.
- [ ] **Step 2: Chạy `pytest tests/test_transformer_core.py -v`**, kỳ vọng FAIL.
- [ ] **Step 3: Cài đặt** `tiny.py` và phần lõi của `transformer.py`.
- [ ] **Step 4: Chạy test**, kỳ vọng PASS (CPU, dưới 30 giây).
- [ ] **Step 5: Commit** `feat: transformer ABSA heads, losses, decoding and class weights`

---

### Task 9: Huấn luyện transformer, Colab, PhoBERT/ViSoBERT và E3–E6

**Phụ trách:** B (A chạy E3, E5 trên Colab)

**Files:**
- Create: `src/absa/training.py`, `configs/phobert_joint.yaml`, `configs/visobert_joint.yaml`, `configs/ablation/E3_phobert_strip_emoji.yaml`, `E4_phobert_weight.yaml`, `E5_phobert_star.yaml`, `E6_phobert_split.yaml`, `configs/smoke/tiny_transformer.yaml`, `notebooks/02_colab_train.ipynb`
- Modify: `src/absa/transformer.py` (thêm `TransformerAbsa`)
- Test: `tests/test_training.py`, `tests/test_transformer_model.py`

**Interfaces:**
- Consumes: `AbsaHead`, `absa_loss`, `decode`, `compute_class_weights`, `best_threshold`, `build_tiny` (Task 8); `Model` Protocol (Task 5); `evaluate` (Task 2)
- Produces:
  - `set_seed(seed: int) -> None`: đặt seed cho `random`, `numpy`, `torch` và `torch.cuda`.
  - `fit_loop(module: nn.Module, train_batches: Callable[[], Iterable[dict]], predict_dev: Callable[[], np.ndarray] | None, Y_dev: np.ndarray | None, loss_fn: Callable[[nn.Module, dict], Tensor], *, epochs: int, lr: float, weight_decay: float = 0.01, warmup_ratio: float = 0.1, patience: int = 2, fp16: bool = False, device: str = "cpu", grad_clip: float = 1.0, log: Callable[[str], None] = print) -> dict`
    - Dùng AdamW và lịch `get_linear_schedule_with_warmup` (số bước tính từ một lượt duyệt `train_batches()`).
    - fp16 chỉ bật khi `device` là cuda: `torch.autocast` + `torch.amp.GradScaler`.
    - Sau mỗi epoch, nếu có `predict_dev`, tính `evaluate(Y_dev, predict_dev())["sentiment_macro_f1"]`. Giữ bản sao `state_dict` tốt nhất (trên CPU). Dừng khi `patience` epoch liên tiếp không cải thiện. Cuối cùng nạp lại trọng số tốt nhất.
    - Trả `{"best_epoch": int (đếm từ 1), "best_dev_sentiment_macro_f1": float | None, "history": list[dict]}`.
  - `class TransformerAbsa(model_name: str, mode: Literal["joint", "split"] = "joint", max_length: int = 256, lr: float = 2e-5, batch_size: int = 16, eval_batch_size: int = 64, epochs: int = 5, class_weight: bool = False, fp16: bool = True, patience: int = 2, dropout: float = 0.1, seed: int = 42)`
    - `__init__`: `class_weight=True` cùng `mode="split"` thì `ValueError`; `max_length > 256` thì `ValueError`.
    - `model_name == "__tiny__"`: lúc `fit`, gọi `build_tiny(train_texts)`. Ngược lại dùng `AutoModel.from_pretrained` và `AutoTokenizer.from_pretrained`.
    - Token hóa theo từng lô với `truncation=True, max_length=max_length, padding=True`.
    - Sau `fit` ở mode split có dev: đặt `self.threshold = best_threshold(...)` trên dev. Không thì `0.5`.
    - `info`: `{"best_epoch", "best_dev_sentiment_macro_f1", "threshold", "history", "device"}`.
    - `save(path: Path)`: ghi `absa_config.json` (gồm `model_type: "transformer"`, `model_name`, `mode`, `max_length`, `threshold`, `dropout`, và `preprocess`, khóa này do `run_experiment` gán vào `model.preprocess_cfg` trước khi save), `model.pt` (`state_dict` của `AbsaHead`), `encoder.config.save_pretrained(path)` (tạo `config.json`, để Task 12 dựng lại encoder mà không cần mạng, kể cả với `__tiny__`) và `tokenizer.save_pretrained(path)`.
  - Sửa nhỏ `run_experiment` (Task 5): gán `model.preprocess_cfg = cfg["preprocess"]` trước `model.save`.

- [ ] **Step 1: Viết `tests/test_training.py` (fail).**
  - `test_set_seed_reproducible`: hai lần `set_seed(1); torch.rand(3)` cho kết quả bằng nhau.
  - `test_early_stopping_restores_best`: module là `nn.Linear(1, 1)`. `predict_dev` lần lượt trả các ma trận cho `sentiment_macro_f1` = 1.0, 0.0, 0.0, 0.0 (đúng hết, rồi sai hết). Mỗi lần gọi, lưu lại `module.weight.clone()`. Với `epochs=5, patience=2`: kỳ vọng `best_epoch == 1`, `len(history) == 3`, và trọng số cuối bằng bản đã lưu ở lần gọi thứ nhất.
  - `test_no_dev_runs_all_epochs`: `predict_dev=None, epochs=2` → `len(history) == 2`, `best_dev_sentiment_macro_f1 is None`.
- [ ] **Step 2: Viết `tests/test_transformer_model.py` (fail).** Dùng `model_name="__tiny__"`, `epochs=1`, `batch_size=4`, `fp16=False`.
  - `test_fit_predict_joint` và `test_fit_predict_split`: 8 văn bản → `predict` cho shape `(8, 10)`, int8. Ở split có dev: `info["threshold"] in THRESHOLD_GRID`.
  - `test_truncation_long_input`: `predict([" ".join(["pin"] * 2000)])` chạy không lỗi với `max_length=256`.
  - `test_class_weight_split_raises`; `test_max_length_over_256_raises`.
  - `test_save_writes_files(tmp_path)`: có `absa_config.json`, `model.pt`, `config.json` và file tokenizer; `absa_config.json` có `mode` và `threshold`.
- [ ] **Step 3: Chạy hai file test**, kỳ vọng FAIL.
- [ ] **Step 4: Cài đặt `training.py` và `TransformerAbsa`.** Thiết bị: `"cuda" if torch.cuda.is_available() else "cpu"`. Khi dự đoán thì gọi `module.eval()` và `torch.no_grad()`.
- [ ] **Step 5: Chạy toàn bộ test**, kỳ vọng PASS.
- [ ] **Step 6: Viết các file YAML.**

```yaml
# configs/phobert_joint.yaml
name: phobert_joint
seed: 42
groups: [E3, E4, E5, E6]
preprocess: {lowercase: false, strip_emoji: false, strip_punct: false, segment: vncorenlp, use_star: false}
model: {type: transformer, model_name: vinai/phobert-base-v2, mode: joint, max_length: 256,
        lr: 2.0e-5, batch_size: 16, epochs: 5, class_weight: false, fp16: true, patience: 2}
```

  Thay `max_length: 256` bằng giá trị đã ghi ở Task 4 Step 6.
  - Ablation (`base: ../phobert_joint.yaml`): `E3_phobert_strip_emoji` → `preprocess.strip_emoji: true`; `E4_phobert_weight` → `model.class_weight: true`; `E5_phobert_star` → `preprocess.use_star: true`; `E6_phobert_split` → `model.mode: split`. Mỗi file có `groups` là nhóm của chính nó.
  - `visobert_joint.yaml`: `base: phobert_joint.yaml`, `name: visobert_joint`, `preprocess.segment: null`, `model.model_name: uitnlp/visobert`, `max_length` lấy theo số đo ViSoBERT (≤ 256), không có groups.
  - `smoke/tiny_transformer.yaml`: `base: ../phobert_joint.yaml`, `name: tiny_transformer`, `preprocess.segment: null`, `model: {model_name: __tiny__, epochs: 1, batch_size: 8, fp16: false}`.
- [ ] **Step 7: Viết `notebooks/02_colab_train.ipynb`.** Các ô theo thứ tự:
  1. Kiểm tra GPU (`!nvidia-smi`).
  2. Mount Google Drive; đặt `DRIVE=/content/drive/MyDrive/nlp-absa-visfd`.
  3. `git clone` repo của nhóm (biến `REPO_URL` để trống, nhóm tự điền).
  4. `pip install -e . -r requirements.txt`, rồi `pip freeze > requirements.lock.colab.txt`.
  5. `!java -version`; nếu thiếu thì `apt-get install -y openjdk-17-jre-headless`.
  6. `python scripts/download_data.py`.
  7. Khôi phục `data/processed` từ Drive nếu đã có (`results/runs` đã nằm trong git, D13).
  8. `pytest -q`.
  9. Kiểm tra nạp được tokenizer: `AutoTokenizer.from_pretrained("vinai/phobert-base-v2")` và `"uitnlp/visobert"`.
  10. Chạy đo thời gian.
  11. Một ô chạy cho mỗi cấu hình.
  12. Sao lưu `results/`, `data/processed/` và `models/final_*` lên Drive sau mỗi lần chạy.
- [ ] **Step 8: Đo thời gian trên Colab T4.** Chạy `python scripts/run.py --config configs/phobert_joint.yaml --limit 500 --set model.epochs=1`. Ghi thời gian và VRAM vào `docs/decisions.md`, rồi ước lượng thời gian một lần chạy đầy đủ. Nếu một lần chạy hơn 60 phút: giảm `max_length` theo EDA trước, rồi mới tăng `batch_size` lên 32. Ghi lại quyết định.
- [ ] **Step 9: Chạy đầy đủ, seed 42, trên Colab:** `phobert_joint`, `E6_phobert_split`, `E4_phobert_weight`, `E3_phobert_strip_emoji`, `E5_phobert_star`, `visobert_joint`. Tức là 6 lần chạy; `phobert_joint` là nhánh gốc chung, không chạy lại cho từng khảo sát (D4). Sao lưu sau mỗi lần chạy. Kéo `results/runs/*.json` về máy, chạy `python scripts/make_tables.py`, kiểm tra có `ablation_E3_dev.md` đến `ablation_E6_dev.md`.
- [ ] **Step 10: Commit** `feat: transformer training loop, PhoBERT/ViSoBERT configs and Colab notebook` (gồm code, configs, notebook, `docs/decisions.md`, `results/tables/`).

---

### Task 10: Bi-LSTM + fastText (tái hiện xấp xỉ mốc của bài báo)

**Phụ trách:** A

**Files:**
- Create: `src/absa/bilstm.py`, `scripts/prepare_fasttext.py`, `configs/bilstm_fasttext.yaml`
- Test: `tests/test_bilstm.py`

**Interfaces:**
- Consumes: `absa_loss`, `decode` (Task 8, `mode="joint"`); `fit_loop`, `set_seed` (Task 9); `preprocess_df`, `load_split`, `load_config` (Task 1, 3, 5)
- Produces:
  - `PAD, UNK = 0, 1`; `build_vocab(texts: list[str], min_freq: int = 1) -> list[str]`: phần tử 0 là `"<pad>"`, 1 là `"<unk>"`, sau đó là token theo tần suất giảm dần (bằng nhau thì theo thứ tự chữ cái); tách token bằng `str.split()`.
  - `encode(texts: list[str], vocab: list[str], max_len: int) -> list[list[int]]`: token lạ thành `UNK`; cắt ở `max_len`; văn bản rỗng thành `[UNK]`.
  - `class BiLstmAbsa(embedding_path: str | None = None, vocab_path: str | None = None, emb_dim: int = 300, hidden: int = 256, layers: int = 1, dropout: float = 0.3, epochs: int = 20, lr: float = 1e-3, batch_size: int = 32, max_len: int = 128, freeze_embedding: bool = False, patience: int = 3, seed: int = 42)`
    - Có `embedding_path` thì nạp ma trận `.npy` và từ vựng `.json`, và `emb_dim` lấy theo ma trận. Không có thì gọi `build_vocab(train_texts)` và khởi tạo ngẫu nhiên.
    - Kiến trúc: `Embedding(padding_idx=0)` → dropout → `LSTM(bidirectional=True, batch_first=True)` qua `pack_padded_sequence` → max-pool có mask trên các vị trí thật → dropout → `Linear(2*hidden, 40)` → `(B, 10, 4)`. Loss là `absa_loss(..., "joint")`, huấn luyện bằng `fit_loop` (fp16 tắt).
    - `info`: `{"best_epoch", "best_dev_sentiment_macro_f1", "vocab_size", "oov_rate_dev"}`.
  - `scripts/prepare_fasttext.py --vec PATH_TO/cc.vi.300.vec.gz --config configs/bilstm_fasttext.yaml`
    - Tiền xử lý Train theo `preprocess` của cấu hình, rồi `build_vocab`.
    - Đọc file `.vec.gz` theo dòng, chỉ giữ từ có trong vocab.
    - Ghi `models/fasttext_vi_train_vocab.json` và `models/fasttext_vi_train_vocab.npy` (float32, shape `(len(vocab), 300)`). Hàng PAD bằng 0; từ không có vector thì lấy `normal(0, 0.1)` với seed 42.
    - In tỉ lệ phủ theo token và theo kiểu từ.
  - File fastText (khoảng 1,2 GB): `https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.vec.gz`. Tải trên Colab, không tải về máy local.

- [ ] **Step 1: Viết test (fail)** với embedding ngẫu nhiên, `emb_dim=8, hidden=8, epochs=1`:
  - `test_build_vocab_order`: `build_vocab(["b a a", "c"])[:3] == ["<pad>", "<unk>", "a"]`.
  - `test_encode_oov_and_empty`: `encode(["xyz", ""], vocab, 5) == [[1], [1]]`.
  - `test_bilstm_truncates`: một văn bản 500 token với `max_len=128` → `len(encode(...)[0]) == 128`, và `predict` chạy được.
  - `test_output_shape`: `predict` trả `(n, 10)` int8, giá trị ⊆ {0..3}.
  - `test_padding_invariance`: ở chế độ eval, logits của `"pin trâu"` khi chạy một mình bằng logits của nó khi chung lô với một văn bản 50 token (`torch.allclose`, `atol=1e-5`).
  - `test_load_pretrained_embedding(tmp_path)`: tạo `.npy` (5×8) và `.json` với vocab 5 từ → `fit` chạy, và `model.module.embedding.weight[2]` khớp hàng 2 của ma trận khi `freeze_embedding=True`.
- [ ] **Step 2: Chạy `pytest tests/test_bilstm.py -v`**, kỳ vọng FAIL.
- [ ] **Step 3: Cài đặt `bilstm.py` và `prepare_fasttext.py`.** Viết `configs/bilstm_fasttext.yaml`:

```yaml
name: bilstm_fasttext
seed: 42
preprocess: {lowercase: true, strip_emoji: false, strip_punct: false, segment: null, use_star: false}
model: {type: bilstm, embedding_path: models/fasttext_vi_train_vocab.npy, vocab_path: models/fasttext_vi_train_vocab.json,
        hidden: 256, layers: 1, dropout: 0.3, epochs: 20, lr: 1.0e-3, batch_size: 32, max_len: 128,
        freeze_embedding: false, patience: 3}
```

- [ ] **Step 4: Chạy test** (PASS). Trên Colab: tải fastText, chạy `python scripts/prepare_fasttext.py ...`, ghi tỉ lệ phủ vào `docs/decisions.md`, rồi `python scripts/run.py --config configs/bilstm_fasttext.yaml`. Sao lưu lên Drive.
- [ ] **Step 5: Commit** `feat: BiLSTM+fastText approximate reproduction baseline`

---

### Task 11: Chốt cấu hình, chạy 3 seed, đánh giá Test, phân tích lỗi

**Phụ trách:** cả hai (A: chạy Test và bảng; B: phân tích lỗi)

**Files:**
- Create: `src/absa/errors.py`, `notebooks/03_error_analysis.ipynb`, `configs/final/final_tfidf.yaml`, `final_bilstm.yaml`, `final_phobert.yaml`, `final_visobert.yaml`, `results/error_analysis/selected_examples.md`
- Test: `tests/test_errors.py`

**Interfaces:**
- Consumes: `run_experiment`, `assert_final_config` (Task 5); bảng ở Task 7; `confusion_by_aspect` (Task 2); `parse_label`, `labels_to_matrix`, `format_label` (Task 1)
- Produces:
  - `load_predictions(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]`: đọc CSV dự đoán, trả `(df, Y_true, Y_pred)`.
  - `collect_errors(df: pd.DataFrame, Y_true: np.ndarray, Y_pred: np.ndarray) -> pd.DataFrame`. Các cột: `index, comment, n_star, gold, pred` (chuỗi theo `format_label`), `missed, spurious, wrong_polarity` (`list[str]` theo thứ tự `ASPECTS`), `n_errors`, và `error_type, cause, fix` (để trống, nhóm điền tay). Chỉ giữ hàng có lỗi.
  - `ERROR_TYPES = ("khía cạnh ngầm", "khen chê lẫn lộn", "mỉa mai", "phủ định", "teencode/viết tắt", "số sao mâu thuẫn", "nghi nhãn sai", "khác")`
  - `sample_errors(errors: pd.DataFrame, n: int = 40, seed: int = 42) -> pd.DataFrame`.
    - Mỗi hàng có một nhóm chính: `wrong_polarity` nếu có, không thì `missed`, không thì `spurious`.
    - Lấy mẫu theo tỉ lệ nhóm, mỗi nhóm không rỗng ít nhất 1 hàng; tổng không quá `n`.
    - Cùng `seed` thì cùng kết quả. Nếu `len(errors) <= n` thì trả toàn bộ.
  - `plot_confusions(Y_true: np.ndarray, Y_pred: np.ndarray, aspects: list[str], path: Path) -> None`: lưới heatmap 4×4 (nhãn None/Pos/Neu/Neg), tiêu đề mỗi ô ghi tên khía cạnh và `support`; dùng backend `Agg`.

- [ ] **Step 1: Viết test (fail)**
  - `test_collect_errors_categories`: gold `{CAMERA: Positive, PRICE: Negative}`, dự đoán `{CAMERA: Negative, BATTERY: Positive}`, thêm một hàng đúng hoàn toàn. Kỳ vọng còn 1 hàng; `missed == ["PRICE"]`, `spurious == ["BATTERY"]`, `wrong_polarity == ["CAMERA"]`, `n_errors == 3`.
  - `test_sample_errors_deterministic_and_stratified`: 30 hàng lỗi giả (10 mỗi nhóm), `n=9` → 9 hàng; đủ cả 3 nhóm; gọi hai lần cho cùng chỉ mục. Với `n=100` thì trả 30 hàng.
  - `test_load_predictions_roundtrip(tmp_path)`: ghi CSV có `gold` và `pred` dạng `format_label` → `Y_true` khớp `labels_to_matrix`.
  - `test_plot_confusions_writes_png(tmp_path)`: file tồn tại và kích thước > 0.
- [ ] **Step 2: Chạy `pytest tests/test_errors.py -v`**, kỳ vọng FAIL.
- [ ] **Step 3: Cài đặt `errors.py`.** Chạy test, kỳ vọng PASS.
- [ ] **Step 4: Chốt cấu hình, chỉ dựa trên `results/tables/comparison_dev.md` và các bảng ablation.**
  - `final_tfidf`: chọn giữa `tfidf_lr`, `tfidf_svm` và các nhánh E1–E3, E6_tfidf, theo `dev.sentiment_macro_f1`; bằng nhau thì xét `aspect_macro_f1`.
  - `final_bilstm`: `bilstm_fasttext`.
  - `final_phobert`: chọn trong `phobert_joint`, E3, E4, E6. **Không** chọn E5 (số sao), vì spec E5 chỉ báo cáo riêng.
  - `final_visobert`: `visobert_joint`.

  Mỗi file có dạng `base: ../<cấu hình thắng>.yaml`, `name: final_<x>`, `groups: [final]`. `final_phobert` thêm `save: true` (để làm demo); nếu ViSoBERT tốt hơn trên dev thì đặt `save: true` cho `final_visobert`. Ghi lý do chọn kèm số dev vào `docs/decisions.md`. **Commit các cấu hình này trước khi chạy Test**, để lịch sử git chứng minh không chọn lại sau khi đã xem Test.
- [ ] **Step 5: Chạy Test, lần duy nhất.**
  - Máy local: `python scripts/run.py --config configs/final/final_tfidf.yaml --seeds 42 43 44 --eval-test`. TF-IDF+LR không có yếu tố ngẫu nhiên nên std = 0; ghi chú điều này dưới bảng.
  - Colab: lần lượt `final_bilstm`, `final_phobert`, `final_visobert` với `--seeds 42 43 44 --eval-test`. Sao lưu sau mỗi seed.
- [ ] **Step 6: Chạy `python scripts/make_tables.py`.** Mở `results/tables/comparison_test.md`, kiểm tra: đủ 4 mô hình `final_*`, mỗi mô hình `n_seeds == 3`, và có hàng published (đúng cột, hoặc ghi "không cùng định nghĩa", theo `docs/metric_notes.md`). Kiểm tra `per_aspect_test.md` có cột `support` (STORAGE ở Test là 27).
- [ ] **Step 7: Phân tích lỗi trên Dev** trong `notebooks/03_error_analysis.ipynb`:
  1. Nạp `results/predictions/final_phobert__seed42__dev.csv` (hoặc của mô hình transformer tốt nhất).
  2. `collect_errors` → ghi `results/error_analysis/dev_errors_final_phobert.csv`.
  3. `sample_errors(n=40)` → điền tay `error_type` (một giá trị trong `ERROR_TYPES`), `cause` và `fix` → ghi `dev_errors_sample_labeled.csv`. Thống kê số lượng theo `error_type`.
  4. So sánh chéo: đếm số bình luận dev mà TF-IDF sai nhưng PhoBERT đúng, và ngược lại. Ghi `results/tables/cross_errors_dev.csv`.
  5. Chọn 8–10 ví dụ tiêu biểu, phủ ít nhất 5 loại lỗi. Ghi `results/error_analysis/selected_examples.md` thành bảng: bình luận, n_star, nhãn đúng, nhãn dự đoán, loại lỗi, nguyên nhân giả định, hướng khắc phục.
- [ ] **Step 8: Vẽ confusion matrix trên Test** cho mô hình transformer tốt nhất (seed 42; chỉ để báo cáo, không dùng để chọn gì): `plot_confusions(..., ["GENERAL", "PERFORMANCE", "BATTERY", "FEATURES", "STORAGE"], Path("results/figures/confusion_test_final_phobert.png"))`.
- [ ] **Step 9: Chạy `pytest -q`**, kỳ vọng toàn bộ PASS.
- [ ] **Step 10: Commit** `feat: final runs, test evaluation and error analysis` (gồm `src/absa/errors.py`, notebook, `results/tables`, `results/figures`, `results/error_analysis`, `docs/decisions.md`).

---

### Task 12: Demo Gradio

**Phụ trách:** A

**Files:**
- Create: `src/absa/inference.py`, `demo/app.py`
- Test: `tests/test_inference.py`

**Interfaces:**
- Consumes: checkpoint `models/final_<x>__seed42/` (Task 9 `save`, Task 11); `normalize`, `segment` (Task 3); `AbsaHead`, `decode` (Task 8); `matrix_to_labels` (Task 1); `aspect_polarity_table` (Task 4)
- Produces:
  - `ASPECT_VI = {"SCREEN": "Màn hình", "CAMERA": "Camera", "FEATURES": "Tính năng", "BATTERY": "Pin", "PERFORMANCE": "Hiệu năng", "STORAGE": "Bộ nhớ", "DESIGN": "Thiết kế", "PRICE": "Giá", "GENERAL": "Tổng quan", "SER&ACC": "Dịch vụ & phụ kiện"}`
  - `COLORS = {"Positive": "#1a7f37", "Neutral": "#6e7781", "Negative": "#cf222e"}`
  - `class Predictor`:
    - `from_dir(path: Path) -> Predictor`: đọc `absa_config.json`; `preprocess.use_star` bằng true thì báo `ValueError("Demo không có số sao")`; nạp tokenizer (`AutoTokenizer.from_pretrained(path)`), dựng encoder bằng `AutoModel.from_config(AutoConfig.from_pretrained(path))` từ `config.json` trong thư mục checkpoint (không cần mạng), bọc bằng `AbsaHead(mode, dropout)`, rồi nạp `model.pt`.
    - `predict(text: str) -> dict[str, str]`: nếu `normalize(text, strip_emoji=True) == ""` thì trả `{}` ngay; ngược lại tiền xử lý đúng như `preprocess` đã lưu.
    - `predict_batch(texts: list[str], batch_size: int = 32) -> list[dict[str, str]]`.
  - `render_html(text: str, pred: dict[str, str]) -> str`: một `<div>` chứa văn bản đã `html.escape`, kèm các nhãn `<span style="background:{màu}">{ASPECT_VI[a]}: {Tích cực|Trung tính|Tiêu cực}</span>` theo thứ tự `ASPECTS`. Không có nhãn thì hiện `"Không phát hiện khía cạnh nào"`.
  - `summarize_predictions(preds: list[dict[str, str]]) -> pd.DataFrame` = `aspect_polarity_table(pd.DataFrame({"aspects": preds}))`.
  - `demo/app.py --model PATH [--share] [--max-rows 2000]`: `gr.Blocks` có 2 tab.
    - Tab 1: ô nhập văn bản → HTML. Có sẵn 5 ví dụ lấy từ `selected_examples.md` (dán chuỗi vào code).
    - Tab 2: tải lên CSV → kiểm tra có cột `comment` (thiếu thì `gr.Error("File CSV cần có cột 'comment'")`) → cắt còn `--max-rows` dòng và ghi chú nếu bị cắt → bảng tổng hợp và biểu đồ cột chồng khen/chê theo khía cạnh (matplotlib) → nêu khía cạnh bị chê nhiều nhất.

- [ ] **Step 1: Viết test (fail)**, không cần model thật:
  - `test_render_negative_color`: `render_html("pin yếu", {"BATTERY": "Negative"})` chứa `"Pin"` và `"#cf222e"`.
  - `test_render_empty`: `render_html("", {})` bắt đầu bằng `"<div"`, kết thúc bằng `"</div>"`, và chứa `"Không phát hiện"`.
  - `test_render_escapes_script`: `render_html("<script>alert(1)</script>", {})` không chứa `"<script>"` mà chứa `"&lt;script&gt;"`.
  - `test_predict_empty_and_emoji_only`: tạo `Predictor` qua `object.__new__`, gán mô hình giả (gọi đến là báo lỗi) → `predict("")` và `predict("😍😍")` đều trả `{}` mà không gọi mô hình.
  - `test_summarize_predictions`: `[{"BATTERY": "Negative"}, {"BATTERY": "Positive"}, {}]` → `.loc["BATTERY", "Total"] == 2`.
  - `test_from_dir_roundtrip(tmp_path)`: train `TransformerAbsa("__tiny__", epochs=1)`, gán `preprocess_cfg`, `save`, rồi `Predictor.from_dir` → `predict("pin trâu")` trả dict có khóa ⊆ `ASPECTS`, và khớp với `model.predict(["pin trâu"])` sau khi đổi về dict.
- [ ] **Step 2: Chạy `pytest tests/test_inference.py -v`**, kỳ vọng FAIL.
- [ ] **Step 3: Cài đặt `inference.py` và `demo/app.py`.**
- [ ] **Step 4: Chạy test** (PASS). Chạy thật:
  - Trên Colab: `python demo/app.py --model models/final_phobert__seed42 --share`.
  - Trên máy local: chỉ chạy nếu có Java (vì PhoBERT cần VnCoreNLP) và đã tải checkpoint từ Drive.

  Thử 5 câu trong `selected_examples.md`, một câu rỗng, một câu toàn emoji, một CSV thiếu cột `comment`, và `results/predictions/...test.csv` (có cột `comment`). Chụp màn hình vào `results/figures/demo_*.png` để đưa vào báo cáo.
- [ ] **Step 5: Commit** `feat: Gradio demo with aspect highlighting and CSV summary`

---

### Task 13: README, `run_all.py`, kiểm tra tái lập

**Phụ trách:** B

**Files:**
- Create: `README.md`, `scripts/run_all.py`, `requirements.lock.txt` (cập nhật)
- Test: `tests/test_smoke_pipeline.py`

**Interfaces:**
- Consumes: mọi script trước đó
- Produces: `python scripts/run_all.py [--smoke] [--workdir PATH]`.
  - Chế độ đầy đủ: tải dữ liệu → chạy notebook EDA → chạy các cấu hình TF-IDF (base, svm, E1, E2, E3, E6_tfidf) → `make_tables` → in hướng dẫn chạy phần GPU bằng `notebooks/02_colab_train.ipynb`.
  - `--smoke`: dùng `configs/smoke/tfidf_smoke.yaml` và `configs/smoke/tiny_transformer.yaml` với `--limit 200`, trên CPU, rồi `make_tables`, bỏ qua EDA.
  - Mỗi bước gọi `subprocess.run([sys.executable, ...], check=True)` và in tên bước kèm thời gian. Lỗi thì dừng và in bước bị lỗi.
  - `--workdir`: mọi output (`results`, `data/processed`) ghi vào thư mục này; dữ liệu thô vẫn đọc từ `data/raw` của repo. Cần thêm tham số `--out-dir` và `--cache-dir` cho `scripts/run.py`, và `--runs`, `--out` cho `make_tables.py`.

- [ ] **Step 1: Viết test (fail).** `test_smoke_pipeline` (skip nếu chưa có `data/raw/Train.csv`): chạy `subprocess.run([sys.executable, "scripts/run_all.py", "--smoke", "--workdir", str(tmp_path)], check=True, timeout=900)`. Kỳ vọng có `tmp_path/results/tables/comparison_dev.md` và trong đó có `tfidf_smoke` và `tiny_transformer`.
- [ ] **Step 2: Chạy test**, kỳ vọng FAIL.
- [ ] **Step 3: Cài đặt `run_all.py`** cùng các tham số `--out-dir`, `--cache-dir`, `--runs`, `--out`.
- [ ] **Step 4: Viết `README.md`.** Bố cục:
  1. Mục tiêu và hai bài toán con.
  2. Dữ liệu: nguồn, trích dẫn BibTeX của Phan et al. (2021), ghi chú "chỉ dùng cho nghiên cứu", không commit dữ liệu.
  3. Cấu trúc thư mục (rút gọn từ plan).
  4. Cài đặt cho Windows (python.org, venv, torch CPU, Java) và Colab.
  5. Chạy từng bước, mỗi lệnh một khối `bash`.
  6. Bảng ánh xạ "bảng/hình trong báo cáo → script/notebook → file kết quả", phủ mọi file trong `results/tables` và `results/figures`.
  7. Thời gian chạy đo được (lấy từ `docs/decisions.md`).
  8. Giới hạn đã biết (emoticon dạng chữ, STORAGE ít mẫu, cách tính F1 khác bài báo nếu có).
  9. Phạm vi đã dùng công cụ AI.
  10. Phân công.
- [ ] **Step 5: Chạy `pytest -q`**, kỳ vọng toàn bộ PASS.
- [ ] **Step 6: Kiểm tra tái lập trên máy khác (A làm, theo đúng README, không hỏi B).** Clone repo vào thư mục mới, cài đặt, rồi chạy `python scripts/run_all.py`. So `results/tables/comparison_dev.csv` mới với bản đã commit: các hàng TF-IDF phải giống hệt (sai lệch bằng 0). Chỗ nào trong README gây vướng thì sửa README.
- [ ] **Step 7: Chạy `pip freeze > requirements.lock.txt`**, rồi commit `docs: README, run_all and reproducibility check`

---

### Task 14: Báo cáo PDF và slide (không phải code)

**Phụ trách:** cả hai, theo cột "Phụ trách" ở spec mục 7 (tuần 8–10)

**Files:**
- Create: `report/NLP_BTL_MaNhom.pdf` (đổi `MaNhom` thành mã nhóm), `report/slides.pdf`

- [ ] **Step 1: Dàn ý theo spec mục 9**, dài 20–30 trang chưa tính phụ lục. Mỗi mục ghi rõ bảng/hình lấy từ file nào trong `results/` (dùng bảng ánh xạ trong README):
  - Dữ liệu: `eda_*`.
  - Thực nghiệm: E0–E6 lấy từ `ablation_*_dev.md` và `sweep_E0_*`.
  - Kết quả: `comparison_test.md`, `per_aspect_test.md`, confusion matrix, `cross_errors_dev.csv`.
  - Phân tích lỗi: `selected_examples.md` và thống kê `error_type`.
- [ ] **Step 2: Trả lời RQ1–RQ3** bằng số liệu có trong bảng: RQ1 so `final_phobert` với `final_tfidf` trên Test; RQ2 dùng E1, E2, E4 và F1 theo khía cạnh của STORAGE; RQ3 dùng thống kê `error_type`. Báo cáo kết quả E5 trong một mục riêng.
- [ ] **Step 3: Kiểm tra chéo số liệu.** Người không viết mục nào thì kiểm tra mục đó: mỗi con số trong báo cáo phải đối chiếu được với một file `results/` (đánh dấu ✓ trong bản nháp).
- [ ] **Step 4: Phụ lục:** phân công (dẫn link commit), phạm vi dùng AI, `docs/decisions.md` rút gọn.
- [ ] **Step 5: Slide 10 phút** theo spec mục 10. Demo chạy trực tiếp bằng Colab `--share`; chuẩn bị video quay sẵn để dự phòng khi mất mạng. Tập vấn đáp chéo: mỗi người giải thích được phần của người kia.
- [ ] **Step 6: Commit** `docs: final report and slides`

---

## Lịch chạy GPU (Colab) và ước lượng

| Lượt | Cấu hình | Số lần chạy |
|---|---|---|
| Đo thời gian | `phobert_joint --limit 500 --set model.epochs=1` | 1 |
| Dev, seed 42 | `phobert_joint`, E3/E4/E5/E6 PhoBERT, `visobert_joint`, `bilstm_fasttext` | 7 |
| Test, 3 seed | `final_phobert`, `final_visobert`, `final_bilstm` | 9 |

Tổng cộng khoảng 17 lần chạy GPU. Thời gian mỗi lần đo ở Task 9 Step 8; cần đủ dư để chia ra nhiều phiên Colab. Sao lưu lên Drive sau **mỗi** lần chạy, vì phiên Colab có thể bị ngắt.
