# Phân tích cảm xúc theo khía cạnh trên UIT-ViSFD

Bài tập lớn môn Xử lý ngôn ngữ tự nhiên. Thành viên: A = [@CungDoHaiPhong279](https://github.com/CungDoHaiPhong279), B = [@Longdaica11](https://github.com/Longdaica11).

## 1. Mục tiêu

Với mỗi bình luận điện thoại tiếng Việt, hệ thống trả về tập cặp (khía cạnh, cảm xúc) trên 10 khía cạnh: SCREEN, CAMERA, FEATURES, BATTERY, PERFORMANCE, STORAGE, DESIGN, PRICE, GENERAL, SER&ACC. Bài toán gồm hai bài toán con:

1. **Phát hiện khía cạnh:** bình luận có nhắc tới khía cạnh nào (10 bài toán nhị phân).
2. **Phân loại cảm xúc:** với mỗi khía cạnh được nhắc, cảm xúc là Positive, Neutral hay Negative.

Nhóm so sánh baseline TF-IDF + Logistic Regression với PhoBERT, ViSoBERT và Bi-LSTM + fastText, khảo sát E0–E6, phân tích lỗi và làm demo Gradio. Kết quả trên Test (3 seed, macro-F1 %):

| Mô hình | Khía cạnh | Cảm xúc |
|---|---|---|
| PhoBERT (có trọng số lớp) | 88,21 ± 0,25 | 68,45 ± 0,86 |
| ViSoBERT | 84,88 ± 0,34 | 56,60 ± 1,40 |
| TF-IDF + LR | 81,77 ± 0,00 | 52,85 ± 0,00 |
| Bi-LSTM + fastText | 81,16 ± 0,92 | 56,60 ± 0,27 |

Số của bài báo gốc (84,48 / 63,06) dùng cách tính khác nên không so trực tiếp được; xem [docs/metric_notes.md](docs/metric_notes.md).

## 2. Dữ liệu

UIT-ViSFD gồm 11.122 bình luận (Train 7.786 · Dev 1.112 · Test 2.224). `scripts/download_data.py` tải bộ dữ liệu từ <https://github.com/LuongPhan/UIT-ViSFD> và kiểm tra SHA-256. Dữ liệu **chỉ dùng cho nghiên cứu** và **không** được commit vào repo này (`data/` nằm trong `.gitignore`).

```bibtex
@InProceedings{10.1007/978-3-030-82147-0_53,
  author    = "Luc Phan, Luong and Huynh Pham, Phuc and Thi-Thanh Nguyen, Kim
               and Khai Huynh, Sieu and Thi Nguyen, Tham and Thanh Nguyen, Luan
               and Van Huynh, Tin and Van Nguyen, Kiet",
  editor    = "Qiu, Han and Zhang, Cheng and Fei, Zongming and Qiu, Meikang and Kung, Sun-Yuan",
  title     = "SA2SL: From Aspect-Based Sentiment Analysis to Social Listening System
               for Business Intelligence",
  booktitle = "Knowledge Science, Engineering and Management",
  year      = "2021",
  publisher = "Springer International Publishing",
  address   = "Cham",
  pages     = "647--658",
  isbn      = "978-3-030-82147-0"
}
```

## 3. Cấu trúc thư mục

```text
src/absa/        gói Python: data, preprocess, baseline, transformer, bilstm, evaluate, report, inference…
scripts/         download_data.py, run.py, run_all.py, make_tables.py, prepare_fasttext.py
configs/         YAML thí nghiệm: gốc, ablation/ (E1–E6), final/ (4 mô hình chốt), smoke/
notebooks/       01_eda, 02_colab_train (GPU), 03_error_analysis
demo/app.py      demo Gradio
tests/           pytest
results/         runs/*.json (có commit), tables/, figures/, error_analysis/; predictions/ không commit
docs/            spec, plan, decisions.md, metric_notes.md, TEAM.md
data/, models/   dữ liệu và checkpoint, không commit
```

Thành viên mới đọc theo thứ tự: [docs/TEAM.md](docs/TEAM.md) → [CONTRIBUTING.md](CONTRIBUTING.md) → [AGENTS.md](AGENTS.md) → [plan](docs/superpowers/plans/2026-11-13-absa-visfd.md), [spec](docs/spec/phac_thao_du_an_ABSA_UIT-ViSFD.md), [quyết định](docs/decisions.md). Tiến độ ở [Issues](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/issues).

## 4. Cài đặt

### Windows

1. Cài Python 3.11 từ <https://www.python.org> (không dùng Python của MSYS2).
2. Cài Java 17 trở lên (VnCoreNLP cần Java để tách từ), ví dụ Temurin từ <https://adoptium.net>. Kiểm tra bằng `java -version`.
3. Trong PowerShell, từ thư mục gốc repo:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e . -r requirements.txt -r requirements-dev.txt
python scripts/download_data.py
pytest -q
```

Phiên bản trên máy A ghi ở `requirements.lock.txt`. Phiên bản thật của từng lần chạy, kể cả trên Colab (Python 3.13, torch 2.14.1+cu130, transformers 5.22.1), nằm trong khóa `versions` của mỗi `results/runs/*.json`.

### Colab (phần GPU)

Mở `notebooks/02_colab_train.ipynb` trên Colab, chọn GPU T4 rồi chạy lần lượt các ô. Notebook tự clone repo, cài thư viện, tải dữ liệu, mount Drive để sao lưu `results/runs` và checkpoint sau mỗi lần chạy.

## 5. Chạy từng bước

Toàn bộ phần CPU (tải dữ liệu, EDA, 8 cấu hình TF-IDF, tạo bảng; trừ lưới C của E0, lệnh ở dưới) bằng một lệnh:

```bash
python scripts/run_all.py
```

Chạy thử nhanh (200 dòng, TF-IDF và transformer tí hon, không cần GPU), ghi kết quả ra thư mục riêng:

```bash
python scripts/run_all.py --smoke --workdir tmp_smoke
```

Từng bước riêng lẻ:

```bash
python scripts/download_data.py
```

```bash
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output-dir tmp_nb
```

```bash
python scripts/run.py --config configs/tfidf_lr_base.yaml
```

```bash
python scripts/run.py --config configs/ablation/E1_char_25.yaml configs/ablation/E2_word.yaml
```

```bash
python scripts/run.py --config configs/tfidf_lr_base.yaml --set model.C=4
```

Lệnh trên là một điểm của E0; `--set` tự đặt tên lần chạy là `tfidf_lr__C=4`. Cả lưới E0 (LR: C = 0,25 đến 1024, nhân đôi mỗi bước; SVM: C = 0,25 đến 32) trong PowerShell:

```powershell
foreach ($c in 0.25,0.5,1,2,4,8,16,32,64,128,256,512,1024) { python scripts/run.py --config configs/tfidf_lr_base.yaml --set model.C=$c }
foreach ($c in 0.25,0.5,1,2,4,8,16,32) { python scripts/run.py --config configs/tfidf_svm.yaml --set model.C=$c }
```

```bash
python scripts/make_tables.py
```

Các mô hình GPU chạy trên Colab (GPU T4). `notebooks/02_colab_train.ipynb` dựng môi trường và chạy 6 cấu hình PhoBERT/ViSoBERT trên Dev, ví dụ:

```bash
python scripts/run.py --config configs/phobert_joint.yaml
```

Bi-LSTM chạy bằng các lệnh dưới đây trong cùng phiên Colab, sau ô cài đặt của notebook. Tải vector fastText (khoảng 1,2 GB, nên tải trên Colab):

```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.vec.gz
```

Bi-LSTM chạy bằng các lệnh dưới đây trong cùng phiên Colab, sau ô cài đặt của notebook. Tải vector fastText (khoảng 1,2 GB, nên tải trên Colab):

```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.vec.gz
```

```bash
python scripts/prepare_fasttext.py --vec cc.vi.300.vec.gz --config configs/bilstm_fasttext.yaml
```

```bash
python scripts/run.py --config configs/bilstm_fasttext.yaml
```

Phân tích lỗi: chạy `notebooks/03_error_analysis.ipynb`. Notebook cần `results/predictions/`: mỗi lần chạy ghi dự đoán vào thư mục này, nhưng thư mục không được commit vì chứa nguyên văn dữ liệu UIT-ViSFD; tạo lại bằng các lệnh final ở cuối mục này, hoặc xin nhóm bản sao (Drive nhóm `nlp-absa-visfd/backups/`). Demo:

```bash
python demo/app.py --model models/final_phobert__seed42
```

Checkpoint `final_phobert__seed42` (540 MB) không có trên Git. Nhóm gửi link Drive (quyền xem) của thư mục `models/final_phobert__seed42/` kèm bài nộp; hoặc tự tạo lại trên Colab bằng `python scripts/run.py --config configs/final/final_phobert.yaml --seeds 42` (cấu hình có `save: true`, khoảng 12 phút trên T4; lần chạy có `--limit` không ghi checkpoint). Trên Colab thêm `--share` để có đường link công khai. Tab tổng hợp CSV chỉ phân tích 2.000 dòng đầu; đổi bằng `--max-rows`.

Quy tắc: tập Test chỉ được đánh giá **một lần** (`--eval-test`, đã chạy ở PR #49), không chạy lại. Lệnh đã dùng, ghi lại để truy vết (TF-IDF trên máy cá nhân, ba mô hình còn lại trên Colab T4):

```bash
python scripts/run.py --config configs/final/final_tfidf.yaml configs/final/final_bilstm.yaml configs/final/final_phobert.yaml configs/final/final_visobert.yaml --seeds 42 43 44 --eval-test
``` Lệnh đã dùng, ghi lại để truy vết (TF-IDF trên máy cá nhân, ba mô hình còn lại trên Colab T4):

```bash
python scripts/run.py --config configs/final/final_tfidf.yaml configs/final/final_bilstm.yaml configs/final/final_phobert.yaml configs/final/final_visobert.yaml --seeds 42 43 44 --eval-test
```

## 6. Bảng và hình trong báo cáo

| Nội dung trong báo cáo | Tạo bởi | File kết quả |
|---|---|---|
| Thống kê dữ liệu, phân bố cảm xúc theo khía cạnh | `notebooks/01_eda.ipynb` | `results/tables/eda_aspect_polarity_{train,dev,test}.csv`, `eda_aspects_per_comment.csv`, `eda_length.csv`, `eda_subword_len.csv`, `eda_star_vs_polarity.csv`, `eda_teencode.csv`, `eda_emoji.json`, `eda_duplicates.json` |
| Hình EDA | `notebooks/01_eda.ipynb` | `results/figures/eda_aspect_freq.png`, `eda_aspect_polarity_stacked.png`, `eda_length_hist.png`, `eda_star_polarity_heatmap.png` |
| So sánh mô hình trên Dev | `scripts/make_tables.py` | `results/tables/comparison_dev.{md,csv}` |
| So sánh mô hình trên Test (RQ1) | `scripts/make_tables.py` | `results/tables/comparison_test.{md,csv}` |
| F1 theo khía cạnh trên Test | `scripts/make_tables.py` | `results/tables/per_aspect_test.{md,csv}` |
| E0: dò tham số C | `scripts/make_tables.py` | `results/tables/sweep_E0_tfidf_lr_dev.*`, `sweep_E0_tfidf_svm_dev.*` |
| E1–E6 (RQ2) | `scripts/make_tables.py` | `results/tables/ablation_E{1..6}_dev.*`, `ablation_E6_tfidf_dev.*` |
| Chọn cấu hình cuối | `scripts/make_tables.py` | `results/tables/ablation_final_dev.*` |
| Ma trận nhầm lẫn PhoBERT | `notebooks/03_error_analysis.ipynb` | `results/figures/confusion_test_final_phobert.png` |
| So lỗi chéo TF-IDF với PhoBERT | `notebooks/03_error_analysis.ipynb` | `results/tables/cross_errors_dev.csv` |
| Phân tích lỗi (RQ3) | `notebooks/03_error_analysis.ipynb` (gán nhãn lỗi bằng tay) | `results/error_analysis/dev_errors_sample_labeled.csv`, `selected_examples.md`; danh sách đủ 617 lỗi `dev_errors_final_phobert.csv` chỉ tạo ra trên máy, không đưa lên Git (chứa nguyên văn dữ liệu) |
| Ảnh demo | `demo/app.py` (chụp màn hình) | `results/figures/demo_single_comment.png`, `demo_csv_summary.png` |
| Dự đoán minh họa | `notebooks/03_error_analysis.ipynb`, `demo/app.py` | `results/error_analysis/selected_examples.md` (9 ví dụ: nhãn đúng và dự đoán), `dev_errors_sample_labeled.csv` (40 mẫu), ảnh demo ở trên; dự đoán đầy đủ `results/predictions/` không commit vì chứa nguyên văn dữ liệu |
| Dự đoán minh họa | `notebooks/03_error_analysis.ipynb`, `demo/app.py` | `results/error_analysis/selected_examples.md` (9 ví dụ: nhãn đúng và dự đoán), `dev_errors_sample_labeled.csv` (40 mẫu), ảnh demo ở trên; dự đoán đầy đủ `results/predictions/` không commit vì chứa nguyên văn dữ liệu |

Không sửa tay `results/tables/*`; chạy lại `make_tables.py`.

## 7. Thời gian chạy đo được

| Việc | Máy | Thời gian |
|---|---|---|
| `run_all.py --smoke` | Laptop CPU của A | khoảng 1 phút |
| `run_all.py` (phần CPU đầy đủ: EDA + 8 cấu hình TF-IDF) | Laptop CPU của A | khoảng 7 phút (đo 6 phút 48 giây) |
| Một lần train TF-IDF + LR | Laptop CPU của A | khoảng 12 giây |
| PhoBERT, tối đa 10 epoch | Colab T4 | khoảng 11–12 phút (~70 giây/epoch) |
| ViSoBERT | Colab T4 | khoảng 13 phút |
| Bi-LSTM + fastText | Colab T4 | khoảng 1,5 phút train |
| Demo: nạp mô hình / một câu | Laptop CPU của A | 4 giây / dưới 1 giây |

Chi tiết trong [docs/decisions.md](docs/decisions.md).

## 8. Giới hạn đã biết

- Emoticon dạng chữ như `:)` không được xử lý; `strip_emoji` chỉ bỏ emoji Unicode.
- STORAGE rất ít mẫu (27 trên Test), nên F1 của nó dao động mạnh và kéo macro-F1 lên xuống nhiều điểm.
- Chỉ PhoBERT được khảo sát trọng số lớp (E4); chênh lệch PhoBERT–ViSoBERT một phần do trọng số lớp.
- TF-IDF + LR tất định nên 3 seed cho cùng kết quả (std = 0).
- Bi-LSTM chỉ tái hiện xấp xỉ mô hình của bài báo.
- Cách tính F1 của bài báo khác của nhóm (macro có tính `Others`, cách tính F1 cảm xúc không được nêu), nên không so trực tiếp. Xem [docs/metric_notes.md](docs/metric_notes.md).
- Số sao không dùng trong demo, vì người dùng chỉ nhập văn bản.

## 9. Phạm vi dùng công cụ AI

Nhóm dùng trợ lý AI để hỗ trợ viết code, test và tài liệu: A dùng Claude Code, B dùng Codex. Quy tắc cho trợ lý nằm ở [AGENTS.md](AGENTS.md): làm theo plan, viết test trước, mọi thay đổi qua Pull Request và CI. Thiết kế thí nghiệm, quyết định trong `docs/decisions.md`, việc chạy GPU trên Colab, gán nhãn lỗi và duyệt PR do thành viên nhóm làm và kiểm tra. Phụ lục báo cáo ghi chi tiết.

## 10. Phân công

| Phần | Người làm |
|---|---|
| Khung dự án, tải dữ liệu, tiền xử lý, EDA (Task 1, 3, 4) | A |
| Hàm đánh giá, cấu hình và bộ chạy (Task 2, 5) | B |
| Baseline TF-IDF, bảng tự động (Task 6, 7) | B làm, A sửa sau review |
| Lõi transformer, huấn luyện, Bi-LSTM (Task 8, 9, 10) | A làm code; B và A chạy Colab |
| Chốt cấu hình, Test, phân tích lỗi (Task 11) | Cả hai |
| Demo Gradio (Task 12) | A |
| README, `run_all.py` (Task 13) | A (thay B) |
| Báo cáo và slide (Task 14) | Cả hai |

Báo cáo `NLP_BTL_MaNhom.pdf` và slide nộp riêng theo kênh của môn học, không có trong repo.

Báo cáo `NLP_BTL_MaNhom.pdf` và slide nộp riêng theo kênh của môn học, không có trong repo.

Chi tiết theo từng PR ở [lịch sử Pull Request](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/pulls?q=is%3Apr).
