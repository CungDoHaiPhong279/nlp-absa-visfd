# Quyết định thiết kế

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
- **D13 – Commit `results/runs/*.json`:** mỗi lần chạy một file tên riêng (`{name}__seed{seed}.json`) nên không xung đột; cả nhóm cần chung các file này để `make_tables.py` sinh cùng bảng và để mọi số trong báo cáo truy được. `results/predictions/` vẫn không commit (lớn), chia sẻ qua Google Drive.
- **D14 – Quy trình nhóm:** mọi thay đổi vào `main` qua Pull Request, CI xanh, người còn lại duyệt; quy tắc chi tiết ở `CONTRIBUTING.md`; quy tắc cho trợ lý AI ở `AGENTS.md`.

## Nhật ký quyết định khi chạy
- **Task 3 – giới hạn emoji/emoticon:** `strip_emoji` chỉ bỏ các dải Unicode emoji (U+1F000–1FAFF, U+2600–27BF, U+2B00–2BFF, U+FE0F, U+200D, U+20E3). Emoticon dạng chữ như `:)` không được xử lý (không bị bỏ bởi `strip_emoji`; `strip_punct` chỉ thay dấu câu của chúng bằng khoảng trắng).
- **Task 4 – trùng lặp giữa các split:** `eda_duplicates.json` cho `train_dup=2`, `train_dev_overlap=3`, `train_test_overlap=5`, `dev_test_overlap=0` (so sánh trên `normalize(..., lowercase=True)`, đếm số văn bản khác nhau). Không xóa mẫu để giữ split gốc.
- **Task 4 – `max_length` (D9):** đo trên Train, gồm token đặc biệt (`results/tables/eda_subword_len.csv`). PhoBERT (VnCoreNLP tách từ): mean 40,8, p95 86, p99 119, max 199, nên `max_length = 128`. ViSoBERT (văn bản thô sau `normalize`): mean 68,3, p95 145, p99 197, max 350, nên `max_length = 224` (p99 làm tròn lên bội số của 32, dưới trần 256; khoảng 1% bình luận bị cắt ở cả hai mô hình).
- **Task 4 – protobuf cho tokenizer ViSoBERT:** tokenizer sentencepiece của ViSoBERT cần `protobuf` để nạp dưới transformers 5.x (thiếu thì rơi về TikToken và báo thiếu `tiktoken`). Đã thêm `protobuf>=4.21` vào `requirements.txt`; không cần `tiktoken`. Lệnh nạp đã chạy được (không cần `use_fast=False`): `AutoTokenizer.from_pretrained("uitnlp/visobert")`. Task 9 phải nạp đúng như vậy.
- **Task 6 – TF-IDF giữ emoji và từ 1 ký tự:** `token_pattern` mặc định của scikit-learn (`\b\w\w+\b`) bỏ mọi emoji và từ 1 ký tự (ví dụ "ý"), nên E3 (giữ/bỏ emoji) với TF-IDF cho kết quả giống hệt nhau. `TfidfAbsa` với `analyzer="word"` dùng `WORD_TOKEN_PATTERN` (từ mọi độ dài, mỗi emoji là một token, cùng dải Unicode với `strip_emoji`). Mọi run TF-IDF đã chạy lại sau thay đổi này.
- **Task 6 – E0, lưới C mở rộng so với D10:** với lưới {0,25…4}, C tốt nhất của LR nằm ở mép trên nên dò tiếp tới khi `dev.sentiment_macro_f1` giảm. Sentiment macro-F1 trên Dev (%):

  | C | 0,25 | 0,5 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | LR | 29,44 | 34,67 | 38,89 | 42,72 | 45,71 | 47,36 | 47,66 | 48,64 | 48,97 | 49,44 | 49,76 | **50,49** | 50,33 |
  | SVM | 44,60 | 47,88 | 49,34 | 49,92 | **50,20** | 50,13 | 49,95 | 49,88 | | | | | |

  Chốt LR `C=512` (gần như không regularize; aspect macro-F1 gần phẳng từ C=32, 78,9–79,0%) và SVM `C=4`. Thời gian train `tfidf_lr` khoảng 12 giây trên CPU (laptop của A).
- **Task 9 – đo thời gian trên Colab T4 (Step 8):** `run.py --config configs/phobert_joint.yaml --limit 500 --set model.epochs=1` mất 17 giây train (47 giây cả lệnh, gồm tải mô hình và tách từ VnCoreNLP). Ước lượng tuyến tính cho Train đầy đủ (7.786 dòng): khoảng 4–5 phút mỗi epoch, tối đa 5 epoch, tức khoảng 25 phút mỗi lần chạy PhoBERT; ViSoBERT (`max_length` 224) dài hơn. Dưới ngưỡng 60 phút nên giữ `max_length` 128/224 và `batch_size` 16. VRAM chưa đo được (lệnh `nvidia-smi` chạy sau khi tiến trình đã kết thúc).
- **Notebook Colab:** chỉ lưu bằng Ctrl+S vào Drive, không dùng "Lưu bản sao vào GitHub" (lệnh này đẩy thẳng lên `main`, bỏ qua PR và nbstripout). Ba commit "Được tạo bằng Colab" ngày 2026-11-20 đã được đưa notebook về bản sạch.
- **Task 9 – PhoBERT tối đa 10 epoch (thay 5):** lần chạy đầy đủ đầu tiên của `phobert_joint` (seed 42, T4) cho Dev sentiment macro-F1 theo epoch 24,4 → 36,9 → 44,7 → 46,7 → **48,2**, vẫn tăng ở epoch cuối, nên early stopping chưa hề kích hoạt (aspect 80,17 / sentiment 48,16, 346 giây). Tăng `epochs` lên 10 trong `phobert_joint.yaml` (các file E3–E6 và `visobert_joint` kế thừa), giữ `patience: 2` để dừng khi Dev không tăng nữa. Mỗi epoch khoảng 70 giây trên T4. Lần chạy 5 epoch không được giữ lại trong `results/runs/`.
