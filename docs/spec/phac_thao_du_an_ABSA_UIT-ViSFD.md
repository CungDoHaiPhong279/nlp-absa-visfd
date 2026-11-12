# Bản phác thảo dự án – Bài tập lớn Xử lý ngôn ngữ tự nhiên

> **Tên đề tài (đề xuất):** Phân tích cảm xúc theo khía cạnh trong đánh giá điện thoại của khách hàng trên sàn thương mại điện tử tiếng Việt
> **Bộ dữ liệu:** UIT-ViSFD – https://github.com/LuongPhan/UIT-ViSFD
> **Nhóm:** 2 thành viên (Thành viên A, Thành viên B)
> **Trạng thái:** Bản phác thảo – cần xác nhận với giảng viên trước khi đăng ký

---

## 0. Phạm vi

Làm **đầy đủ nhất có thể**, không giới hạn bởi hạn nộp: cả hai bài toán con, cả hai dạng đầu ra PhoBERT, ViSoBERT, tái lập Bi-LSTM + fastText, 3 seed, demo.

Việc còn lại: đọc bài báo gốc để biết **chính xác cách tính** F1 khía cạnh (84,48) và F1 cảm xúc (63,06). Nếu cách tính khác thì chỉ so sánh trong nội bộ các mô hình của nhóm.

---

## 1. Bài toán

### 1.1. Bối cảnh
Các sàn thương mại điện tử có hàng nghìn bình luận mỗi ngày. Số sao chỉ cho biết khách hài lòng hay không, chứ không cho biết **họ khen hay chê điều gì**. Một bình luận 3 sao có thể khen camera nhưng chê pin. Người bán và nhà sản xuất cần biết khía cạnh nào đang bị chê nhiều để cải thiện sản phẩm và dịch vụ.

### 1.2. Đối tượng sử dụng
| Đối tượng | Nhu cầu |
|---|---|
| Người bán / nhà sản xuất | Biết khía cạnh nào bị chê nhiều nhất của từng mẫu máy |
| Người mua | Đọc nhanh tóm tắt ưu, nhược điểm thay vì đọc hàng trăm bình luận |
| Bộ phận chăm sóc khách hàng | Lọc bình luận chê về dịch vụ, phụ kiện để phản hồi |

### 1.3. Đầu vào / đầu ra
- **Đầu vào:** một bình luận tiếng Việt (văn bản tự do, có thể có teencode, emoji, viết tắt).
- **Đầu ra:** tập các cặp *(khía cạnh, cảm xúc)*.

```
Đầu vào : "Máy đẹp, chụp ảnh nét nhưng pin tụt nhanh quá, xài nửa ngày là hết"
Đầu ra  : {DESIGN: Positive, CAMERA: Positive, BATTERY: Negative}
```

### 1.4. Hai bài toán con
1. **Phát hiện khía cạnh (Aspect Detection):** phân loại đa nhãn. Một bình luận có thể nhắc 0 đến nhiều khía cạnh trong 10 khía cạnh.
2. **Xác định cảm xúc theo khía cạnh (Aspect Sentiment):** với mỗi khía cạnh được nhắc đến, xác định Positive / Neutral / Negative.

### 1.5. Câu hỏi nghiên cứu
- **RQ1:** Mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt (PhoBERT) cải thiện bao nhiêu so với baseline TF-IDF trên cả hai bài toán con?
- **RQ2:** Tách từ tiếng Việt, n-gram và xử lý mất cân bằng nhãn ảnh hưởng thế nào đến kết quả?
- **RQ3:** Mô hình sai nhiều nhất ở loại bình luận nào (khía cạnh ngầm, mỉa mai, khen chê lẫn lộn)?

### 1.6. Tiêu chí thành công
- Pipeline chạy lại được từ dữ liệu thô đến bảng kết quả bằng một lệnh hoặc một notebook.
- Mô hình cải tiến **vượt baseline của nhóm** trên tập test, xét theo macro-F1 của cả hai bài toán con.
- Có bảng đối chiếu với kết quả đã công bố (Bi-LSTM + fastText: 84,48 / 63,06), **nếu cách tính F1 giống nhau**.
- Có phân tích lỗi với ít nhất 8–10 ví dụ cụ thể.

---

## 2. Dữ liệu

### 2.1. Thông tin bộ dữ liệu (theo trang GitHub)
| Mục | Giá trị |
|---|---|
| Số bình luận | 11.122 |
| Nguồn | Một sàn thương mại điện tử lớn tại Việt Nam, bình luận về điện thoại |
| Split | Train 7.786 · Dev 1.112 · Test 2.224 (chia ngẫu nhiên, đã cố định sẵn) |
| Cột | `comment`, `n_star`, `date_time`, `label` |
| Nhãn | 10 khía cạnh × 3 cảm xúc |
| Điều khoản | Miễn phí cho mục đích nghiên cứu; dùng thương mại phải liên hệ tác giả |
| Trích dẫn | Phan, L. L. et al. (2021). *SA2SL: From Aspect-Based Sentiment Analysis to Social Listening System for Business Intelligence.* KSEM 2021, Springer, pp. 647–658 |

**Đã kiểm tra trực tiếp trên file dữ liệu (Train.csv / Dev.csv / Test.csv):**
- Cột: `index, comment, n_star, date_time, label`; không có giá trị thiếu.
- 10 khía cạnh: SCREEN, CAMERA, FEATURES, BATTERY, PERFORMANCE, STORAGE, DESIGN, PRICE, GENERAL, SER&ACC (dịch vụ và phụ kiện).
- Định dạng nhãn: `{CAMERA#Positive};{BATTERY#Negative};{OTHERS};`. Cảm xúc thuộc {Positive, Neutral, Negative}. Một khía cạnh xuất hiện tối đa một lần mỗi bình luận.
- Nhãn đặc biệt `{OTHERS}` (không có cảm xúc): xuất hiện ở 1.694 bình luận train; 115 bình luận train (18 dev, 37 test) **chỉ** có `{OTHERS}`.
- Tần suất khía cạnh trong train: GENERAL 4.866, PERFORMANCE 4.140, BATTERY 3.604, FEATURES 2.642, CAMERA 2.146, PRICE 2.061, SER&ACC 1.995, DESIGN 1.378, SCREEN 949, **STORAGE chỉ 91**.
- Không có bình luận trùng trong train và không trùng giữa train với test.

### 2.2. Kế hoạch khảo sát dữ liệu (EDA)
- Phân bố số khía cạnh trên mỗi bình luận (0, 1, 2, ≥3).
- Tần suất từng khía cạnh, kiểm tra **mất cân bằng** (dự kiến các khía cạnh như STORAGE sẽ hiếm).
- Phân bố cảm xúc theo từng khía cạnh (ví dụ PRICE thường Positive hay Negative?).
- Độ dài bình luận (ký tự, âm tiết, từ sau tách từ), dùng để chọn `max_length` cho PhoBERT.
- Tương quan **số sao và cảm xúc** (bình luận 5 sao có còn chê không?).
- Kiểm tra trùng lặp và rò rỉ giữa các split (bình luận giống hệt nhau xuất hiện ở cả train và test).
- Thống kê teencode, emoji, từ viết tắt phổ biến.

### 2.3. Tiền xử lý (mỗi bước đều phải ghi lý do)
| Bước | Làm hay không | Lý do |
|---|---|---|
| Chuẩn hóa Unicode (NFC) | Có | Tránh cùng một chữ nhưng nhiều mã |
| Chuẩn hóa dấu thanh (hoá → hóa) | Có | Dùng `underthesea.text_normalize` |
| Chữ thường | Có với TF-IDF; không cần với PhoBERT | PhoBERT tự xử lý |
| Bỏ dấu câu | **Thử nghiệm** | "!!!" và "..." có thể mang cảm xúc |
| Bỏ emoji | **Thử nghiệm** | Emoji thường mang cảm xúc mạnh, bỏ đi có thể mất tín hiệu |
| Stopwords | **Không bỏ máy móc** | Các từ như "không", "chưa", "quá" mang nghĩa phủ định hoặc mức độ |
| Tách từ | **Thử nghiệm**; bắt buộc với PhoBERT | PhoBERT được huấn luyện trên văn bản đã tách từ (RDRSegmenter / VnCoreNLP) |

> Nguyên tắc: mọi bước có học tham số (vectorizer, từ điển) **chỉ fit trên tập train**.

---

## 3. Phương pháp

### 3.1. Sơ đồ pipeline
```
Dữ liệu thô (zip)
   └─► Đọc + tách nhãn  ─► EDA (bảng, biểu đồ)
          └─► Tiền xử lý (chuẩn hóa, tách từ)
                 ├─► [Baseline] TF-IDF ─► Logistic Regression / SVM (one-vs-rest)
                 └─► [Cải tiến] PhoBERT tokenizer ─► PhoBERT fine-tune (đa nhãn / đa nhiệm)
                        └─► Dự đoán trên dev (chọn tham số) ─► Dự đoán trên test (1 lần)
                               └─► Đánh giá ─► Bảng so sánh ─► Phân tích lỗi
```

### 3.2. Baseline: TF-IDF + bộ phân loại tuyến tính
- **Biểu diễn:** TF-IDF n-gram mức từ (1–2) và mức ký tự (2–5, thử nghiệm).
- **Bài toán khía cạnh:** 10 bộ Logistic Regression nhị phân (có/không khía cạnh).
- **Bài toán cảm xúc:** với mỗi khía cạnh, một bộ phân loại 3 lớp, chỉ huấn luyện trên các bình luận có khía cạnh đó.
- **Lý do chọn:** nhanh, dễ giải thích (xem được từ nào quyết định nhãn), làm mốc dưới.

### 3.3. Phương án cải tiến: PhoBERT fine-tune
- **Mô hình:** `vinai/phobert-base-v2` (135M tham số, max length 256).
- **Kiến trúc:** encoder PhoBERT, rồi lấy vector [CLS], rồi qua một trong hai dạng đầu ra:
  - **Dạng 1 – gộp:** 10 đầu ra, mỗi đầu ra 4 lớp {None, Positive, Neutral, Negative}. Giải quyết cả hai bài toán con cùng lúc.
  - **Dạng 2 – tách:** một đầu ra sigmoid cho khía cạnh, một đầu ra softmax cho cảm xúc (đa nhiệm).
- **Tham số khởi đầu:** lr 2e-5, batch 16–32, 3–5 epoch, fp16, early stopping theo macro-F1 trên tập dev.
- **Xử lý mất cân bằng:** class weight trong hàm loss (thử nghiệm có/không).

### 3.4. Phương án mở rộng (chỉ làm nếu còn thời gian)
- ViSoBERT (mô hình cho tiếng Việt mạng xã hội) – không cần tách từ.
- So sánh với Bi-LSTM + fastText (tái lập baseline của bài báo gốc).

---

## 4. Thiết kế thực nghiệm

### 4.1. Nguyên tắc
- Dùng **đúng split gốc** (7.786 / 1.112 / 2.224) để so sánh được với kết quả đã công bố.
- Chọn tham số **chỉ dựa trên tập dev**; tập test chỉ chạy **một lần** ở cuối.
- Cố định random seed (ví dụ 42); với PhoBERT chạy **3 seed** và báo cáo trung bình ± độ lệch chuẩn.
- Ghi lại phiên bản thư viện (`requirements.txt`), cấu hình và thời gian chạy.

### 4.2. Thước đo
| Bài toán | Chỉ số chính | Bổ sung |
|---|---|---|
| Phát hiện khía cạnh | Macro-F1 trên 10 khía cạnh | Precision/recall từng khía cạnh, micro-F1 |
| Cảm xúc theo khía cạnh | Macro-F1 trên các cặp (khía cạnh, cảm xúc) | Confusion matrix 3×3 cho từng khía cạnh lớn |
| Tổng thể | Tỉ lệ bình luận dự đoán đúng **toàn bộ** tập nhãn | Dùng để minh họa độ khó |

> Không dùng accuracy làm chỉ số chính, vì nhãn mất cân bằng.

### 4.3. Khảo sát có kiểm soát (mỗi lần chỉ đổi một yếu tố)
| # | Yếu tố thay đổi | Mô hình áp dụng | Câu hỏi |
|---|---|---|---|
| E1 | n-gram (1) / (1–2) / (1–3) | TF-IDF | Bậc n-gram nào tốt nhất? |
| E2 | Âm tiết / từ (sau tách từ) | TF-IDF | Tách từ có giúp tiếng Việt không? |
| E3 | Giữ / bỏ emoji | TF-IDF, PhoBERT | Emoji có mang tín hiệu cảm xúc không? |
| E4 | Có / không class weight | PhoBERT | Có giúp khía cạnh hiếm không? |
| E5 | Có / không đưa số sao vào đầu vào | PhoBERT | Số sao giúp bao nhiêu? (báo cáo riêng, không dùng làm kết quả chính) |
| E6 | Đầu ra dạng gộp / dạng tách | PhoBERT | Kiến trúc nào tốt hơn? |

### 4.4. Bảng kết quả dự kiến (khung – điền khi có số liệu thật)
| Mô hình | F1 khía cạnh (macro) | F1 cảm xúc (macro) | Ghi chú |
|---|---|---|---|
| Bi-LSTM + fastText (bài báo gốc) | 84,48 | 63,06 | Số liệu đã công bố – kiểm tra cách tính |
| TF-IDF + LR (nhóm) | … | … | Baseline |
| TF-IDF + SVM (nhóm) | … | … | |
| PhoBERT-base (nhóm) | … ± … | … ± … | 3 seed |
| PhoBERT-base + class weight | … ± … | … ± … | |

> Tuyệt đối không điền số khi chưa chạy. Mọi số trong báo cáo phải truy được về code và log.

---

## 5. Phân tích lỗi (kế hoạch)

Lấy 8–10 ví dụ sai từ tập **dev** (không dùng test để "dò"), phân nhóm:

| Loại lỗi | Ví dụ minh họa (tự đặt, không phải dữ liệu thật) |
|---|---|
| Khía cạnh ngầm | "xài nửa ngày là sập nguồn" → BATTERY nhưng không có chữ "pin" |
| Khen chê lẫn lộn | "máy đẹp nhưng giá chát" → cảm xúc bị gán nhầm khía cạnh |
| Mỉa mai | "pin trâu ghê, sạc 3 lần một ngày" |
| Phủ định | "không hề lag" → PERFORMANCE Positive |
| Teencode / viết tắt | "cam chụp ok, pin hơi yếu, đt nóng" |
| Số sao mâu thuẫn nội dung | 5 sao nhưng nội dung chê |
| Nhãn có thể sai | Trường hợp nghi người gán nhãn sai – ghi nhận là giới hạn dữ liệu |

Với mỗi ví dụ ghi: bình luận, nhãn đúng, nhãn dự đoán, nguyên nhân giả định, hướng khắc phục.

---

## 6. Giả định và rủi ro

| Giả định / rủi ro | Mức độ | Cách kiểm tra / giảm thiểu |
|---|---|---|
| Định nghĩa F1 của bài báo giống cách nhóm tính | Trung bình | Đọc kỹ bài báo; nếu khác thì chỉ so với baseline của nhóm |
| PhoBERT fine-tune kịp trên Colab T4 | Thấp – trung bình | Chạy thử 1 epoch ở tuần 3 để đo thời gian; dùng fp16, max length theo EDA |
| Tách từ (VnCoreNLP cần Java) chạy được trên Colab | Thấp | Cài thử từ tuần 1; dự phòng: `underthesea.word_tokenize` |
| Khía cạnh hiếm quá ít mẫu để học | Trung bình | Class weight; báo cáo F1 từng khía cạnh để thấy rõ |
| Kết quả cảm xúc không vượt 63,06 | Trung bình | Không sao: báo cáo trung thực, phân tích nguyên nhân |
| Một thành viên không hiểu phần của người kia khi vấn đáp | Trung bình | Họp mỗi tuần, mỗi người giải thích lại phần của người kia |

---

## 7. Kế hoạch 10 tuần

| Tuần | Thành viên A | Thành viên B | Mốc |
|---|---|---|---|
| 1 | Tải dữ liệu, hàm tách nhãn + kiểm thử | Dựng repo, môi trường Colab, `requirements.txt`, thử VnCoreNLP | Đăng ký đề tài |
| 2 | EDA đầy đủ (bảng, biểu đồ) | Đọc bài báo gốc, viết script đánh giá (macro-F1 hai bài toán) | Có báo cáo EDA |
| 3 | Tiền xử lý + lưu dữ liệu trung gian | Chạy thử PhoBERT 1 epoch, đo thời gian | Pipeline chạy từ đầu đến cuối |
| 4 | Baseline TF-IDF + LR/SVM | PhoBERT dạng gộp, chọn tham số trên dev | Kết quả baseline |
| 5 | Khảo sát E1, E2, E3 (TF-IDF) | PhoBERT dạng tách, khảo sát E4, E6 | |
| 6 | Khảo sát E3 trên PhoBERT, khảo sát E5 (số sao) | Chạy 3 seed cho cấu hình tốt nhất | **Chốt cấu hình** |
| 7 | Chạy tập test (1 lần), bảng so sánh | Phân tích lỗi trên dev, chọn 8–10 ví dụ | Bảng kết quả cuối |
| 8 | Viết báo cáo: Dữ liệu, Thực nghiệm | Viết báo cáo: Phương pháp, Kết quả | Bản nháp báo cáo |
| 9 | Demo (giao diện) | README, kiểm tra chạy lại từ đầu trên máy khác | Bản báo cáo hoàn chỉnh |
| 10 | Slide + tập thuyết trình | Slide + tập vấn đáp chéo | **Nộp bài + bảo vệ** |

**Cả hai cùng làm:** phần Giới thiệu, Kết luận, phân tích lỗi, kiểm tra chéo số liệu.

---

## 8. Cấu trúc repository

```
nlp-absa-visfd/
├── README.md                 # mục tiêu, dữ liệu, cách chạy, tái tạo kết quả
├── requirements.txt          # phiên bản thư viện
├── data/
│   ├── raw/                  # UIT-ViSFD gốc (không đẩy lên git nếu giấy phép không cho)
│   └── processed/            # dữ liệu đã chuẩn hóa / tách từ
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_tfidf.ipynb
│   ├── 03_phobert.ipynb
│   └── 04_error_analysis.ipynb
├── src/
│   ├── data.py               # đọc + tách nhãn
│   ├── preprocess.py         # chuẩn hóa, tách từ
│   ├── evaluate.py           # macro-F1 hai bài toán con
│   ├── train_baseline.py
│   └── train_phobert.py
├── configs/                  # tham số từng thí nghiệm (yaml)
├── results/
│   ├── tables/               # bảng so sánh (csv)
│   ├── figures/              # biểu đồ EDA, confusion matrix
│   └── predictions/          # dự đoán minh họa
├── demo/
│   └── app.py                # Gradio / Streamlit
└── report/
    └── NLP_BTL_MaNhom.pdf
```

---

## 9. Dàn ý báo cáo (gắn với rubric)

| Mục báo cáo | Nội dung chính | Tiêu chí rubric |
|---|---|---|
| 1. Giới thiệu | Bối cảnh, đối tượng, RQ1–RQ3 | Xác định bài toán (15) |
| 2. Dữ liệu | Nguồn, thống kê, EDA, tiền xử lý có lý do | Xác định bài toán và dữ liệu (15) |
| 3. Phương pháp | TF-IDF, PhoBERT, sơ đồ pipeline, tham số | Thiết kế và triển khai (25) |
| 4. Thực nghiệm | Split, seed, thước đo, các khảo sát E1–E6 | Thực nghiệm và chỉ số (25) |
| 5. Kết quả | Bảng so sánh, biểu đồ, phân tích lỗi | Phân tích và thảo luận (15) |
| 6. Kết luận | Đạt được, chưa đạt, hướng phát triển | Phân tích (15) |
| 7. Phân công + tài liệu | Đóng góp từng người, trích dẫn, phụ lục dùng AI | Báo cáo và tái lập (20) |

---

## 10. Ý tưởng demo (10 phút)

1. **(1 phút)** Bài toán: một bình luận 3 sao khen camera, chê pin – số sao không nói lên điều đó.
2. **(2 phút)** Dữ liệu và EDA: khía cạnh nào hay được nhắc, mất cân bằng ra sao.
3. **(3 phút)** Demo trực tiếp: dán bình luận, giao diện **tô màu** từng khía cạnh (xanh = khen, đỏ = chê). Thử cả câu khó: mỉa mai, khía cạnh ngầm.
4. **(1 phút)** Bảng tổng hợp theo khía cạnh: "Khía cạnh bị chê nhiều nhất trong tập test là…" – thể hiện tính ứng dụng.
5. **(2 phút)** Bảng so sánh baseline với PhoBERT, và một khảo sát có kiểm soát nổi bật.
6. **(1 phút)** Giới hạn và hướng phát triển.

---

## 11. Bản nháp phiếu đăng ký đề tài

| Mục | Nội dung |
|---|---|
| **Tên đề tài** | Phân tích cảm xúc theo khía cạnh trong đánh giá điện thoại của khách hàng trên sàn thương mại điện tử tiếng Việt |
| **Thành viên** | [Họ tên A] – [MSSV]; [Họ tên B] – [MSSV] |
| **Mục tiêu** | Xây dựng hệ thống xác định khách hàng nhắc đến khía cạnh nào của sản phẩm (pin, camera, giá…) và cảm xúc với từng khía cạnh; so sánh phương pháp truyền thống với mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt |
| **Dữ liệu** | UIT-ViSFD: 11.122 bình luận, 10 khía cạnh × 3 cảm xúc, split chuẩn 7.786 / 1.112 / 2.224; dùng cho mục đích nghiên cứu, trích dẫn bài báo gốc (Phan et al., KSEM 2021) |
| **Phương án 1 (baseline)** | TF-IDF n-gram + Logistic Regression / SVM (one-vs-rest theo khía cạnh) |
| **Phương án 2 (cải tiến)** | Fine-tune PhoBERT-base cho bài toán đa nhãn theo khía cạnh, có xử lý mất cân bằng nhãn |
| **Khảo sát có kiểm soát** | Bậc n-gram, tách từ, giữ/bỏ emoji, class weight, ảnh hưởng của số sao |
| **Thước đo** | Macro-F1 cho phát hiện khía cạnh và cảm xúc theo khía cạnh; confusion matrix |
| **Kết quả dự kiến** | PhoBERT vượt baseline TF-IDF trên cả hai bài toán con; bảng đối chiếu với kết quả công bố; phân tích lỗi với khía cạnh ngầm, mỉa mai, khen chê lẫn lộn; demo web tô màu khía cạnh |

---

## 12. Câu hỏi còn mở

- Có cần thu thập thêm một tập bình luận mới (ví dụ về một mẫu điện thoại ra mắt gần đây) để thử mô hình trên dữ liệu ngoài phân phối không? (Tùy chọn – chú ý điều khoản trang web và ẩn thông tin cá nhân.)
