# Giải thích dự án cho người mới

Tài liệu này giải thích dự án bằng lời thường: dự án làm gì, các thuật ngữ nghĩa là gì, từng task làm gì và đã làm thế nào. Đọc từ trên xuống. Gặp từ lạ thì tra ở **mục 3 (Từ điển)**.

Số liệu trong file lấy từ `results/` tại thời điểm viết (sau PR #38). Khi chạy lại, số mới nằm ở `results/tables/`.

---

## 1. Dự án làm gì?

Đầu vào là một bình luận về điện thoại bằng tiếng Việt. Đầu ra cho biết bình luận **nhắc tới khía cạnh nào** của điện thoại và **khen hay chê** khía cạnh đó.

Ví dụ: *"Pin dùng lâu nhưng camera chụp tối"*

| Khía cạnh | Cảm xúc |
|---|---|
| Pin (BATTERY) | Tích cực |
| Camera (CAMERA) | Tiêu cực |
| 8 khía cạnh còn lại | Không nhắc tới |

Bài toán này gọi là **ABSA** (Aspect-Based Sentiment Analysis, *phân tích cảm xúc theo khía cạnh*). Nó khó hơn phân tích cảm xúc thường. Phân tích cảm xúc thường chỉ trả lời "cả câu khen hay chê"; câu ví dụ trên vừa khen vừa chê nên câu trả lời đó vô nghĩa.

ABSA gồm **hai bài toán con**:
1. **Phát hiện khía cạnh:** bình luận nhắc tới những khía cạnh nào trong 10 khía cạnh?
2. **Phân loại cảm xúc:** với mỗi khía cạnh được nhắc, cảm xúc là Tích cực, Trung tính hay Tiêu cực?

Nhóm xây nhiều mô hình giải bài toán này, từ đơn giản (TF-IDF) tới mạnh (PhoBERT). Mọi mô hình được chấm bằng **cùng một hàm đánh giá** trên **cùng dữ liệu**, để so sánh công bằng và trả lời câu hỏi "mô hình nào tốt hơn, tốt hơn ở đâu, vì sao".

---

## 2. Dữ liệu: UIT-ViSFD

- **Nguồn:** bộ dữ liệu công khai của nhóm UIT (ĐH CNTT, ĐHQG TP.HCM), bài báo Phan et al. (2021). Gồm 11.122 bình luận điện thoại lấy từ một trang thương mại điện tử.
- **Chia sẵn 3 phần** (gọi là *split*), nhóm **không được trộn hay chia lại**:

  | Phần | Số bình luận | Dùng để |
  |---|---|---|
  | Train | 7.786 | Cho mô hình học |
  | Dev | 1.112 | Thử và chọn cấu hình tốt nhất |
  | Test | 2.224 | Chấm điểm cuối cùng, **chỉ một lần** |

- **Mỗi dòng gồm:** `comment` (bình luận), `n_star` (số sao 1–5), `label` (nhãn).
- **Nhãn có dạng:** `{BATTERY#Positive};{CAMERA#Negative};`. Mỗi cặp `{khía cạnh#cảm xúc}` là một ý.
- **10 khía cạnh**, thứ tự cố định:

  | Mã | Nghĩa | Số lần xuất hiện ở Train |
  |---|---|---|
  | SCREEN | Màn hình | 949 |
  | CAMERA | Camera | 2.146 |
  | FEATURES | Tính năng (vân tay, wifi, cảm ứng…) | 2.642 |
  | BATTERY | Pin | 3.604 |
  | PERFORMANCE | Hiệu năng (mượt, lag…) | 4.140 |
  | STORAGE | Bộ nhớ | **91** (rất hiếm) |
  | DESIGN | Thiết kế | 1.378 |
  | PRICE | Giá | 2.061 |
  | GENERAL | Đánh giá chung | 4.866 |
  | SER&ACC | Dịch vụ và phụ kiện (giao hàng, bảo hành, sạc…) | 1.995 |

- **`{OTHERS}`** không phải khía cạnh. Nhãn này dán cho bình luận không nói gì về 10 khía cạnh trên. Trong code, bình luận như vậy có kết quả "không có khía cạnh nào".
- **Giấy phép:** chỉ dùng cho nghiên cứu, nên **không commit dữ liệu vào repo** (repo đang public). Ai cũng tải bằng `python scripts/download_data.py`.

### Cách code lưu nhãn: "ma trận nhãn"
Mọi mô hình đều trả về một bảng số: mỗi bình luận một hàng, 10 cột ứng với 10 khía cạnh. Mỗi ô là một mã:

| Mã | Nghĩa |
|---|---|
| 0 | Không nhắc tới (None) |
| 1 | Tích cực (Positive) |
| 2 | Trung tính (Neutral) |
| 3 | Tiêu cực (Negative) |

Câu ví dụ ở mục 1 thành hàng `[0, 3, 0, 1, 0, 0, 0, 0, 0, 0]` (CAMERA là cột 2 = 3, BATTERY là cột 4 = 1). Nhờ mọi mô hình trả cùng dạng này, chỉ cần **một** hàm `evaluate` để chấm tất cả. Đây là quy ước quan trọng nhất của dự án, không ai được đổi.

---

## 3. Từ điển thuật ngữ

### 3.1. Học máy cơ bản

| Thuật ngữ | Nghĩa |
|---|---|
| **Mô hình (model)** | Chương trình học từ ví dụ có đáp án, rồi đoán đáp án cho ví dụ mới. |
| **Huấn luyện (train, fit)** | Cho mô hình xem dữ liệu Train để học. Trong code là hàm `fit`. |
| **Dự đoán (predict)** | Mô hình đã học đưa ra đáp án cho văn bản mới. Trong code là hàm `predict`. |
| **Nhãn (label), gold** | Đáp án đúng do người gán. "Gold" = đáp án chuẩn, "pred" = đáp án mô hình đoán. |
| **Đặc trưng (feature)** | Cách biến văn bản thành con số để máy tính xử lý. Máy không đọc chữ được, chỉ tính toán được với số. |
| **Bộ phân loại (classifier)** | Mô hình chọn một trong vài lớp, ví dụ chọn "có/không nhắc tới pin" hoặc "Tích cực/Trung tính/Tiêu cực". |
| **Lớp (class)** | Một lựa chọn đầu ra. Bài phát hiện khía cạnh có 2 lớp (có/không); bài cảm xúc có 3 lớp. |
| **Train / Dev / Test** | Ba phần dữ liệu (mục 2). Quy tắc vàng: học trên Train, chọn cấu hình trên Dev, chấm điểm cuối trên Test **một lần**. Nếu nhìn Test để chọn cấu hình thì điểm Test bị "ăn gian", không còn phản ánh mô hình gặp dữ liệu mới. |
| **Siêu tham số (hyperparameter)** | Các núm vặn do người chọn trước khi huấn luyện (ví dụ `C`, `ngram_range`, `epochs`). Mô hình không tự học các giá trị này; ta thử nhiều giá trị trên Dev rồi chọn. |
| **Dò tham số (sweep, E0)** | Chạy cùng một mô hình với nhiều giá trị của một siêu tham số rồi chọn giá trị tốt nhất trên Dev. |
| **Overfitting (học vẹt)** | Mô hình nhớ thuộc lòng Train nhưng đoán kém dữ liệu mới. |
| **Regularization** | "Phạt" mô hình khi nó quá phức tạp, để nó bớt học vẹt. |
| **Seed** | Con số khởi đầu cho phần ngẫu nhiên của máy tính. Cùng seed thì chạy lại ra cùng kết quả. Dự án dùng seed 42; khi chốt kết quả cuối thì chạy 3 seed (42, 43, 44) để xem kết quả có ổn định không. |
| **Baseline** | Mô hình đơn giản làm mốc so sánh. Mô hình mạnh phải vượt mốc này mới có ý nghĩa. |
| **Dummy (mô hình giả)** | Mô hình luôn đoán lớp phổ biến nhất. Code dùng nó khi dữ liệu train của một bộ phân loại chỉ có một lớp, vì khi đó không có gì để học. |

### 3.2. Xử lý văn bản

| Thuật ngữ | Nghĩa |
|---|---|
| **Tiền xử lý (preprocess)** | Làm sạch văn bản trước khi đưa vào mô hình. |
| **Chuẩn hóa Unicode (NFC)** | Cùng một chữ "ố" có thể được gõ theo 2 cách mã hóa khác nhau trong máy. NFC đưa về một cách duy nhất, để máy không coi là hai chữ khác nhau. |
| **lowercase** | Đưa về chữ thường ("Pin" → "pin"). |
| **strip_emoji / strip_punct** | Bỏ emoji / bỏ dấu câu. Là tùy chọn, bật hay tắt tùy thí nghiệm. |
| **Token** | Một đơn vị văn bản sau khi cắt, thường là một từ hoặc âm tiết. |
| **Âm tiết (syllable)** | Tiếng Việt viết cách nhau bằng dấu cách theo âm tiết: "điện thoại" là 2 âm tiết. |
| **Tách từ (word segmentation)** | Gộp các âm tiết thành từ: "điện thoại" → "điện_thoại" (1 từ). Tiếng Anh không cần bước này; tiếng Việt cần, vì một từ có thể gồm nhiều âm tiết. |
| **underthesea, VnCoreNLP** | Hai công cụ tách từ tiếng Việt. PhoBERT bắt buộc dùng VnCoreNLP vì nó được huấn luyện trên văn bản tách bằng công cụ này. VnCoreNLP viết bằng Java nên máy cần cài Java 17. |
| **n-gram** | Cụm n token liền nhau. Unigram (1-gram) = từng từ đơn ("pin", "trâu"). Bigram (2-gram) = cụm 2 từ ("pin trâu"). Bigram giữ được ngữ cảnh, ví dụ "không tốt" khác hẳn "tốt". `ngram_range: [1, 2]` nghĩa là dùng cả unigram lẫn bigram. |
| **char n-gram (`char_wb`)** | n-gram theo **ký tự** thay vì theo từ, ví dụ "pin" → "pi", "in", "pin". Giúp chịu được lỗi chính tả, viết tắt. |
| **Emoji / emoticon** | Emoji là hình (😍 👍); emoticon là ký tự gõ tay (`:)`). Code chỉ xử lý emoji. |
| **Teencode** | Cách viết tắt kiểu mạng: "k" = không, "dc" = được, "ok". EDA có đếm tần suất. |
| **Subword (token con)** | PhoBERT/ViSoBERT cắt từ hiếm thành các mảnh nhỏ hơn, nên số token con thường nhiều hơn số từ. |
| **max_length** | Số token con tối đa mô hình đọc; phần dài hơn bị cắt. Chọn theo EDA: PhoBERT 128, ViSoBERT 224 (chỉ khoảng 1% bình luận bị cắt). |
| **Cache** | Lưu sẵn kết quả tách từ ra file (`data/processed/`) để lần sau không phải tách lại, vì VnCoreNLP chạy chậm. |

### 3.3. Mô hình TF-IDF (baseline)

| Thuật ngữ | Nghĩa |
|---|---|
| **TF-IDF** | Cách biến văn bản thành dãy số. Mỗi từ (hoặc cụm từ) được một điểm. **TF** (term frequency): từ xuất hiện nhiều trong bình luận này thì điểm cao. **IDF** (inverse document frequency): từ xuất hiện ở quá nhiều bình luận (như "máy", "là") thì điểm bị hạ, vì nó không phân biệt được gì. Kết quả: từ đặc trưng như "pin", "lag", "trâu" có điểm cao. |
| **Vectorizer** | Bộ chuyển văn bản → dãy số TF-IDF. Nó **chỉ được học từ vựng trên Train**; Dev/Test chỉ được chuyển đổi theo từ vựng đó, để không "nhìn trộm" dữ liệu kiểm tra. |
| **min_df** | Bỏ các từ xuất hiện ở ít hơn `min_df` bình luận (dự án dùng 2), để bớt nhiễu từ gõ sai. |
| **LR (Logistic Regression, hồi quy logistic)** | Bộ phân loại tuyến tính đơn giản: mỗi từ có một "trọng số" cộng hoặc trừ điểm; tổng điểm cao thì chọn lớp đó. Ví dụ "trâu" cộng điểm cho "pin tích cực", "chai" trừ điểm. Tên có chữ "regression" nhưng thực chất dùng để phân loại. |
| **SVM (Support Vector Machine)** | Một bộ phân loại tuyến tính khác. Thay vì tính xác suất như LR, nó tìm "đường ranh giới" cách xa hai nhóm dữ liệu nhất. Với văn bản, SVM và LR thường ngang nhau; SVM thường train nhanh hơn. Code dùng `LinearSVC` của scikit-learn. |
| **C** | Siêu tham số của LR và SVM, điều khiển mức **regularization**. **C nhỏ** = phạt mạnh, mô hình đơn giản, dễ "học thiếu". **C lớn** = phạt nhẹ, mô hình bám sát Train, dễ học vẹt. Nhóm thử nhiều giá trị C trên Dev (bước E0). Kết quả: SVM tốt nhất ở C=4, LR tốt nhất ở C=512. |
| **mode `split` (tách)** | Làm 2 bước, đúng như bài toán: bước 1 có 10 bộ phân loại "có/không nhắc khía cạnh"; bước 2, với khía cạnh có mặt, một bộ phân loại 3 lớp đoán cảm xúc. Đây là chế độ mặc định của dự án. |
| **mode `joint` (gộp)** | Mỗi khía cạnh một bộ phân loại 4 lớp (None/Pos/Neu/Neg), làm cả hai việc một lúc. So sánh hai chế độ là khảo sát E6. |
| **token_pattern** | Quy tắc cắt văn bản thành token trong TF-IDF. Quy tắc mặc định của scikit-learn bỏ emoji và từ 1 ký tự; nhóm đã sửa để giữ lại cả hai (PR #37). |

### 3.4. Mô hình học sâu

| Thuật ngữ | Nghĩa |
|---|---|
| **Mạng nơ-ron / học sâu (deep learning)** | Mô hình nhiều tầng tính toán, tự học cách biểu diễn văn bản thay vì dùng đặc trưng do người thiết kế như TF-IDF. Mạnh hơn nhưng cần GPU và nhiều thời gian. |
| **Transformer** | Kiến trúc mạng nơ-ron hiện đại cho ngôn ngữ (nền tảng của ChatGPT, BERT). Điểm mạnh: khi đọc một từ, nó "nhìn" cả câu để hiểu ngữ cảnh. |
| **BERT / RoBERTa** | Họ mô hình transformer được **huấn luyện trước (pre-train)** trên lượng văn bản khổng lồ, nên đã "hiểu" ngôn ngữ từ trước. |
| **PhoBERT** (`vinai/phobert-base-v2`) | BERT cho tiếng Việt của VinAI, học trên báo chí và Wikipedia. Cần văn bản đã tách từ bằng VnCoreNLP. Là mô hình chính của dự án. |
| **ViSoBERT** (`uitnlp/visobert`) | BERT tiếng Việt của UIT, học trên mạng xã hội, nên quen teencode và emoji. Không cần tách từ. |
| **Fine-tune (tinh chỉnh)** | Lấy mô hình đã pre-train rồi huấn luyện thêm một chút trên dữ liệu của mình. Nhóm **không** huấn luyện PhoBERT từ đầu, vì việc đó cần hàng tuần trên nhiều GPU. |
| **Head (đầu ra)** | Lớp nhỏ gắn vào cuối PhoBERT để ra đúng dạng đáp án của mình (10 khía cạnh × 4 mã). Code: `AbsaHead`. |
| **Loss (hàm mất mát)** | Con số đo mô hình đoán sai bao nhiêu. Huấn luyện là quá trình giảm loss. |
| **Epoch** | Một lượt mô hình xem hết Train. Dự án dùng tối đa 5 epoch. |
| **Batch** | Một nhóm bình luận đưa vào mô hình cùng lúc (ví dụ 16), để tính nhanh hơn. |
| **Learning rate (lr)** | Mỗi bước học, mô hình điều chỉnh mạnh hay nhẹ. Quá lớn thì học loạn; quá nhỏ thì học chậm. |
| **Early stopping, patience** | Sau mỗi epoch chấm trên Dev; nếu `patience` (= 2) epoch liền không tiến bộ thì dừng sớm và giữ bản tốt nhất. Tránh học vẹt. |
| **Threshold (ngưỡng)** | Ở chế độ split, mô hình cho xác suất "có nhắc khía cạnh" từ 0 đến 1; vượt ngưỡng (mặc định 0,5) thì coi là có. Ngưỡng được chọn trên Dev trong khoảng 0,30–0,70. |
| **Class weight (trọng số lớp)** | Lớp hiếm (như Trung tính, hay STORAGE chỉ 91 mẫu) bị mô hình bỏ qua; trọng số lớp tăng "mức phạt" khi đoán sai lớp hiếm. Khảo sát E4 kiểm tra cách này có giúp không. |
| **GPU, Colab, T4** | GPU là chip tính toán song song, cần cho học sâu. Máy của nhóm không có GPU mạnh nên dùng **Google Colab** (máy ảo miễn phí của Google, có GPU T4). |
| **fp16** | Tính toán với số độ chính xác thấp hơn để chạy nhanh và tốn ít bộ nhớ GPU hơn. |
| **Checkpoint** | File lưu trọng số mô hình đã huấn luyện, để demo nạp lại mà không phải train lại. Lưu trong `models/` và Google Drive (file lớn, không đưa lên Git). |
| **Tiny transformer** | Một transformer "đồ chơi" rất nhỏ, chạy trên CPU trong vài giây. Chỉ dùng để test code, không dùng để lấy kết quả. |
| **Bi-LSTM** | Loại mạng nơ-ron cũ hơn transformer, đọc câu theo thứ tự từ trái sang phải và từ phải sang trái. Bài báo gốc dùng mô hình này. |
| **fastText, embedding** | Embedding là cách biểu diễn mỗi từ bằng một dãy số, sao cho từ nghĩa gần nhau thì dãy số gần nhau. fastText (của Facebook) là một bộ embedding có sẵn cho tiếng Việt (`cc.vi.300`). Bi-LSTM dùng nó làm đầu vào. |

### 3.5. Đánh giá (chấm điểm)

Ví dụ để hiểu: mô hình đoán 10 bình luận có nhắc "pin"; trong đó 8 đúng. Thực tế có 16 bình luận nhắc "pin".

| Thuật ngữ | Nghĩa | Ví dụ |
|---|---|---|
| **Precision (độ chính xác)** | Trong những lần mô hình nói "có", bao nhiêu phần đúng? | 8/10 = 80% |
| **Recall (độ phủ)** | Trong những trường hợp thật sự "có", mô hình tìm ra bao nhiêu? | 8/16 = 50% |
| **F1** | Trung bình "điều hòa" của precision và recall; chỉ cao khi **cả hai** đều cao. | 2·0,8·0,5/(0,8+0,5) ≈ 61,5% |
| **Support** | Số mẫu đúng thật sự của một lớp. | 16 |
| **Macro-F1** | Tính F1 riêng cho từng khía cạnh (hoặc từng lớp) rồi lấy trung bình cộng. Khía cạnh hiếm (STORAGE) có trọng lượng ngang khía cạnh phổ biến, nên đây là thước đo khắt khe. **Đây là chỉ số chính của dự án.** | |
| **Micro-F1** | Gộp mọi khía cạnh lại thành một rồi tính F1. Bị khía cạnh phổ biến chi phối. | |
| **Exact match** | Tỉ lệ bình luận mà mô hình đoán đúng **toàn bộ** 10 ô. Rất khắt khe. | |
| **Confusion matrix (ma trận nhầm lẫn)** | Bảng đếm "đáp án thật là X, mô hình đoán Y" bao nhiêu lần, cho thấy mô hình hay nhầm gì với gì. Dự án dùng bảng 4×4 (None, Pos, Neu, Neg). | |

Các chỉ số trong file kết quả (`results/runs/*.json`):

| Tên trong code | Đo gì |
|---|---|
| `aspect_macro_f1` | Bài 1: phát hiện đúng khía cạnh (bỏ qua cảm xúc). |
| `sentiment_macro_f1` | Bài 2: đúng **cả** khía cạnh **lẫn** cảm xúc. Mỗi cặp (khía cạnh, cảm xúc) là một lớp, tức 30 lớp. Đây là chỉ số dùng để chọn cấu hình. |
| `polarity_macro_f1_detected` | Chỉ xét những khía cạnh mà mô hình đã phát hiện đúng, mô hình đoán cảm xúc đúng tới đâu. Dùng để thử so với cách tính của bài báo. |
| `exact_match` | Như trên. |
| `per_aspect` | Precision/recall/F1/support của từng khía cạnh. |

**Lưu ý so với bài báo gốc:** bài báo báo cáo 84,48 (khía cạnh) / 63,06 (cảm xúc), nhưng chưa rõ họ tính F1 kiểu gì. Khi chưa xác minh (issue #18), bảng của nhóm chỉ ghi số của họ ở cột ghi chú "không cùng định nghĩa", **không** đặt cạnh số của nhóm để so trực tiếp.

### 3.6. Thí nghiệm

| Thuật ngữ | Nghĩa |
|---|---|
| **Khảo sát có kiểm soát (ablation)** | Đổi **đúng một** yếu tố, giữ nguyên mọi thứ khác, xem kết quả thay đổi bao nhiêu. Nhờ vậy mới biết chính yếu tố đó gây ra thay đổi. |
| **E0** | Dò tham số C (TF-IDF) và ngưỡng (transformer). |
| **E1** | Loại đặc trưng: unigram / unigram+bigram / 1–3-gram / char n-gram. |
| **E2** | Âm tiết hay từ (có tách từ underthesea hay không). |
| **E3** | Giữ hay bỏ emoji. |
| **E4** | Có hay không trọng số lớp (PhoBERT). |
| **E5** | Có đưa số sao vào đầu vào không (PhoBERT). Chỉ báo cáo riêng, vì trong thực tế không phải lúc nào cũng có số sao. |
| **E6** | Chế độ joint hay split. |
| **Cấu hình (config) YAML** | File văn bản mô tả một thí nghiệm: tên, seed, cách tiền xử lý, loại mô hình và tham số. Ví dụ `configs/tfidf_lr_base.yaml`. File ablation "kế thừa" file gốc qua `base:` và chỉ ghi điều khác. |
| **Run** | Một lần chạy một cấu hình với một seed. Kết quả ghi ra `results/runs/{tên}__seed{seed}.json`. |

### 3.7. Công cụ làm việc nhóm

| Thuật ngữ | Nghĩa |
|---|---|
| **Python, `.venv`** | Ngôn ngữ của dự án (bản 3.11). `.venv` là "môi trường ảo": một bản Python riêng cho dự án, thư viện cài vào đây không ảnh hưởng máy. |
| **scikit-learn, pandas, numpy** | Thư viện học máy (TF-IDF, LR, SVM, F1), xử lý bảng dữ liệu, và tính toán số. |
| **PyTorch, transformers (Hugging Face)** | Thư viện học sâu, và thư viện tải và dùng PhoBERT/ViSoBERT. |
| **Gradio** | Thư viện làm giao diện web nhanh bằng Python, dùng cho demo. |
| **matplotlib** | Thư viện vẽ biểu đồ. |
| **Jupyter Notebook** | File `.ipynb` vừa chứa code vừa chứa chữ và hình; dùng cho EDA và phân tích lỗi. |
| **Git** | Phần mềm lưu lịch sử thay đổi code. |
| **Commit** | Một lần "chụp" lại thay đổi, kèm lời mô tả. |
| **Branch (nhánh)** | Bản làm việc riêng tách từ `main`, để làm một task mà không ảnh hưởng người khác. Tên dạng `task/6-tfidf-baseline`. |
| **GitHub** | Trang web lưu repo Git trực tuyến: `CungDoHaiPhong279/nlp-absa-visfd`. |
| **Pull Request (PR)** | Đề nghị gộp một nhánh vào `main`, để người kia xem và duyệt trước. |
| **Merge, squash** | Gộp nhánh vào `main`. "Squash" nén mọi commit của nhánh thành một commit gọn. |
| **Review, Approve** | Người kia đọc code trong PR, góp ý, rồi bấm đồng ý. |
| **Branch protection** | Luật khóa `main`: không ai đẩy thẳng vào được; phải qua PR, CI xanh và có người duyệt. |
| **CI (GitHub Actions)** | Máy chủ của GitHub tự chạy test mỗi khi có PR. Dấu ✓ xanh là test qua. |
| **pytest, test** | Đoạn code kiểm tra code khác chạy đúng. `pytest -q` chạy toàn bộ test (hiện 68 test). |
| **TDD** | Viết test trước, thấy test **fail**, rồi mới viết code cho test **pass**. Plan yêu cầu làm theo cách này. |
| **pre-commit, ruff, nbstripout** | Bộ kiểm tra tự chạy trước mỗi commit. ruff kiểm tra và định dạng code Python; nbstripout xóa output của notebook; các hook khác sửa dòng trống cuối file, dấu cách thừa… |
| **Issue, Milestone, Label** | Issue là một việc cần làm trên GitHub (mỗi task một issue). Milestone gom issue theo giai đoạn. Label đánh dấu người phụ trách (A/B) và loại việc. |
| **CODEOWNERS** | File chỉ định ai tự động được mời review khi một file bị sửa. |
| **LF / CRLF** | Hai kiểu ký tự xuống dòng (Linux/Mac dùng LF, Windows dùng CRLF). Repo thống nhất dùng LF để tránh xung đột. |
| **Codex, Claude Code** | Trợ lý AI viết code của B và của A. Cả hai đọc quy tắc chung trong `AGENTS.md`. |

---

## 4. Toàn bộ quy trình (pipeline)

```
UIT-ViSFD (CSV)
   │  Task 1: tải dữ liệu, đọc nhãn → ma trận (n, 10)
   ▼
Tiền xử lý (Task 3): chuẩn hóa, [bỏ emoji], [tách từ], có cache
   │
   ├──► TF-IDF + LR/SVM (Task 6) ............ chạy trên CPU, vài giây
   ├──► PhoBERT / ViSoBERT (Task 8, 9) ....... chạy trên Colab GPU
   └──► Bi-LSTM + fastText (Task 10) ......... chạy trên Colab GPU
   │
   ▼  mọi mô hình trả ma trận (n, 10)
Hàm đánh giá chung (Task 2) → results/runs/*.json  (bộ chạy: Task 5)
   │
   ▼
Bảng so sánh (Task 7) → results/tables/*.md
   │
   ├──► Chốt cấu hình, chạy Test 3 seed, phân tích lỗi (Task 11)
   ├──► Demo web Gradio (Task 12)
   └──► README, kiểm tra tái lập (Task 13) → Báo cáo, slide (Task 14)
```

Lệnh chạy một thí nghiệm:

```bash
python scripts/run.py --config configs/tfidf_svm.yaml
python scripts/make_tables.py
```

---

## 5. Từng task: mục tiêu, cách làm, trạng thái

Chi tiết kỹ thuật đầy đủ nằm trong [plan](superpowers/plans/2026-11-13-absa-visfd.md). Đây là bản tóm tắt bằng lời thường.

### Đã xong

**Task 1 – Khung dự án, tải dữ liệu, đọc nhãn** (A, commit trước khi khóa `main`)
- *Mục tiêu:* có nền móng để mọi task sau dùng chung.
- *Cách làm:* tạo gói Python `absa`; script `download_data.py` tải bộ dữ liệu và kiểm tra mã SHA-256 (để chắc file không hỏng hay bị thay); `data.py` đọc chuỗi nhãn `{BATTERY#Positive};…` thành ma trận số và ngược lại.
- *Công cụ:* pandas, numpy.

**Task 1b, 1c – Quy tắc làm việc nhóm** (A; 1c ở PR #30)
- *Mục tiêu:* hai người (và hai trợ lý AI) làm cùng một repo mà không giẫm chân nhau.
- *Cách làm:* CI chạy test cho mọi PR; pre-commit; khóa `main`; `CONTRIBUTING.md` (quy trình Git, ai phụ trách file nào, các quy ước không được đổi), `AGENTS.md` (luật cho AI), `docs/TEAM.md` (công cụ, họp tuần, checklist ngoài code); 29 issue và 4 milestone.

**Task 2 – Hàm đánh giá** (A làm thay B, PR #31)
- *Mục tiêu:* một hàm chấm điểm duy nhất, dùng cho mọi mô hình.
- *Cách làm:* `evaluate.py` tính các chỉ số ở mục 3.5, bằng scikit-learn. Có test với ví dụ tính tay để chắc công thức đúng.
- *Còn thiếu:* Step 5, đọc bài báo gốc để biết họ tính F1 thế nào (issue #18, B).

**Task 3 – Tiền xử lý và cache** (A, PR #34)
- *Mục tiêu:* làm sạch văn bản theo các tùy chọn bật/tắt được, để khảo sát E2, E3, E5 chỉ cần đổi cấu hình.
- *Cách làm:* `preprocess.py`: chuẩn hóa Unicode NFC và chuẩn hóa dấu tiếng Việt (underthesea), tùy chọn chữ thường / bỏ emoji / bỏ dấu câu / tách từ (underthesea hoặc VnCoreNLP) / thêm số sao. Kết quả tách từ được cache ra file.
- *Khó khăn đã xử lý:* thư viện VnCoreNLP tải mô hình bằng `wget`, mà Windows không có, nên nhóm tự viết hàm tải; ngoài ra code tự tìm Java khi máy chưa đặt biến `JAVA_HOME`.

**Task 4 – Khảo sát dữ liệu (EDA)** (A, PR #35)
- *Mục tiêu:* hiểu dữ liệu trước khi làm mô hình. Báo cáo cũng cần phần này.
- *Cách làm:* `eda.py` và `notebooks/01_eda.ipynb` ghi bảng ra `results/tables/eda_*` và 4 hình ra `results/figures/`.
- *Phát hiện chính:*
  - Dữ liệu **lệch**: GENERAL 4.866 lần nhưng STORAGE chỉ 91 lần; Trung tính hiếm hơn hẳn (trừ PRICE, nơi Trung tính lại nhiều nhất: 1.136).
  - Bình luận trung bình ~36 âm tiết, dài nhất ~200.
  - Chỉ khoảng 3,9% bình luận có emoji (nhiều nhất là 👍).
  - Có vài bình luận trùng giữa các phần (Train∩Test: 5); nhóm giữ nguyên để không phá split gốc.
  - Từ độ dài, chốt `max_length`: PhoBERT 128, ViSoBERT 224.

**Task 5 – Bộ chạy thí nghiệm** (B, PR #36)
- *Mục tiêu:* chạy mọi thí nghiệm bằng một lệnh, kết quả ghi ra file thống nhất, không ai "chạy tay rồi chép số".
- *Cách làm:*
  - `config.py` đọc YAML có kế thừa `base:`.
  - `registry.py` chọn lớp mô hình theo tên (`tfidf`, `transformer`, `bilstm`).
  - `experiment.py` chạy theo trình tự: nạp dữ liệu → tiền xử lý → train → chấm Dev (và Test nếu được phép) → ghi JSON và file dự đoán.
  - `scripts/run.py` là lệnh chạy. Có `--set model.C=2` để đổi nhanh một tham số.
- *Chốt an toàn:* chỉ chạy Test được khi cấu hình nằm trong `configs/final/`; code chặn cứng.

**Task 6 – Baseline TF-IDF và khảo sát E0–E3, E6** (B làm ở PR #36, A sửa ở PR #37)
- *Mục tiêu:* có mốc điểm đơn giản, nhanh, chạy được trên CPU.
- *Cách làm:* `baseline.py` (lớp `TfidfAbsa`): TF-IDF → LR hoặc SVM, chế độ split hoặc joint. Dò C trên Dev, rồi chạy các khảo sát.
- *Sửa sau review (PR #37):*
  - (1) Quy tắc cắt token mặc định của TF-IDF bỏ mất emoji, nên E3 ban đầu cho kết quả y hệt baseline, tức thí nghiệm vô nghĩa. Đã sửa để giữ emoji.
  - (2) Lưới C ban đầu dừng ở 4, đúng chỗ LR còn đang tăng. Đã dò tiếp tới 1024.
  - (3) File kết quả thiếu dòng trống cuối file làm pre-commit báo lỗi.
- *Kết quả:* xem mục 6.

**Task 7 – Bảng so sánh tự động** (B, PR #36)
- *Mục tiêu:* bảng trong báo cáo được **sinh từ file kết quả**, không gõ tay, để mọi con số truy ngược được.
- *Cách làm:* `report.py` và `scripts/make_tables.py` đọc `results/runs/*.json` và ghi `comparison_dev`, `ablation_E*_dev`, `sweep_E0_*` (dạng `.csv` và `.md`). Khi có nhiều seed, ô ghi dạng `80,31 ± 0,40` (trung bình ± độ lệch chuẩn).

### Chưa làm

**Task 8 – Lõi transformer** (B, issue #10). Chưa phân công lại.
- *Mục tiêu:* các mảnh ghép cho PhoBERT: đầu ra `AbsaHead` (joint hoặc split), hàm loss, hàm `decode` biến xác suất thành mã 0–3, tính trọng số lớp (E4), chọn ngưỡng trên Dev.
- *Ghi chú:* đây là "nút thắt": Task 9, 10, 12 đều cần nó. Test chạy bằng tiny transformer trên CPU, không cần GPU.

**Task 9 – Huấn luyện PhoBERT/ViSoBERT** (B, issue #11; chạy GPU: issue #19)
- *Mục tiêu:* fine-tune PhoBERT và ViSoBERT; chạy khảo sát E3–E6 trên PhoBERT.
- *Cách làm dự kiến:* `training.py` (vòng huấn luyện có early stopping), lớp `TransformerAbsa`, notebook Colab `02_colab_train.ipynb`. A chạy E3, E5 trên Colab.

**Task 10 – Bi-LSTM + fastText** (A, issue #12; GPU: issue #20)
- *Mục tiêu:* tái hiện gần đúng mô hình của bài báo gốc, để có mốc so sánh với số 84,48 / 63,06.
- *Ghi chú:* gọi là "tái hiện xấp xỉ" vì bài báo không công bố đủ chi tiết.

**Task 11 – Chốt cấu hình, chạy Test, phân tích lỗi** (cả hai, issue #13, #21, #22)
- *Mục tiêu:*
  - Chọn cấu hình tốt nhất **chỉ dựa trên Dev**, chép vào `configs/final/`, rồi chạy Test **một lần duy nhất** với 3 seed.
  - Phân tích lỗi: gán loại lỗi cho 40 câu sai (khía cạnh ngầm, mỉa mai, khen chê lẫn lộn…), chọn 8–10 ví dụ đưa vào báo cáo.

**Task 12 – Demo Gradio** (A, issue #14, #23)
- *Mục tiêu:* trang web để gõ một bình luận và thấy khía cạnh được tô màu theo cảm xúc; tab thứ hai tải lên file CSV để thống kê khen/chê theo khía cạnh.
- *Ghi chú:* đã có bản thử giao diện (ngoài repo); cần checkpoint PhoBERT từ Task 11 mới chạy thật được.

**Task 13 – README, `run_all.py`, kiểm tra tái lập** (B, issue #15; A kiểm tra: issue #24)
- *Mục tiêu:* người khác (giảng viên) làm theo README chạy lại được toàn bộ; có chế độ `--smoke` chạy thử nhanh trên CPU.

**Task 14 – Báo cáo và slide** (cả hai, issue #25–#28)
- Báo cáo `NLP_BTL_MaNhom.pdf` viết trên Overleaf, slide 10 phút, video demo dự phòng, tập vấn đáp chéo, nộp bài.

**Việc tay khác:** đăng ký đề tài (#16), chốt cách tính F1 của bài báo (#18), họp tuần (#29).

---

## 6. Kết quả hiện có và cách đọc

Tất cả trên **Dev**, seed 42, đơn vị %. Nguồn: `results/tables/comparison_dev.md`.

| Cấu hình | Thay đổi so với `tfidf_lr` | aspect macro-F1 | sentiment macro-F1 |
|---|---|---|---|
| `tfidf_lr` (LR, C=512, 1–2-gram, split) | — | 79,00 | **50,49** |
| `tfidf_svm` (SVM, C=4) | đổi LR → SVM | **80,31** | 50,20 |
| `E1_ngram_1` | chỉ unigram | 75,66 | 47,12 |
| `E1_ngram_13` | 1–3-gram | 79,50 | 49,95 |
| `E1_char_25` | n-gram ký tự 2–5 | 77,39 | 47,72 |
| `E2_word` | tách từ underthesea | 78,86 | 49,57 |
| `E3_tfidf_strip_emoji` | bỏ emoji | 78,92 | 50,23 |
| `E6_tfidf_joint` | chế độ joint | 77,48 | 49,26 |

**Đọc thế nào:**
- **Phát hiện khía cạnh khá dễ** (~80%), **phân loại đúng cảm xúc khó hơn nhiều** (~50%). Phần cảm xúc là chỗ PhoBERT được kỳ vọng vượt TF-IDF.
- **Bigram giúp rõ** (75,7 → 79,0 aspect): cụm 2 từ như "không tốt", "pin trâu" mang nghĩa mà từ đơn không có. Lên 3-gram thì không thêm được gì.
- **Tách từ không giúp TF-IDF**, vì bigram âm tiết đã tự bắt được các từ ghép như "điện thoại".
- **Bỏ emoji làm giảm nhẹ** (50,49 → 50,23): emoji có mang chút tín hiệu cảm xúc, nhưng chỉ ~4% bình luận có emoji.
- **Split tốt hơn joint** cho TF-IDF.
- **Theo từng khía cạnh** (SVM): BATTERY 96,5 và CAMERA 92,3 rất tốt; **STORAGE chỉ 36,4** vì chỉ có 91 mẫu để học. Đây là chỗ đáng phân tích trong báo cáo.
- Các khác biệt dưới ~0,5 điểm với **một seed** có thể chỉ là ngẫu nhiên; đừng kết luận mạnh từ chúng.

**Chưa có:** kết quả PhoBERT, ViSoBERT, Bi-LSTM; kết quả trên Test; so sánh với bài báo gốc.

---

## 7. Thư mục nào chứa gì

| Đường dẫn | Chứa gì | Lên Git? |
|---|---|---|
| `src/absa/` | Code chính, mỗi file một bước (data, preprocess, eda, evaluate, config, registry, experiment, baseline, report) | Có |
| `scripts/` | Lệnh chạy: `download_data.py`, `run.py`, `make_tables.py` | Có |
| `configs/` | File YAML cấu hình thí nghiệm; `ablation/` cho E1–E6; `smoke/` để chạy thử | Có |
| `tests/` | Test (pytest) | Có |
| `notebooks/` | Notebook EDA (sau này thêm Colab, phân tích lỗi) | Có (đã xóa output) |
| `results/runs/` | Một file JSON cho mỗi lần chạy | Có |
| `results/tables/`, `results/figures/` | Bảng và hình sinh tự động | Có |
| `results/predictions/` | Dự đoán từng câu (lớn) | Không, chia sẻ qua Drive |
| `data/` | Dữ liệu tải về và cache | Không (giấy phép) |
| `models/` | Checkpoint mô hình | Không, chia sẻ qua Drive |
| `docs/spec/` | Bản phác thảo dự án (yêu cầu gốc) | Có |
| `docs/superpowers/plans/` | Plan chi tiết 14 task | Có |
| `docs/decisions.md` | Mọi quyết định và lý do (D1–D14, nhật ký khi chạy) | Có |
| `CONTRIBUTING.md`, `docs/TEAM.md`, `AGENTS.md` | Quy trình nhóm, công cụ, luật cho AI | Có |

Thứ tự ưu tiên khi các tài liệu mâu thuẫn nhau: spec → plan → `decisions.md` → code.
