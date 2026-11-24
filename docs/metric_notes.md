# Ghi chú về chỉ số của bài báo gốc

Nguồn đã đọc: Phan et al. (2021), *SA2SL: From Aspect-Based Sentiment Analysis to Social
Listening System for Business Intelligence*, KSEM 2021. Bản đọc là arXiv:2105.15079v2 (bản HTML
<https://arxiv.org/html/2105.15079v2>, đọc ngày 2026-11-24). Bản HTML không có số trang nên trích
dẫn ghi theo số mục và số bảng; bản Springer tương ứng là trang 647–658.

## Trả lời bốn câu hỏi

- **(a) F1 khía cạnh 84,48 là micro, macro hay weighted?** Macro. Mục 5.2 ghi: "precision,
  recall, and F1-score (macro average)". Hàng cuối Bảng 5 tên là "Macro Avg" (khía cạnh
  85,77 / 83,22 / 84,48). Bài báo không đưa công thức.
  - Bảng 5 có **11 hàng**, gồm cả `Others` (F1 60,82). Trung bình 11 F1 khía cạnh trong Bảng 5
    là khoảng 84,42, gần 84,48. Như vậy macro của bài báo **có tính `Others`**, còn
    `aspect_macro_f1` của nhóm chỉ lấy 10 khía cạnh (plan, Global Constraints: `{OTHERS}` không phải khía cạnh).
  - Precision khía cạnh ở Bảng 4 là 87,55 nhưng ở Bảng 5 là 85,77. Bài báo không giải thích.
- **(b) F1 cảm xúc 63,06 tính trên toàn bộ cặp, hay chỉ trên khía cạnh phát hiện đúng?** Bài báo
  **không nói**. Mục 5.2 chỉ ghi `Others` không có cảm xúc nên để NaN. Bảng 5 cho F1 cảm xúc theo
  từng khía cạnh (10 hàng), nhưng trung bình 10 hàng đó là khoảng 63,91, không bằng 63,06. Không
  suy ra được cách tính.
- **(c) Dev hay Test?** Bài báo **không nói**. Mục 3.4 chia Train/Dev/Test theo tỉ lệ 7:1:2
  (7.786 / 1.112 / 2.224 bình luận, khớp split nhóm dùng), nhưng Bảng 4 và 5 không ghi tập nào.
- **(d) Ánh xạ sang khóa của `evaluate`:** **không cùng định nghĩa.**
  - 84,48 gần nghĩa nhất với `aspect_macro_f1`, nhưng khác ở chỗ có tính `Others`.
  - 63,06 không xác định được: không rõ có điều kiện theo khía cạnh đúng hay không, và không khớp
    trung bình các hàng trong Bảng 5.
  - Vì vậy giữ `published_metric: null`; bảng so sánh ghi "không cùng định nghĩa: 84.48 / 63.06".
    Khi viết báo cáo, chỉ so sánh định tính (cùng bộ dữ liệu, cùng split, cách tính khác).

## Số liệu bài báo (để dẫn trong báo cáo)

Bảng 4 (mục 5.2), định dạng P / R / F1 khía cạnh ; P / R / F1 cảm xúc:

| Mô hình | Khía cạnh | Cảm xúc |
|---|---|---|
| Naive Bayes | 72,18 / 59,53 / 64,65 | 49,07 / 30,43 / 37,56 |
| SVM | 36,45 / 51,34 / 42,63 | 16,09 / 23,35 / 19,69 |
| Random Forest | 45,72 / 50,15 / 47,83 | 17,11 / 24,56 / 20,17 |
| CNN | 77,25 / 63,49 / 69,70 | 33,34 / 22,92 / 27,16 |
| LSTM | 82,61 / 78,05 / 80,27 | 56,51 / 48,39 / 52,13 |
| Bi-LSTM + fastText ("Our Approach") | 87,55 / 83,22 / 84,48 | 65,82 / 60,53 / 63,06 |

F1 theo khía cạnh của Bi-LSTM (Bảng 5), khía cạnh ; cảm xúc: SCREEN 87,26 ; 62,30 · CAMERA
85,17 ; 74,69 · FEATURES 88,64 ; 64,77 · BATTERY 94,60 ; 72,73 · PERFORMANCE 88,77 ; 66,75 ·
STORAGE 76,48 ; 30,10 · DESIGN 87,85 ; 71,41 · PRICE 90,19 ; 71,41 · GENERAL 81,72 ; 65,72 ·
SER&ACC 87,16 ; 59,22 · OTHERS 60,82 ; NaN.

Mô hình "Our Approach" (mục 4) là Bi-LSTM có thêm một lớp tích chập và hai lớp pooling, dùng
fastText. Bi-LSTM của nhóm (Task 10) chỉ tái hiện xấp xỉ.

```yaml
published_metric: null
```
