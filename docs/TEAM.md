# Công cụ, quy định nhóm và checklist ngoài code

## 1. Mục đích

File này cho hai thành viên (A, B) biết nhóm dùng công cụ gì, làm việc với nhau theo quy định nào và còn những việc ngoài code nào phải làm. Thứ tự đọc: `docs/TEAM.md` (con người, công cụ) → [CONTRIBUTING.md](../CONTRIBUTING.md) (Git) → [AGENTS.md](../AGENTS.md) (trợ lý AI) → [plan](superpowers/plans/2026-11-13-absa-visfd.md).

## 2. Công cụ

| Công cụ | Dùng cho | Ghi chú / quy định |
|---|---|---|
| Python 3.11 + `.venv` | Chạy code, test | Không dùng Python của MSYS2. Cài đặt: [CONTRIBUTING.md mục 2](../CONTRIBUTING.md) |
| Git + GitHub | Repo, Issues (checklist), Milestones (giai đoạn), Pull Request, Actions (CI) | Quy trình nhánh/PR: [CONTRIBUTING.md mục 3](../CONTRIBUTING.md). Cách dùng Issues: mục 7 dưới đây |
| VS Code (khuyến nghị) + extension Python, Jupyter, Ruff, EditorConfig | Soạn code và notebook | Trình soạn khác cũng được nếu tôn trọng `.editorconfig` |
| Java 17+ | VnCoreNLP tách từ cho PhoBERT | Bắt buộc khi chạy tiền xử lý cho PhoBERT |
| pre-commit, ruff, nbstripout, pytest | Kiểm tra định dạng, xóa output notebook, chạy test | Xem [CONTRIBUTING.md mục 9](../CONTRIBUTING.md) |
| Jupyter Notebook | EDA, phân tích lỗi | Output bị xóa khi commit (nbstripout) |
| Google Colab (GPU T4) | Huấn luyện PhoBERT, ViSoBERT, Bi-LSTM | Dùng `notebooks/02_colab_train.ipynb` |
| Google Drive | Chia sẻ predictions, checkpoint, cache | Thư mục chung `nlp-absa-visfd/`, cấu trúc ở mục 4 |
| Hugging Face Hub | Tải `vinai/phobert-base-v2`, `uitnlp/visobert` | Không cần token |
| Thư viện chính: pandas, scikit-learn, underthesea, py_vncorenlp, PyTorch, transformers, Gradio, matplotlib | Xử lý dữ liệu, mô hình, demo, biểu đồ | Phiên bản: `requirements.txt` và `requirements.lock*.txt` |
| Overleaf (LaTeX) | Báo cáo `NLP_BTL_MaNhom.pdf` và slide (Beamer) | Xem mục 5 |
| Zalo | Trao đổi hằng ngày | Chỉ trao đổi nhanh, xem mục 3 |
| Google Meet | Họp tuần | Xem mục 3 |
| Trợ lý AI (Claude Code, Gemini CLI…) | Hỗ trợ viết code | Theo [AGENTS.md](../AGENTS.md); ghi phạm vi dùng vào phụ lục báo cáo |

## 3. Quy định nhóm

- **Họp tuần:** Google Meet mỗi tuần một lần (giờ cố định do nhóm chốt). Mỗi người demo phần mình đã làm, **giải thích lại phần của người kia** (rủi ro vấn đáp, xem [spec mục 6](spec/phac_thao_du_an_ABSA_UIT-ViSFD.md)), cập nhật Issues. Ghi biên bản ngắn vào một comment trong issue "Họp tuần" ([#29](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/issues/29)) hoặc vào Zalo note.
- **Zalo chỉ để trao đổi nhanh.** Mọi quyết định kỹ thuật phải nằm trong Issue, PR hoặc [docs/decisions.md](decisions.md); quyết định chỉ có trên Zalo coi như chưa có.
- **Nhận việc:** ai nhận việc thì gán nhãn `A`/`B` cho issue và chuyển sang "đang làm" bằng comment; không hai người cùng làm một issue.
- **Định nghĩa "xong" một task:** test xanh trên máy và CI; PR được người kia duyệt và merge; issue đóng qua `Closes #`; checkbox trong plan được tick; `docs/decisions.md` cập nhật nếu có quyết định mới.
- **Chạy GPU trên Colab:** trước khi chạy, comment trên issue cấu hình sẽ chạy (tránh hai người chạy trùng, tốn quota); sau mỗi lần chạy sao lưu lên Drive; commit `results/runs/*.json` bằng PR.
- **Liêm chính học thuật** (hướng dẫn môn học mục 6): không bịa số liệu; mọi số trong báo cáo truy được về `results/runs/` hoặc `results/tables/`; trích dẫn Phan et al. (2021) và mọi nguồn; không sao chép của nhóm khác.
- **Dữ liệu và bảo mật:** không đăng lại UIT-ViSFD (chỉ dùng cho nghiên cứu); không commit token, mật khẩu, file `.env`; repo đang public.

## 4. Google Drive

```
nlp-absa-visfd/
├── data_processed/     # bản sao data/processed (cache tách từ)
├── results_predictions/
├── models/             # chỉ checkpoint final_* (cấu hình có save: true)
├── fasttext/           # cc.vi.300.vec.gz và file đã lọc
└── backups/            # sao lưu results/ theo ngày chạy
```

Quy định: không xóa file của người kia; tên file giữ nguyên như code sinh ra.

## 5. Overleaf

- Một project chung, cả hai có quyền sửa.
- Cấu trúc: `main.tex`, `sections/01_gioi_thieu.tex` … `07_phan_cong.tex` (theo [spec mục 9](spec/phac_thao_du_an_ABSA_UIT-ViSFD.md)), `figures/` (copy từ `results/figures/`), `tables/`.
- Số trong bảng chép từ `results/tables/*.md` và phải khớp.
- Dùng tính năng History của Overleaf thay cho gửi file qua lại.
- Slide: `slides.tex` (Beamer) trong cùng project.
- Ai viết mục nào: theo cột "Phụ trách" ở plan Task 14 và [spec mục 7](spec/phac_thao_du_an_ABSA_UIT-ViSFD.md).

## 6. Checklist việc ngoài code

Mỗi mục có issue tương ứng (xem Issues). Người phụ trách theo [spec mục 7](spec/phac_thao_du_an_ABSA_UIT-ViSFD.md). "Plan Task N" là [plan](superpowers/plans/2026-11-13-absa-visfd.md).

- [ ] Đăng ký đề tài (phiếu ở [spec mục 11](spec/phac_thao_du_an_ABSA_UIT-ViSFD.md)) — cả hai ([#16](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/issues/16))
- [ ] Thêm thành viên B làm collaborator trên GitHub, điền username B vào `.github/CODEOWNERS` — A ([#17](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/issues/17))
- [ ] Đọc bài báo gốc, chốt cách tính F1 vào `docs/metric_notes.md` (plan Task 2 Step 5) — B ([#18](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/issues/18))
- [ ] Chạy GPU: đo thời gian PhoBERT, 6 cấu hình dev (plan Task 9 Step 8–9) — B (A chạy E3, E5) ([#19](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/issues/19))
- [ ] Chạy GPU: Bi-LSTM + fastText (plan Task 10 Step 4) — A ([#20](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/issues/20))
- [ ] Chốt cấu hình final và chạy Test 3 seed, **một lần**, khi cả hai đồng ý (plan Task 11 Step 4–6) — cả hai ([#21](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/issues/21))
- [ ] Gán nhãn 40 lỗi, chọn 8–10 ví dụ (plan Task 11 Step 7) — B ([#22](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/issues/22))
- [ ] Chạy demo thật, chụp màn hình (plan Task 12 Step 4) — A ([#23](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/issues/23))
- [ ] Kiểm tra tái lập trên máy khác theo README (plan Task 13 Step 6) — A ([#24](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/issues/24))
- [ ] Viết báo cáo trên Overleaf theo dàn ý (plan Task 14) — cả hai; kiểm tra chéo số liệu ([#25](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/issues/25))
- [ ] Slide 10 phút + video demo dự phòng — cả hai ([#26](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/issues/26))
- [ ] Tập thuyết trình và vấn đáp chéo — cả hai ([#27](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/issues/27))
- [ ] Nộp bài: PDF `NLP_BTL_MaNhom.pdf`, link repo (cấp quyền xem cho giảng viên), README, thư mục `results/` — cả hai ([#28](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/issues/28))

## 7. GitHub Issues

Danh sách việc sống nằm ở tab Issues của `CungDoHaiPhong279/nlp-absa-visfd`.

- Mỗi task trong plan một issue, tiêu đề `Task N: …`.
- Nhãn: người phụ trách (`A`, `B`, `cả-hai`) và loại việc (`code`, `gpu-colab`, `thủ-công`, `báo-cáo`).
- Milestone theo giai đoạn (không có ngày): "Giai đoạn 1 – Dữ liệu và nền tảng", "Giai đoạn 2 – Mô hình và khảo sát", "Giai đoạn 3 – Chốt kết quả", "Giai đoạn 4 – Báo cáo và bảo vệ".
- Xem tiến độ ở tab Issues → Milestones, hoặc [danh sách milestone](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/milestones).
- Issue chỉ đóng qua PR (`Closes #N`) hoặc khi việc thủ công xong và người kia xác nhận trong comment.
