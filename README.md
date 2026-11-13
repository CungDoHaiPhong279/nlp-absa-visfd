# Phân tích cảm xúc theo khía cạnh trên UIT-ViSFD

Bài tập lớn môn Xử lý ngôn ngữ tự nhiên. Với mỗi bình luận điện thoại tiếng Việt, hệ thống xác định khía cạnh nào được nhắc đến (pin, camera, giá…) và cảm xúc với từng khía cạnh. Nhóm so sánh baseline TF-IDF với PhoBERT, ViSoBERT và Bi-LSTM.

> README đầy đủ (cách tái tạo từng bảng/hình, thời gian chạy) sẽ được viết ở Task 13.

## Thành viên mới: đọc theo thứ tự

1. [docs/TEAM.md](docs/TEAM.md): công cụ, quy định nhóm, checklist việc ngoài code.
2. [CONTRIBUTING.md](CONTRIBUTING.md): cài đặt, quy trình nhánh/PR, ai phụ trách file nào, các bất biến không được đổi.
3. [AGENTS.md](AGENTS.md): quy tắc cho trợ lý AI (Claude Code, Gemini CLI tự đọc).
4. [Plan](docs/superpowers/plans/2026-11-13-absa-visfd.md): yêu cầu chi tiết từng task; [spec](docs/spec/phac_thao_du_an_ABSA_UIT-ViSFD.md); [quyết định](docs/decisions.md).

Tiến độ: xem [Issues](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/issues) và [Milestones](https://github.com/CungDoHaiPhong279/nlp-absa-visfd/milestones).

## Bắt đầu nhanh (Windows)

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e . -r requirements.txt -r requirements-dev.txt
pre-commit install
python scripts/download_data.py
pytest -q
```

Chi tiết và lưu ý (không dùng Python của MSYS2, cần Java 17+): [CONTRIBUTING.md mục 2](CONTRIBUTING.md#2-cài-đặt).

## Dữ liệu

UIT-ViSFD (11.122 bình luận; Train 7.786 · Dev 1.112 · Test 2.224), tải bằng `scripts/download_data.py` từ <https://github.com/LuongPhan/UIT-ViSFD>. Dữ liệu chỉ dùng cho nghiên cứu và **không** được commit vào repo này.

Phan, L. L. et al. (2021). *SA2SL: From Aspect-Based Sentiment Analysis to Social Listening System for Business Intelligence.* KSEM 2021, Springer, pp. 647–658.
