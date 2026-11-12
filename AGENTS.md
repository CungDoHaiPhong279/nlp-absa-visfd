# Quy tắc cho trợ lý AI

- Đọc `CONTRIBUTING.md`, plan (`docs/superpowers/plans/2026-11-13-absa-visfd.md`) và `docs/decisions.md` trước khi sửa code. Làm đúng task được giao, đúng các file trong mục "Files" của task.
- Không đổi các bất biến (thứ tự `ASPECTS`, mã cảm xúc 0/1/2/3, split gốc, seed, chỉ số trong `evaluate.py`, tên khóa JSON của run; chi tiết ở `CONTRIBUTING.md` mục 6). Không đổi chữ ký interface đã có. Không thêm thư viện ngoài `requirements*.txt` mà không ghi vào `docs/decisions.md`.
- TDD theo plan. Chạy `pytest -q` và `pre-commit run --all-files` trước khi báo xong.
- Không chạy `--eval-test`. Không commit/push nếu người dùng chưa đồng ý. Không thêm dòng `Co-Authored-By` hay ghi công AI vào commit/PR.
- Không sửa `results/tables/*` bằng tay. Không commit `data/`, `models/`, `results/predictions/`.
- Python: dùng `.venv` (Python 3.11), không dùng Python của MSYS2.
- Gặp mâu thuẫn giữa plan và code: dừng lại và hỏi người dùng, không tự chọn.
