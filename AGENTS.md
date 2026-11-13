# Hướng dẫn cho trợ lý AI (Codex, Claude Code, Gemini CLI)

File này là nguồn quy tắc duy nhất cho mọi trợ lý AI trong repo. `CLAUDE.md` và `GEMINI.md` chỉ import file này.

## 1. Dự án là gì

Bài tập lớn NLP của 2 sinh viên: phân tích cảm xúc theo khía cạnh (ABSA) trên bộ UIT-ViSFD (bình luận điện thoại tiếng Việt). Mỗi bình luận được gán tập cặp (khía cạnh, cảm xúc) cho 10 khía cạnh. Nhóm so sánh baseline TF-IDF với PhoBERT, ViSoBERT và Bi-LSTM + fastText; có khảo sát E0–E6, phân tích lỗi và demo Gradio.

- Thành viên A: `@CungDoHaiPhong279` (dùng Claude Code). Thành viên B: `@Longdaica11` (dùng Codex).
- Tiến độ: GitHub Issues và Milestones của `CungDoHaiPhong279/nlp-absa-visfd`. Mỗi task trong plan có một issue tiêu đề `Task N: …`.

## 2. Đọc trước khi làm

1. Issue của task được giao.
2. Đúng mục `### Task N` trong [plan](docs/superpowers/plans/2026-11-13-absa-visfd.md): các phần **Files**, **Interfaces**, các **Step**. Đọc thêm mục "Global Constraints" của plan.
3. [docs/decisions.md](docs/decisions.md): các quyết định đã chốt (D1–D14 và nhật ký).
4. [CONTRIBUTING.md](CONTRIBUTING.md) mục 5–6: ai phụ trách file nào, các bất biến.

Thứ tự ưu tiên khi mâu thuẫn: spec (`docs/spec/`) → plan → `docs/decisions.md` → code.

## 3. Kiến trúc và quy ước cốt lõi

- Gói Python `absa` ở `src/absa/`, mỗi module một bước pipeline. Test ở `tests/`, script CLI ở `scripts/`, cấu hình thí nghiệm YAML ở `configs/`.
- **Ma trận nhãn:** mọi mô hình nhận `list[str]` và trả `np.ndarray` int8 shape `(n, 10)`, mã `0=None, 1=Positive, 2=Neutral, 3=Negative`. Cột theo đúng thứ tự `absa.data.ASPECTS`. Một hàm `absa.evaluate.evaluate` dùng chung cho mọi mô hình.
- `{OTHERS}` không phải khía cạnh; bình luận chỉ có `{OTHERS}` là hàng toàn 0.
- Đã có: `absa.data` (`ASPECTS`, `POLARITIES`, `POLARITY_CODE`, `parse_label`, `has_others`, `format_label`, `labels_to_matrix`, `matrix_to_labels`, `load_split`) và `absa.evaluate` (`evaluate`, `confusion_by_aspect`). Dùng lại, không viết lại.
- Module chưa có thì tạo đúng tên và chữ ký trong mục Interfaces của task; task sau phụ thuộc vào đúng tên đó.

## 4. Lệnh

Windows (PowerShell), từ thư mục gốc repo:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e . -r requirements.txt -r requirements-dev.txt
pre-commit install
python scripts/download_data.py
```

Linux (Codex cloud, Colab, CI). Dùng làm "setup script" của môi trường Codex cloud; cần mạng ở bước này:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e . -r requirements.txt -r requirements-dev.txt
python scripts/download_data.py
```

Kiểm tra (chạy trước khi báo xong):

```bash
pytest -q
pre-commit run --all-files    # nếu không có pre-commit: ruff check . && ruff format --check .
```

- Không có dữ liệu (`data/raw/` trống, ví dụ sandbox không có mạng) thì test dùng dữ liệu thật tự **skip**. Đó là hành vi đúng; không sửa điều kiện skip.
- Trên Windows không dùng Python của MSYS2 (`C:\msys64\...`); chỉ dùng `.venv` Python 3.11.
- Ruff đã ghim `0.17.5` ở cả `requirements-dev.txt` và `.pre-commit-config.yaml`.

## 5. Quy trình làm một task

1. Nhánh mới từ `main` mới nhất: `task/<số task>-<tên-ngắn>` (số task trong plan, không phải số issue).
2. TDD đúng như plan: viết test của task (giá trị trong plan là bắt buộc), chạy thấy FAIL, cài đặt, chạy thấy PASS.
3. Chỉ sửa file liệt kê trong mục Files của task. Cần sửa file khác thì dừng lại hỏi.
4. Tick checkbox các Step đã xong trong plan, ngay trong cùng PR.
5. Chạy các lệnh kiểm tra ở mục 4.
6. PR theo template `.github/pull_request_template.md`, mô tả có `Closes #<số issue>`. Một PR = một task.

## 6. Quy tắc cứng

- Không đổi bất biến: thứ tự `ASPECTS`, mã 0–3, split gốc 7.786/1.112/2.224, seed 42 / `[42, 43, 44]`, định nghĩa chỉ số trong `evaluate.py`, tên khóa JSON của run, chữ ký interface đã có. Muốn đổi: chỉ đề xuất qua PR sửa `docs/decisions.md` (CONTRIBUTING mục 6).
- Không thêm thư viện ngoài `requirements*.txt` mà không ghi vào `docs/decisions.md`.
- Không chạy `--eval-test` (đánh giá tập Test chỉ một lần, do cả nhóm quyết định).
- Không commit/push/mở PR khi người dùng chưa đồng ý. Không thêm dòng `Co-Authored-By` hay ghi công AI vào commit hay PR.
- Không sửa tay `results/tables/*` (chạy lại `scripts/make_tables.py`). Không commit `data/`, `models/`, `results/predictions/`. Có commit `results/runs/*.json`.
- Không đóng issue thay người dùng; issue đóng qua `Closes #` khi PR merge.
- Gặp mâu thuẫn giữa plan và code, hoặc yêu cầu không rõ: dừng lại và hỏi, không tự chọn.
- Viết tài liệu bằng tiếng Việt có dấu; code, tên hàm, commit message bằng tiếng Anh.
