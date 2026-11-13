# Hướng dẫn làm việc nhóm

Công cụ, quy định nhóm và checklist ngoài code: xem [docs/TEAM.md](docs/TEAM.md).

Repo có hai thành viên (A và B), mỗi người có thể dùng trợ lý AI. File này giúp hai người không lệch hướng và không xung đột khi merge.

## 1. Nguồn sự thật

Thứ tự ưu tiên khi các file mâu thuẫn nhau:

1. `docs/spec/phac_thao_du_an_ABSA_UIT-ViSFD.md`
2. `docs/superpowers/plans/2026-11-13-absa-visfd.md`
3. `docs/decisions.md`

Code mâu thuẫn với các file này là lỗi.

## 2. Cài đặt

Windows (PowerShell). **Không dùng Python của MSYS2.**

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e . -r requirements.txt -r requirements-dev.txt
pre-commit install
python scripts/download_data.py
pytest -q
```

- VnCoreNLP cần Java 17 trở lên.
- Colab: dùng `notebooks/02_colab_train.ipynb` (tạo ở Task 9).

## 3. Quy trình một task

- Tạo nhánh từ `main` mới nhất: `task/<số task>-<tên-ngắn>`, ví dụ `task/6-tfidf-baseline`.
- Mỗi task có một GitHub Issue (tiêu đề `Task N: …`). Tên nhánh dùng **số task** trong plan; mô tả PR dùng **số issue**: `Closes #<số issue>`. Hai số này khác nhau, ví dụ Task 2 là issue #4.
- Một PR = một task trong plan. Tick checkbox của task trong plan ngay trong PR đó.
- Trước khi mở PR: `git pull --rebase origin main`.
- Không push thẳng vào `main`. Không force-push nhánh của người khác.

## 4. Commit message

Conventional Commits bằng tiếng Anh như plan: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`. Không có dòng `Co-Authored-By` hay bất kỳ dòng ghi công AI nào.

## 5. Phân công và quyền sở hữu file

Người phụ trách viết/sửa; người kia review PR. Sửa file của người kia thì báo trước trong nhóm chat.

| File / thư mục | Phụ trách | Review |
|---|---|---|
| `src/absa/data.py`, `preprocess.py`, `eda.py`, `baseline.py`, `report.py`, `bilstm.py`, `inference.py` | A | B |
| `demo/`, `notebooks/01_eda.ipynb` | A | B |
| `scripts/download_data.py`, `scripts/make_tables.py`, `scripts/prepare_fasttext.py` | A | B |
| `configs/tfidf_*.yaml`, `configs/bilstm_fasttext.yaml` | A | B |
| `configs/ablation/E1_*`, `E2_*`, `E3_tfidf_*`, `E6_tfidf_*` | A | B |
| `src/absa/evaluate.py`, `config.py`, `registry.py`, `experiment.py`, `tiny.py`, `transformer.py`, `training.py` | B | A |
| `scripts/run.py`, `scripts/run_all.py` | B | A |
| `configs/phobert_*.yaml`, `configs/visobert_*.yaml`, `configs/ablation/*_phobert_*` | B | A |
| `notebooks/02_colab_train.ipynb`, `README.md` | B | A |
| `src/absa/errors.py`, `notebooks/03_error_analysis.ipynb`, `configs/final/` | Cả hai | Cả hai đồng ý |
| `docs/`, `.github/`, `pyproject.toml`, `requirements*.txt` | Cả hai | Cả hai đồng ý |

## 6. Bất biến: không tự ý đổi

- Thứ tự khía cạnh: `("SCREEN", "CAMERA", "FEATURES", "BATTERY", "PERFORMANCE", "STORAGE", "DESIGN", "PRICE", "GENERAL", "SER&ACC")`.
- Mã cảm xúc: `0=None, 1=Positive, 2=Neutral, 3=Negative`. `POLARITIES = ("Positive", "Neutral", "Negative")`.
- Split gốc: Train 7.786 · Dev 1.112 · Test 2.224. Không trộn, không chia lại.
- Chỉ chọn tham số trên Dev. `--eval-test` chỉ dùng với cấu hình trong `configs/final/` và chỉ chạy một lần, khi cả hai đồng ý.
- Định nghĩa các chỉ số trong `evaluate.py`; tên khóa JSON của mỗi run.
- Chữ ký các interface trong mục "Interfaces" của từng task trong plan.
- Seed mặc định 42; thí nghiệm nhiều seed dùng `[42, 43, 44]`.

Muốn đổi: mở một PR chỉ sửa `docs/decisions.md` (ghi lý do và ảnh hưởng), cả hai duyệt, rồi mới sửa code.

## 7. Thí nghiệm và kết quả

- Chạy bằng `scripts/run.py`. Không chạy tay trong notebook rồi chép số.
- Commit `results/runs/*.json` (D13).
- `results/tables/*` là file sinh ra. Khi xung đột thì không sửa tay: chạy lại `python scripts/make_tables.py` rồi commit kết quả.
- `results/predictions/` và `models/` chia sẻ qua Google Drive (thư mục `nlp-absa-visfd` trên Drive của nhóm).
- Không commit dữ liệu UIT-ViSFD (giấy phép chỉ cho nghiên cứu, repo public).

## 8. Notebook

- `nbstripout` xóa output khi commit (qua pre-commit).
- Hình và bảng phải được code ghi ra `results/`.
- Mỗi notebook có một người phụ trách; người kia không sửa song song.

## 9. Trước khi mở PR

- [ ] `pytest -q` xanh
- [ ] `pre-commit run --all-files` sạch
- [ ] Đã cập nhật `docs/decisions.md` nếu có quyết định mới
- [ ] Không có file trong `data/`, `models/`, `.venv/`
- [ ] Nếu nâng ruff: sửa cả `requirements-dev.txt` và `rev` ruff trong `.pre-commit-config.yaml` cùng lúc

## 10. Dùng trợ lý AI

- Trợ lý phải đọc `AGENTS.md`.
- Người dùng AI vẫn chịu trách nhiệm hiểu và giải thích được code (yêu cầu môn học).
- Ghi phạm vi dùng AI vào báo cáo.
