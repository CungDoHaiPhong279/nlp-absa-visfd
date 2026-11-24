import importlib.util
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

_spec = importlib.util.spec_from_file_location(
    "demo_app", Path(__file__).resolve().parents[1] / "demo" / "app.py"
)
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)


def _write(tmp_path: Path, text: str, encoding: str = "utf-8") -> Path:
    path = tmp_path / "c.csv"
    path.write_bytes(text.encode(encoding))
    return path


def test_read_comments_comma_and_quoted_commas(tmp_path):
    df = app.read_comments(_write(tmp_path, 'id,comment\n1,"Máy đẹp, pin yếu"\n2,ok\n'), 10)
    assert df["comment"].tolist() == ["Máy đẹp, pin yếu", "ok"]


def test_read_comments_semicolon_and_bom(tmp_path):
    df = app.read_comments(_write(tmp_path, "﻿comment;n_star\npin trâu;5\n"), 10)
    assert df["comment"].tolist() == ["pin trâu"]


def test_read_comments_keeps_na_like_text(tmp_path):
    df = app.read_comments(_write(tmp_path, "comment\nNA\nNone\nok\n"), 10)
    assert df["comment"].tolist() == ["NA", "None", "ok"]


def test_read_comments_stops_after_max_rows_plus_one(tmp_path):
    text = "comment\n" + "".join(f"c{i}\n" for i in range(50))
    assert len(app.read_comments(_write(tmp_path, text), 5)) == 6


@pytest.mark.parametrize(
    ("text", "encoding"),
    [("", "utf-8"), ("comment\nGiá hơi chát\n", "cp1258"), ("id,text\n1,a\n", "utf-8")],
)
def test_read_comments_bad_files_give_clear_error(tmp_path, text, encoding):
    with pytest.raises(gr.Error):
        app.read_comments(_write(tmp_path, text, encoding), 10)
