from absa.data import ASPECTS, labels_to_matrix, matrix_to_labels
from absa.inference import Predictor, render_html, summarize_predictions
from absa.transformer import TransformerAbsa


def test_render_negative_color() -> None:
    out = render_html("pin yếu", {"BATTERY": "Negative"})
    assert "Pin" in out and "#cf222e" in out


def test_render_empty() -> None:
    out = render_html("", {})
    assert out.startswith("<div") and out.endswith("</div>")
    assert "Không phát hiện" in out


def test_render_escapes_script() -> None:
    out = render_html("<script>alert(1)</script>", {})
    assert "<script>" not in out and "&lt;script&gt;" in out


class _Boom:
    def __call__(self, *args, **kwargs):
        raise AssertionError("model must not be called")


def test_predict_empty_and_emoji_only() -> None:
    p = object.__new__(Predictor)
    p.module = _Boom()
    p.tokenizer = _Boom()
    assert p.predict("") == {}
    assert p.predict("😍😍") == {}
    assert p.predict("!!! ...") == {}


def test_summarize_predictions() -> None:
    t = summarize_predictions([{"BATTERY": "Negative"}, {"BATTERY": "Positive"}, {}])
    assert t.loc["BATTERY", "Total"] == 2


def test_from_dir_roundtrip(tmp_path) -> None:
    texts = ["pin trâu", "pin yếu", "máy đẹp", "giá cao"]
    Y = labels_to_matrix(
        [{"BATTERY": "Positive"}, {"BATTERY": "Negative"}, {"DESIGN": "Positive"}, {}]
    )
    model = TransformerAbsa("__tiny__", epochs=1, batch_size=4, fp16=False).fit(texts, Y)
    model.preprocess_cfg = {"lowercase": True}
    model.save(tmp_path / "ckpt")
    p = Predictor.from_dir(tmp_path / "ckpt")
    pred = p.predict("pin trâu")
    assert set(pred) <= set(ASPECTS)
    assert pred == matrix_to_labels(model.predict(["pin trâu"]))[0]
    assert p.predict_batch(["pin trâu", "", "máy đẹp"])[1] == {}
