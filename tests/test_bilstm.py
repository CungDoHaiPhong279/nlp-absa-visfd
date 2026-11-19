import gzip
import json

import numpy as np
import torch

from absa.bilstm import PAD, UNK, BiLstmAbsa, build_vocab, encode, load_fasttext
from absa.data import labels_to_matrix

TEXTS = ["pin trâu", "pin yếu", "máy đẹp", "giá cao", "camera nét", "hihi"]
Y = labels_to_matrix(
    [
        {"BATTERY": "Positive"},
        {"BATTERY": "Negative"},
        {"DESIGN": "Positive"},
        {"PRICE": "Negative"},
        {"CAMERA": "Positive"},
        {},
    ]
)


def _model(**kw) -> BiLstmAbsa:
    return BiLstmAbsa(**{"emb_dim": 8, "hidden": 8, "epochs": 1, "batch_size": 4, **kw})


def test_build_vocab_order() -> None:
    assert build_vocab(["b a a", "c"])[:3] == ["<pad>", "<unk>", "a"]
    assert (PAD, UNK) == (0, 1)


def test_encode_oov_and_empty() -> None:
    vocab = build_vocab(["pin trâu"])
    assert encode(["xyz", ""], vocab, 5) == [[1], [1]]


def test_bilstm_truncates() -> None:
    long_text = " ".join(["pin"] * 500)
    assert len(encode([long_text], build_vocab(TEXTS), 128)[0]) == 128
    model = _model(max_len=128).fit(TEXTS, Y)
    assert model.predict([long_text]).shape == (1, 10)


def test_output_shape() -> None:
    pred = _model().fit(TEXTS, Y, TEXTS[:2], Y[:2]).predict(TEXTS)
    assert pred.shape == (6, 10) and pred.dtype == np.int8
    assert set(np.unique(pred)) <= {0, 1, 2, 3}


def test_padding_invariance() -> None:
    model = _model().fit(TEXTS, Y)
    alone = model._outputs(["pin trâu"])["logits"][0]
    batched = model._outputs(["pin trâu", " ".join(["máy"] * 50)])["logits"][0]
    assert torch.allclose(alone, batched, atol=1e-5)


def test_load_pretrained_embedding(tmp_path) -> None:
    vocab = ["<pad>", "<unk>", "pin", "trâu", "yếu"]
    matrix = np.arange(40, dtype=np.float32).reshape(5, 8)
    np.save(tmp_path / "emb.npy", matrix)
    (tmp_path / "emb.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    model = BiLstmAbsa(
        embedding_path=str(tmp_path / "emb.npy"),
        vocab_path=str(tmp_path / "emb.json"),
        hidden=8,
        epochs=1,
        batch_size=4,
        freeze_embedding=True,
    ).fit(TEXTS, Y, TEXTS[:3], Y[:3])
    weight = model.module.embedding.weight.detach().cpu()
    assert torch.equal(weight[2], torch.from_numpy(matrix[2]))
    assert model.info["vocab_size"] == 5
    # dev tokens: pin trâu pin yếu máy đẹp -> "máy", "đẹp" are not in the 5-word vocab
    assert abs(model.info["oov_rate_dev"] - 2 / 6) < 1e-9


def test_load_fasttext_keeps_vocab_words(tmp_path) -> None:
    vec = tmp_path / "tiny.vec.gz"
    with gzip.open(vec, "wt", encoding="utf-8") as f:
        f.write("3 2\npin 1 2\nxa 9 9\ntrâu 3 4\n")
    matrix, found = load_fasttext(vec, ["<pad>", "<unk>", "pin", "trâu", "yếu"])
    assert matrix.shape == (5, 2) and matrix.dtype == np.float32
    assert found.tolist() == [False, False, True, True, False]
    assert matrix[PAD].tolist() == [0, 0]
    assert matrix[2].tolist() == [1, 2] and matrix[3].tolist() == [3, 4]
