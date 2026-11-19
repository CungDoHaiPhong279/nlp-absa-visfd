"""Bi-LSTM + fastText baseline: an approximate reproduction of the paper's model (D8)."""

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from absa.data import ASPECTS
from absa.transformer import absa_loss, decode

PAD, UNK = 0, 1
N_ASPECTS = len(ASPECTS)


def build_vocab(texts: list[str], min_freq: int = 1) -> list[str]:
    counts = Counter(tok for text in texts for tok in text.split())
    words = [w for w, c in counts.items() if c >= min_freq and w not in ("<pad>", "<unk>")]
    return ["<pad>", "<unk>", *sorted(words, key=lambda w: (-counts[w], w))]


def encode(texts: list[str], vocab: list[str], max_len: int) -> list[list[int]]:
    index = {w: i for i, w in enumerate(vocab)}
    return [[index.get(tok, UNK) for tok in text.split()[:max_len]] or [UNK] for text in texts]


def load_fasttext(
    vec_path: Path, vocab: list[str], seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Embedding matrix for `vocab` from a fastText `.vec.gz` file, plus a found-mask.

    PAD row is zero; words without a vector get normal(0, 0.1) (seeded).
    """
    index = {w: i for i, w in enumerate(vocab)}
    with gzip.open(vec_path, "rt", encoding="utf-8", errors="replace") as f:
        dim = int(f.readline().split()[1])
        matrix = np.random.default_rng(seed).normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
        found = np.zeros(len(vocab), dtype=bool)
        for line in f:
            parts = line.rstrip().split(" ")
            i = index.get(parts[0])
            if i is not None and i > UNK and len(parts) == dim + 1:
                matrix[i] = np.asarray(parts[1:], dtype=np.float32)
                found[i] = True
    matrix[PAD] = 0.0
    return matrix, found


class _BiLstmNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
        weights: np.ndarray | None,
        freeze: bool,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        if weights is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(torch.as_tensor(weights))
        self.embedding.weight.requires_grad = not freeze
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden,
            num_layers=layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(2 * hidden, N_ASPECTS * 4)

    def forward(self, ids: Tensor, lengths: Tensor) -> dict[str, Tensor]:
        x = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        real = (
            torch.arange(ids.size(1), device=ids.device)[None, :] < lengths.to(ids.device)[:, None]
        )
        pooled = out.masked_fill(~real[..., None], float("-inf")).max(dim=1).values
        return {"logits": self.classifier(self.dropout(pooled)).view(-1, N_ASPECTS, 4)}


def _pad(seqs: list[list[int]]) -> tuple[Tensor, Tensor]:
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    ids = torch.full((len(seqs), int(lengths.max())), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return ids, lengths


class BiLstmAbsa:
    """Joint 4-class output per aspect, trained with the shared `fit_loop`."""

    def __init__(
        self,
        embedding_path: str | None = None,
        vocab_path: str | None = None,
        emb_dim: int = 300,
        hidden: int = 256,
        layers: int = 1,
        dropout: float = 0.3,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 32,
        max_len: int = 128,
        freeze_embedding: bool = False,
        patience: int = 3,
        seed: int = 42,
    ) -> None:
        if (embedding_path is None) != (vocab_path is None):
            raise ValueError("Cần cả embedding_path và vocab_path, hoặc không có cả hai")
        self.embedding_path = embedding_path
        self.vocab_path = vocab_path
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_len = max_len
        self.freeze_embedding = freeze_embedding
        self.patience = patience
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocab: list[str] = []
        self.module: _BiLstmNet | None = None
        self.info: dict = {}

    def _outputs(self, texts: list[str]) -> dict[str, Tensor]:
        self.module.eval()
        seqs = encode(texts, self.vocab, self.max_len)
        parts = []
        with torch.no_grad():
            for i in range(0, len(seqs), self.batch_size * 2):
                ids, lengths = _pad(seqs[i : i + self.batch_size * 2])
                parts.append(self.module(ids.to(self.device), lengths)["logits"].cpu())
        return {"logits": torch.cat(parts)}

    def fit(
        self,
        texts: list[str],
        Y: np.ndarray,
        dev_texts: list[str] | None = None,
        Y_dev: np.ndarray | None = None,
    ) -> "BiLstmAbsa":
        from absa.training import fit_loop, set_seed

        set_seed(self.seed)
        weights = None
        if self.embedding_path is not None:
            weights = np.load(self.embedding_path)
            self.vocab = json.loads(Path(self.vocab_path).read_text(encoding="utf-8"))
            if len(self.vocab) != weights.shape[0]:
                raise ValueError("Số từ trong vocab_path khác số hàng của embedding")
            self.emb_dim = int(weights.shape[1])
        else:
            self.vocab = build_vocab(texts)

        self.module = _BiLstmNet(
            len(self.vocab),
            self.emb_dim,
            self.hidden,
            self.layers,
            self.dropout,
            weights,
            self.freeze_embedding,
        )
        seqs = encode(texts, self.vocab, self.max_len)
        labels = torch.as_tensor(np.asarray(Y), dtype=torch.long)
        rng = np.random.default_rng(self.seed)

        def train_batches():
            order = rng.permutation(len(seqs))
            for i in range(0, len(seqs), self.batch_size):
                idx = order[i : i + self.batch_size]
                ids, lengths = _pad([seqs[j] for j in idx])
                yield {"ids": ids, "lengths": lengths, "labels": labels[idx]}

        def loss_fn(module: nn.Module, batch: dict) -> Tensor:
            out = module(batch["ids"].to(self.device), batch["lengths"])
            return absa_loss(out, batch["labels"].to(self.device), "joint")

        has_dev = dev_texts is not None and Y_dev is not None and len(dev_texts) > 0
        result = fit_loop(
            self.module,
            train_batches,
            (lambda: self.predict(dev_texts)) if has_dev else None,
            Y_dev if has_dev else None,
            loss_fn,
            epochs=self.epochs,
            lr=self.lr,
            patience=self.patience,
            fp16=False,
            device=self.device,
        )

        oov = None
        if has_dev:
            known = set(self.vocab[2:])
            tokens = [tok for text in dev_texts for tok in text.split()]
            oov = sum(tok not in known for tok in tokens) / max(len(tokens), 1)
        self.info = {
            "best_epoch": result["best_epoch"],
            "best_dev_sentiment_macro_f1": result["best_dev_sentiment_macro_f1"],
            "vocab_size": len(self.vocab),
            "oov_rate_dev": oov,
        }
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        if self.module is None:
            raise ValueError("Mô hình chưa được huấn luyện")
        if not texts:
            return np.zeros((0, N_ASPECTS), dtype=np.int8)
        return decode(self._outputs(texts), "joint")
