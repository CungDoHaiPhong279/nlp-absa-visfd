from importlib import import_module
from typing import Protocol

import numpy as np


class Model(Protocol):
    def fit(
        self,
        texts: list[str],
        Y: np.ndarray,
        dev_texts: list[str] | None = None,
        Y_dev: np.ndarray | None = None,
    ) -> "Model": ...

    def predict(self, texts: list[str]) -> np.ndarray: ...


REGISTRY: dict[str, str | type] = {
    "tfidf": "absa.baseline:TfidfAbsa",
    "transformer": "absa.transformer:TransformerAbsa",
    "bilstm": "absa.bilstm:BiLstmAbsa",
}


def build_model(model_cfg: dict, seed: int) -> Model:
    model_type = model_cfg.get("type")
    if model_type not in REGISTRY:
        choices = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Loại mô hình không hợp lệ: {model_type}. Hợp lệ: {choices}")

    registered = REGISTRY[model_type]
    if isinstance(registered, str):
        module_name, class_name = registered.split(":", maxsplit=1)
        model_class = getattr(import_module(module_name), class_name)
    else:
        model_class = registered

    kwargs = {key: value for key, value in model_cfg.items() if key != "type"}
    return model_class(**kwargs, seed=seed)
