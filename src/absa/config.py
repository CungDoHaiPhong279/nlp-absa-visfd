from copy import deepcopy
from pathlib import Path

import yaml


def _deep_merge(base: dict, child: dict) -> dict:
    """Gộp dict lồng nhau; giá trị trong child được ưu tiên."""
    result = deepcopy(base)

    for key, value in child.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def load_config(path: Path) -> dict:
    """Đọc YAML, kế thừa base nếu có, và không kế thừa name/groups."""
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(config, dict):
        raise ValueError(f"Config phải là mapping YAML: {path}")

    base_path = config.get("base")
    if base_path:
        base = load_config(path.parent / base_path)
        base.pop("name", None)
        base.pop("groups", None)

        config = _deep_merge(
            base,
            {key: value for key, value in config.items() if key != "base"},
        )

    if not config.get("name"):
        raise ValueError(f"Config thiếu khóa 'name': {path}")

    config.setdefault("groups", [])
    config.pop("base", None)
    return config


def _format_value(value) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Ghi đè khóa đã tồn tại, ví dụ model.C=0.5."""
    result = deepcopy(cfg)
    result.setdefault("groups", [])
    suffixes = []

    for override in overrides:
        key_path, separator, raw_value = override.partition("=")
        if not separator or not key_path:
            raise ValueError(f"Override phải theo dạng key=value: {override}")

        keys = key_path.split(".")
        target = result

        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise ValueError(f"Không tìm thấy khóa cấu hình: {key_path}")
            target = target[key]

        leaf = keys[-1]
        if leaf not in target:
            raise ValueError(f"Không tìm thấy khóa cấu hình: {key_path}")

        value = yaml.safe_load(raw_value)
        target[leaf] = value
        suffixes.append(f"{leaf}={_format_value(value)}")

    if suffixes:
        result["name"] += "".join(f"__{suffix}" for suffix in suffixes)
        result["groups"] = []

    return result


def assert_final_config(path: Path) -> None:
    path = Path(path).resolve()
    final_dir = (Path("configs") / "final").resolve()
    if final_dir not in path.parents:
        raise ValueError("Chỉ đánh giá Test bằng config trong configs/final/")
