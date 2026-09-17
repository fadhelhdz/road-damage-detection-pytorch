from dataclasses import MISSING, dataclass, fields, is_dataclass
from typing import Any, get_type_hints

import yaml

MODEL_NAMES = ("retinanet", "fcos", "fasterrcnn")
OPTIMIZERS = ("sgd", "adam", "adamw")

class ConfigError(ValueError):
    """
    Raised on unknown keys, missing fields, or invalid values in a config.
    """

@dataclass
class DataConfig:
    dataset_root: str   # required
    image_size: int = 800
    batch_size: int = 2
    num_workers: int = 4

@dataclass
class ModelConfig:
    num_classes: int        # required
    name: str = "retinanet" # one of MODEL_NAMES
    pretrained: bool = True

@dataclass
class TrainConfig:
    epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "sgd"  # one of OPTIMIZERS
    seed: int = 42
    amp: bool = True
    warmup_iters: int = 500   # linear LR warmup over the first N iterations
    grad_clip: float = 0.0    # max grad L2 norm; <= 0 disables clipping

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

def load_config(path: str) -> Config:
    """
    Load a YAML file into a nested Config.

    Args:
        path: Path to the YAML config file.
    Returns:
        A populated Config. Unknown keys and missing required fields raise
        ConfigError instead of being silently ignored.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path!r}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {path!r}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"Config file {path!r} is empty.")

    cfg = _build(Config, raw, path="")
    _validate(cfg)
    return cfg

def _build(cls: type, data: Any, path: str) -> Any:
    """
    Recursively build dataclass `cls` from mapping `data`.
    """
    where = path or "<root>"
    if not isinstance(data, dict):
        raise ConfigError(f"Section {where!r} must be a mapping, got {type(data).__name__}.")

    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}

    # Reject typos before constructing anything
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"Unknown key(s) {unknown} in {where!r}. Allowed: {sorted(known)}.")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        child = f"{path}.{f.name}" if path else f.name

        # Missing key: fall back to default, or error if the field is required
        if f.name not in data:
            if f.default is MISSING and f.default_factory is MISSING:
                raise ConfigError(f"Missing required field {child!r}.")
            continue

        value = data[f.name]
        ftype = hints[f.name]
        kwargs[f.name] = _build(ftype, value, child) if is_dataclass(ftype) else _coerce(value, ftype, child)

    return cls(**kwargs)

def _coerce(value: Any, ftype: type, path: str) -> Any:
    """
    Coerce a scalar YAML value to its annotated type, or raise.

    Fixes the PyYAML quirk where `1e-3` parses as a string, and rejects
    bool<->int mixups (e.g. amp: 1) so bad types never reach training.
    """

    if ftype is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ConfigError(f"Field {path!r} must be a bool, got {value!r}.")

    if ftype is int:
        if isinstance(value, bool):
            raise ConfigError(f"Field {path!r} must be an int, got bool {value!r}.")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                pass
        raise ConfigError(f"Field {path!r} must be an int, got {value!r}.")

    if ftype is float:
        if isinstance(value, bool):
            raise ConfigError(f"Field {path!r} must be an float, got bool {value!r}.")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                pass
        raise ConfigError(f"Field {path!r} must be a float, got {value!r}.")

    if ftype is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"Field {path!r} must be a str, got {type(value).__name__}.")

    return value

def _validate(cfg: Config) -> None:
    """
    Enum and range checks after the structure is built.
    """

    if cfg.model.name not in MODEL_NAMES:
        raise ConfigError(f"model.name must be one of {list(MODEL_NAMES)}, got {cfg.model.name!r}.")
    if cfg.train.optimizer not in OPTIMIZERS:
        raise ConfigError(f"train.optimizer must be one of {list(OPTIMIZERS)}, got {cfg.train.optimizer!r}.")
    if cfg.model.num_classes < 1:
        raise ConfigError(f"model.num_classes must be >= 1, got {cfg.model.num_classes}.")
    if cfg.data.batch_size < 1:
        raise ConfigError(f"data.batch_size must be >= 1, got {cfg.data.batch_size}.")
    if cfg.data.num_workers < 0:
        raise ConfigError(f"data.num_workers must be >= 0, got {cfg.data.num_workers}.")
    if cfg.train.epochs < 1:
        raise ConfigError(f"train.epochs must be >= 1, got {cfg.train.epochs}.")
    if cfg.train.lr <= 0:
        raise ConfigError(f"train.lr must be > 0, got {cfg.train.lr}.")
    if cfg.train.warmup_iters < 0:
        raise ConfigError(f"train.warmup_iters must be >= 0, got {cfg.train.warmup_iters}.")