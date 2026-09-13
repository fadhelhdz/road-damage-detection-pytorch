from pathlib import Path

import pytest

from utils.config import (
    Config,
    ConfigError,
    DataConfig,
    ModelConfig,
    TrainConfig,
    load_config
)

BASELINE = Path(__file__).resolve().parents[1] / "configs" / "baseline_retinanet.yaml"

def test_valid_config_loads_nested_types():
    cfg = load_config(str(BASELINE))

    assert isinstance(cfg, Config)
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.train, TrainConfig)

    # Values parsed with the right types
    assert cfg.model.name == "retinanet"
    assert cfg.model.num_classes == 4
    assert isinstance(cfg.data.batch_size, int)
    assert isinstance(cfg.train.lr, float)
    assert isinstance(cfg.train.amp, bool)

def test_unknown_key_raises(tmp_path):
    bad = tmp_path / "unknown_key.yaml"
    bad.write_text(
        "data: {dataset_root: ./d}\n"
        "model: {num_classes: 80}\n"
        "train: {learnig_rate: 0.01}\n" # typo, not a real field
    )
    with pytest.raises(ConfigError):
        load_config(str(bad))

def test_missing_required_key_raises(tmp_path):
    bad = tmp_path / "missing_required.yaml"
    bad.write_text(
        "data: {image_size: 800}\n" # dataset_root missing
        "model: {num_classes: 80}\n" 
        "train: {}\n"
    )
    with pytest.raises(ConfigError):
        load_config(str(bad))