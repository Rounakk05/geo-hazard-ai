"""Config loader: YAML -> Pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    name: str
    root: str
    splits_file: str
    tile_size: int = 256
    num_classes: int = 2
    bands: dict[str, list[str]] = Field(default_factory=dict)
    class_names: list[str] = Field(default_factory=list)
    class_weights: list[float] = Field(default_factory=list)
    label_mapping: dict[int, int] | None = None


class ModelConfig(BaseModel):
    architecture: str = "unet"
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    in_channels: int = 3
    num_classes: int = 2


class TrainingConfig(BaseModel):
    seed: int = 42
    epochs: int = 60
    batch_size: int = 16
    num_workers: int = 4
    optimizer: str = "adamw"
    lr_encoder: float = 1e-4
    lr_decoder: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    loss: str = "bce_dice"
    focal_gamma: float = 2.0
    amp: bool = True
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(default_factory=lambda: ["iou", "f1"])
    export_formats: list[str] = Field(default_factory=lambda: ["geotiff", "geojson"])


class DebugConfig(BaseModel):
    enabled: bool = False
    max_samples: int = 50
    epochs: int = 2


class HazardConfig(BaseModel):
    hazard: str
    description: str = ""
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)


def load_config(path: str | Path) -> HazardConfig:
    """Load a hazard config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return HazardConfig(**raw)
