"""Smoke tests for Phase 0: imports, config loading, and U-Net forward pass."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent


class TestImports:
    """Verify all package modules can be imported."""

    def test_import_geohazard(self) -> None:
        import geohazard

        assert hasattr(geohazard, "__version__")

    def test_import_config(self) -> None:
        from geohazard.config import HazardConfig, load_config  # noqa: F401

    def test_import_seed(self) -> None:
        from geohazard.utils.seed import set_seed  # noqa: F401

    def test_import_logging(self) -> None:
        from geohazard.utils.logging import get_logger  # noqa: F401

    def test_import_subpackages(self) -> None:
        import geohazard.dashboard  # noqa: F401
        import geohazard.data  # noqa: F401
        import geohazard.features  # noqa: F401
        import geohazard.gis  # noqa: F401
        import geohazard.inference  # noqa: F401
        import geohazard.models  # noqa: F401
        import geohazard.training  # noqa: F401
        import geohazard.utils  # noqa: F401


class TestConfig:
    """Verify config loader parses YAML stubs correctly."""

    @pytest.fixture(params=["flood.yaml", "landslide.yaml", "wildfire.yaml", "damage_xbd.yaml"])
    def config_path(self, request: pytest.FixtureRequest) -> Path:
        return ROOT / "configs" / request.param

    def test_load_config(self, config_path: Path) -> None:
        from geohazard.config import load_config

        cfg = load_config(config_path)
        assert cfg.hazard in {"flood", "landslide", "wildfire", "damage"}
        assert cfg.model.num_classes >= 2
        assert cfg.training.seed == 42
        assert cfg.dataset.tile_size > 0

    def test_config_not_found(self, tmp_path: Path) -> None:
        from geohazard.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


class TestSeed:
    """Verify seed fixing produces deterministic tensors."""

    def test_deterministic(self) -> None:
        from geohazard.utils.seed import set_seed

        set_seed(42)
        a = torch.randn(3, 3)
        set_seed(42)
        b = torch.randn(3, 3)
        assert torch.equal(a, b)


class TestUNetSmoke:
    """Verify a tiny U-Net forward pass works on a random tensor."""

    def test_unet_forward(self) -> None:
        import segmentation_models_pytorch as smp

        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=2,
        )
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 256, 256)

    def test_unet_multiband(self) -> None:
        """Test U-Net with more input channels (simulates S2+S1+DEM fusion)."""
        import segmentation_models_pytorch as smp

        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=10,
            classes=2,
        )
        x = torch.randn(1, 10, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 256, 256)
