import json
from pathlib import Path
from src.config import ExperimentConfig, resolve_drpe_privacy_parameters
from src.models import build_model


def test_eta_sigma_resolution(tmp_path: Path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"selected_clip_norm": 3.0}), encoding="utf-8")
    cfg = ExperimentConfig(calibration_path=str(calibration), clip_norm=None, noise_std=None)
    model = build_model("resnet18", 10, "group", 32)
    result = resolve_drpe_privacy_parameters(cfg, model)
    assert cfg.clip_norm == 3.0
    assert cfg.noise_std > 0
    assert result["encoded_parameter_count"] > 0
