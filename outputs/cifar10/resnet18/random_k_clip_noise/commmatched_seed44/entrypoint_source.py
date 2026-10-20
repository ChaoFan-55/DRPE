"""Random-k + global clipping + Gaussian noise; reuse the existing FL runner.

Run from the drpe_formal_experiments project root. No existing source is edited.
For tensor t with sampling probability p_t=k_t/n_t, the sparse values are
    (S_t clip_C(update)_t + epsilon_t) / p_t
when unbiased=True (the existing Random-k baseline setting).
epsilon has std eta*C/sqrt(K), K=sum_t k_t, BEFORE unbiased scaling.
This matches DRPE's eta convention, not its per-coordinate sigma or a DP budget.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def make_private_compressor(original, settings, diagnostics_path=None):
    import torch
    from random_k import stable_seed
    from src.state import clip_tensor_dict_by_global_l2
    from src.utils import append_csv

    def compress(result, ratio, base_seed, round_index, unbiased, sampling_device):
        sizes = {name: tensor.numel() for name, tensor in result.payload.items()}
        result.payload, raw_norm, clip_scale = clip_tensor_dict_by_global_l2(
            result.payload, float(settings["clip_norm"])
        )
        result.update_norm = raw_norm
        result.clip_scale = clip_scale
        sparse = original(
            result=result, ratio=ratio, base_seed=base_seed,
            round_index=round_index, unbiased=False,
            sampling_device=sampling_device,
        )
        if sparse.selected_numel != settings["encoded_parameter_count"]:
            raise RuntimeError("Selected dimension differs from model-derived K.")
        sigma = float(settings["noise_std"])
        for name, item in sparse.payload.items():
            if sigma > 0:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(stable_seed(
                    "random_k_clip_noise", settings["noise_seed"],
                    round_index, result.client_id, name,
                ))
                noise = torch.randn(
                    item.values.shape, generator=generator,
                    device="cpu", dtype=item.values.dtype,
                )
                item.values.add_(noise, alpha=sigma)
            if unbiased:
                item.values.mul_(sizes[name] / item.values.numel())
        if diagnostics_path is not None:
            append_csv({
                "round": round_index, "client_id": result.client_id,
                "raw_update_norm": raw_norm, "clip_scale": clip_scale,
                "clipped_update_norm": raw_norm * clip_scale,
                "selected_numel": sparse.selected_numel,
                "noise_std_before_unbiased_scaling": sigma,
                "unbiased": unbiased,
            }, diagnostics_path)
        return sparse
    return compress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    root = Path.cwd().resolve()
    baseline_dir = root / "random_k_baseline"
    if not (baseline_dir / "run_random_k.py").is_file():
        raise SystemExit("Run this command from the drpe_formal_experiments root.")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(baseline_dir))
    import run_random_k as runner

    settings = runner.load_settings(args.config, args.set)
    settings.setdefault("clip_norm", 2.950101638259676)
    settings.setdefault("noise_eta", 2.0)
    settings.setdefault("noise_seed", 314159 + int(settings["seed"]) - 42)
    settings["privacy_enabled"] = True
    settings["experiment_variant"] = "random_k_clip_noise"
    settings["noise_placement"] = "after_sampling_before_unbiased_scaling"
    settings["noise_std_rule"] = "noise_eta * clip_norm / sqrt(encoded_parameter_count)"
    if not math.isfinite(float(settings["clip_norm"])) or float(settings["clip_norm"]) <= 0:
        raise ValueError("clip_norm must be finite and positive.")
    if not math.isfinite(float(settings["noise_eta"])) or float(settings["noise_eta"]) < 0:
        raise ValueError("noise_eta must be finite and nonnegative.")
    if settings.get("noise_std") is not None:
        raise ValueError("Use noise_std=null; this entry computes sigma from actual K.")
    output = Path(settings["output_dir"]).resolve()
    # The original runner truncates metrics.csv, so reject reused output folders.
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output is nonempty; choose a NEW output_dir: {output}")

    original_build = runner.build_model
    original_save = runner.save_json
    original_compress = runner.compress_client_result

    def build_model(*build_args, **build_kwargs):
        model = original_build(*build_args, **build_kwargs)
        ratio = float(settings["compression_ratio"])
        count = sum(min(p.numel(), max(1, int(round(p.numel() * ratio))))
                    for p in model.parameters())
        settings["encoded_parameter_count"] = count
        settings["noise_std"] = float(settings["noise_eta"]) * float(settings["clip_norm"]) / math.sqrt(count)
        print(f"[Random-k + clip + noise] K={count}, C={settings['clip_norm']}, "
              f"eta={settings['noise_eta']}, sigma_pre_scale={settings['noise_std']:.12g}, "
              f"unbiased={settings.get('unbiased', True)}", flush=True)
        return model

    def save_json(data, path):
        data = dict(data)
        if Path(path).name in {"method_metadata.json", "summary.json"}:
            data.update({
                "method": "random_k_clip_noise", "clipping": True,
                "privacy_noise": float(settings["noise_std"]) > 0,
                **{key: settings[key] for key in (
                    "clip_norm", "noise_eta", "noise_std", "noise_seed",
                    "encoded_parameter_count", "noise_placement", "noise_std_rule",
                )},
            })
        original_save(data, path)

    runner.build_model = build_model
    runner.save_json = save_json
    # Raw FedAvg client -> clip -> sample -> noise -> existing unbiased correction.
    runner.compress_client_result = make_private_compressor(
        original_compress, settings, output / "privacy_diagnostics.csv"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "entrypoint_source.py").write_text(
        Path(__file__).read_text(encoding="utf-8-sig"), encoding="utf-8"
    )
    runner.run(settings)


if __name__ == "__main__":
    main()
