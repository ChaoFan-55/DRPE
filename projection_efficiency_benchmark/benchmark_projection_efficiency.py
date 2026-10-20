from __future__ import print_function

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import get_dataset_spec
from src.models import build_model
from src.projection import fwht, iter_block_layouts


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_model_compat(cfg):
    spec = get_dataset_spec(cfg.dataset)
    try:
        return build_model(cfg.model_name, cfg.num_classes, cfg.norm, spec.image_size)
    except TypeError:
        return build_model(cfg.model_name, cfg.num_classes, cfg.norm)


def next_power_of_two(v):
    return 1 if v == 1 else 1 << (v - 1).bit_length()


def projected_dim(valid_dim, ratio):
    return min(
        next_power_of_two(valid_dim),
        max(1, int(round(valid_dim * ratio))),
    )


def make_generator(device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def dense_state(n, m, seed, device):
    # Row-orthonormal dense Gaussian projection.
    # QR of an n x m Gaussian matrix gives Q with orthonormal columns;
    # P = Q^T therefore has m orthonormal rows.
    gen = make_generator(device, seed)
    gaussian = torch.randn(
        (n, m),
        generator=gen,
        device=device,
        dtype=torch.float32,
    )
    q, _ = torch.linalg.qr(gaussian, mode="reduced")
    return q.transpose(0, 1).contiguous()


def drpe_state(n, m, seed, device):
    gen = make_generator(device, seed)
    signs = torch.randint(
        0, 2, (n,),
        generator=gen,
        device=device,
        dtype=torch.int64,
    ).to(torch.float32)
    signs = signs.mul_(2.0).sub_(1.0)
    indices = torch.randperm(n, generator=gen, device=device)[:m]
    return signs, indices


def dense_encode(x, state):
    return torch.mv(state, x)


def dense_decode(y, state):
    return torch.mv(state.transpose(0, 1), y)


def drpe_encode(x, state):
    signs, indices = state
    return fwht(x * signs)[indices]


def drpe_decode(y, state, n):
    signs, indices = state
    sparse = torch.zeros(n, device=y.device, dtype=y.dtype)
    sparse[indices] = y
    return fwht(sparse) * signs


def state_bytes(method, state):
    if method == "dense_rp":
        return state.numel() * state.element_size()
    signs, indices = state
    return (
        signs.numel() * signs.element_size()
        + indices.numel() * indices.element_size()
    )


def mean_std(values):
    vals = [float(x) for x in values]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def timed_call(fn, repeats, device):
    times = []
    for _ in range(int(repeats)):
        sync(device)
        t0 = time.perf_counter()
        out = fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
        del out
    return times


def measure_peak_extra_cuda(fn, repeats, device):
    if device.type != "cuda":
        return None
    peaks = []
    for _ in range(int(repeats)):
        torch.cuda.empty_cache()
        sync(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        out = fn()
        sync(device)
        peak = torch.cuda.max_memory_allocated(device)
        peaks.append(max(0, int(peak - baseline)))
        del out
    return float(statistics.mean(peaks))


def benchmark_one(method, seed, n, m, warmup, repeats, memory_repeats, device):
    x_gen = make_generator(device, seed + 100000)
    x = torch.randn(n, generator=x_gen, device=device, dtype=torch.float32)

    if method == "dense_rp":
        make_state = lambda s: dense_state(n, m, s, device)
        encode = dense_encode
        decode = lambda y, state: dense_decode(y, state)
    else:
        make_state = lambda s: drpe_state(n, m, s, device)
        encode = drpe_encode
        decode = lambda y, state: drpe_decode(y, state, n)

    # Warm-up includes projection construction because dynamic projection changes
    # between rounds in the formal setting.
    for i in range(int(warmup)):
        st = make_state(seed + i + 1)
        y = encode(x, st)
        z = decode(y, st)
        del st, y, z
    sync(device)

    # Cached state: isolates the projection kernel when a round's state already exists.
    state = make_state(seed)
    y_cached = encode(x, state)
    sync(device)

    cached_encode_ms = timed_call(
        lambda: encode(x, state),
        repeats,
        device,
    )
    cached_decode_ms = timed_call(
        lambda: decode(y_cached, state),
        repeats,
        device,
    )
    cached_roundtrip_ms = timed_call(
        lambda: decode(encode(x, state), state),
        repeats,
        device,
    )

    cached_peak_bytes = measure_peak_extra_cuda(
        lambda: decode(encode(x, state), state),
        memory_repeats,
        device,
    )

    persistent_state_bytes = state_bytes(method, state)

    # Dynamic construction + encoding/decoding.
    counter = [0]

    def dynamic_setup():
        counter[0] += 1
        return make_state(seed + 1000000 + counter[0])

    setup_ms = timed_call(dynamic_setup, repeats, device)

    counter2 = [0]

    def dynamic_encode():
        counter2[0] += 1
        st = make_state(seed + 2000000 + counter2[0])
        return encode(x, st)

    dynamic_encode_ms = timed_call(dynamic_encode, repeats, device)

    counter3 = [0]

    def dynamic_roundtrip():
        counter3[0] += 1
        st = make_state(seed + 3000000 + counter3[0])
        y = encode(x, st)
        return decode(y, st)

    dynamic_roundtrip_ms = timed_call(dynamic_roundtrip, repeats, device)

    dynamic_peak_bytes = measure_peak_extra_cuda(
        lambda: (
            (lambda st: decode(encode(x, st), st))
            (make_state(seed + 9000000))
        ),
        memory_repeats,
        device,
    )

    e_mean, e_std = mean_std(cached_encode_ms)
    d_mean, d_std = mean_std(cached_decode_ms)
    rt_mean, rt_std = mean_std(cached_roundtrip_ms)
    s_mean, s_std = mean_std(setup_ms)
    de_mean, de_std = mean_std(dynamic_encode_ms)
    drt_mean, drt_std = mean_std(dynamic_roundtrip_ms)

    row = {
        "method": method,
        "projection_seed": int(seed),
        "block_size": int(n),
        "projected_dim": int(m),
        "actual_block_ratio": float(m) / float(n),
        "setup_ms_mean": s_mean,
        "setup_ms_std": s_std,
        "cached_encode_ms_mean": e_mean,
        "cached_encode_ms_std": e_std,
        "cached_decode_ms_mean": d_mean,
        "cached_decode_ms_std": d_std,
        "cached_roundtrip_ms_mean": rt_mean,
        "cached_roundtrip_ms_std": rt_std,
        "dynamic_encode_ms_mean": de_mean,
        "dynamic_encode_ms_std": de_std,
        "dynamic_roundtrip_ms_mean": drt_mean,
        "dynamic_roundtrip_ms_std": drt_std,
        "cached_state_bytes_per_full_block": int(persistent_state_bytes),
        "cached_state_kib_per_full_block": float(persistent_state_bytes) / 1024.0,
        "cached_roundtrip_peak_extra_cuda_bytes": (
            None if cached_peak_bytes is None else int(round(cached_peak_bytes))
        ),
        "cached_roundtrip_peak_extra_cuda_kib": (
            None if cached_peak_bytes is None else cached_peak_bytes / 1024.0
        ),
        "dynamic_roundtrip_peak_extra_cuda_bytes": (
            None if dynamic_peak_bytes is None else int(round(dynamic_peak_bytes))
        ),
        "dynamic_roundtrip_peak_extra_cuda_kib": (
            None if dynamic_peak_bytes is None else dynamic_peak_bytes / 1024.0
        ),
    }

    del y_cached, state, x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def model_storage_stats(model, block_size, ratio):
    total_params = 0
    encoded_params = 0
    block_count = 0
    full_block_count = 0

    # Dense-RP: if every block's row-orthonormal Gaussian matrix were explicitly
    # materialized/cached, this is the exact FP32 matrix footprint for the model's
    # tensor-by-tensor blocking.
    dense_matrix_elements = 0

    # DRPE never requires an explicit dense P. For an optional cached
    # implementation, signs+indices are counted separately below.
    drpe_cached_sign_elements = 0
    drpe_cached_index_elements = 0

    for name, parameter in model.named_parameters():
        numel = int(parameter.numel())
        total_params += numel

        for layout in iter_block_layouts(numel, block_size, ratio):
            block_count += 1
            if layout.valid_dim == block_size:
                full_block_count += 1
            encoded_params += int(layout.projected_dim)

            # Dense baseline described in the project uses a dense
            # row-orthonormal Gaussian matrix of m x valid_dim per block.
            dense_matrix_elements += int(layout.projected_dim) * int(layout.valid_dim)

            # Optional cached structured state: one FP32 sign per transform
            # coordinate + one int64 sampled index per retained coordinate.
            # The actual DRPE implementation can regenerate these from seed,
            # so this is an upper bound for cached metadata, not a requirement.
            drpe_cached_sign_elements += int(layout.transform_dim)
            drpe_cached_index_elements += int(layout.projected_dim)

    dense_bytes_fp32 = dense_matrix_elements * 4
    drpe_optional_cached_bytes = drpe_cached_sign_elements * 4 + drpe_cached_index_elements * 8

    return {
        "total_trainable_parameters": int(total_params),
        "encoded_parameters": int(encoded_params),
        "actual_model_ratio": float(encoded_params) / float(total_params),
        "block_count": int(block_count),
        "full_512_block_count": int(full_block_count) if block_size == 512 else int(full_block_count),
        "dense_rp_explicit_matrix_elements_if_cached": int(dense_matrix_elements),
        "dense_rp_explicit_matrix_bytes_fp32_if_cached": int(dense_bytes_fp32),
        "dense_rp_explicit_matrix_mib_fp32_if_cached": float(dense_bytes_fp32) / (1024.0 ** 2),
        "dense_rp_explicit_matrix_gib_fp32_if_cached": float(dense_bytes_fp32) / (1024.0 ** 3),
        "drpe_explicit_dense_matrix_bytes_required": 0,
        "drpe_optional_cached_sign_index_bytes_upper_bound": int(drpe_optional_cached_bytes),
        "drpe_optional_cached_sign_index_mib_upper_bound": float(drpe_optional_cached_bytes) / (1024.0 ** 2),
        "note": (
            "DRPE is matrix-free: its actual project implementation regenerates signs/indices "
            "from seed/tensor/block/round metadata and does not require storing a dense projection matrix. "
            "The cached sign/index figure is only an optional upper bound if all block metadata were cached."
        ),
    }


def aggregate_seed_rows(seed_rows):
    methods = sorted(set(row["method"] for row in seed_rows))
    metric_names = [
        "setup_ms_mean",
        "cached_encode_ms_mean",
        "cached_decode_ms_mean",
        "cached_roundtrip_ms_mean",
        "dynamic_encode_ms_mean",
        "dynamic_roundtrip_ms_mean",
        "cached_state_kib_per_full_block",
        "cached_roundtrip_peak_extra_cuda_kib",
        "dynamic_roundtrip_peak_extra_cuda_kib",
    ]

    rows = []
    for method in methods:
        subset = [r for r in seed_rows if r["method"] == method]
        out = {
            "method": method,
            "num_projection_seeds": len(subset),
            "block_size": subset[0]["block_size"],
            "projected_dim": subset[0]["projected_dim"],
            "actual_block_ratio": subset[0]["actual_block_ratio"],
        }
        for metric in metric_names:
            values = [r[metric] for r in subset if r[metric] is not None]
            mean, std = mean_std(values)
            out[metric + "_across_seeds"] = mean
            out[metric + "_across_seeds_std"] = std
        rows.append(out)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    training_cfg_path = (PROJECT_ROOT / cfg["training_config"]).resolve()
    train_cfg = load_config(training_cfg_path)

    requested = str(cfg.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(requested)

    model = build_model_compat(train_cfg)
    storage = model_storage_stats(
        model,
        int(cfg["block_size"]),
        float(cfg["compression_ratio"]),
    )

    n = int(cfg["block_size"])
    m = max(1, int(round(n * float(cfg["compression_ratio"]))))
    m = min(m, n)

    print("=" * 100)
    print("Projection efficiency benchmark: DRPE vs Dense-RP")
    print("device=%s | block=%d | projected=%d | ratio=%.6f" % (
        str(device), n, m, float(m) / float(n)
    ))
    print("warmup=%d | timed_repeats=%d | seeds=%s" % (
        int(cfg["warmup"]),
        int(cfg["timed_repeats"]),
        str(cfg["projection_seeds"]),
    ))
    print("Model trainable params=%d | encoded=%d | model ratio=%.6f" % (
        storage["total_trainable_parameters"],
        storage["encoded_parameters"],
        storage["actual_model_ratio"],
    ))
    print("=" * 100)

    seed_rows = []
    for seed in cfg["projection_seeds"]:
        for method in ("dense_rp", "drpe"):
            print("\nRunning method=%s seed=%s ..." % (method, str(seed)), flush=True)
            row = benchmark_one(
                method=method,
                seed=int(seed),
                n=n,
                m=m,
                warmup=int(cfg["warmup"]),
                repeats=int(cfg["timed_repeats"]),
                memory_repeats=int(cfg["memory_repeats"]),
                device=device,
            )
            seed_rows.append(row)
            print(
                " setup=%.4f ms | cached enc=%.4f ms | cached dec=%.4f ms | "
                "dynamic roundtrip=%.4f ms | state=%.2f KiB"
                % (
                    row["setup_ms_mean"],
                    row["cached_encode_ms_mean"],
                    row["cached_decode_ms_mean"],
                    row["dynamic_roundtrip_ms_mean"],
                    row["cached_state_kib_per_full_block"],
                )
            )

    summary_rows = aggregate_seed_rows(seed_rows)

    output_dir = (PROJECT_ROOT / cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "benchmark_seed_results.csv", seed_rows)
    write_csv(output_dir / "benchmark_summary.csv", summary_rows)
    write_json(output_dir / "projection_storage_summary.json", storage)
    write_json(
        output_dir / "resolved_benchmark_config.json",
        {
            "benchmark_config": cfg,
            "training_config": train_cfg.to_dict(),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
            "torch_version": torch.__version__,
        },
    )

    print("\n" + "=" * 100)
    print("Storage comparison for full ResNet-18 parameter blocking")
    print(
        "Dense-RP explicit FP32 matrices if cached: %.2f MiB (%.3f GiB)"
        % (
            storage["dense_rp_explicit_matrix_mib_fp32_if_cached"],
            storage["dense_rp_explicit_matrix_gib_fp32_if_cached"],
        )
    )
    print("DRPE explicit dense matrix required: 0 bytes")
    print(
        "DRPE optional cached signs+indices upper bound: %.2f MiB"
        % storage["drpe_optional_cached_sign_index_mib_upper_bound"]
    )
    print("=" * 100)
    print("Saved:")
    print(output_dir / "benchmark_seed_results.csv")
    print(output_dir / "benchmark_summary.csv")
    print(output_dir / "projection_storage_summary.json")
    print(output_dir / "resolved_benchmark_config.json")


if __name__ == "__main__":
    main()
