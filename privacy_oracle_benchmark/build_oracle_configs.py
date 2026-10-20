from __future__ import print_function

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = (
    ROOT
    / "privacy_main_benchmark"
    / "results"
    / "generated_configs"
)

OUTPUT_ROOT = (
    ROOT
    / "privacy_oracle_benchmark"
    / "results"
    / "generated_configs"
)

RAW_ROOT = (
    ROOT
    / "privacy_oracle_benchmark"
    / "results"
    / "raw"
)

SEEDS = [42, 43, 44]
TARGETS = list(range(10))
RESTARTS = [0, 1]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def source_config(method, seed, target, restart):
    filename = (
        "{method}__m{seed}__T{target:02d}__dlg__r{restart:02d}.json"
    ).format(
        method=method,
        seed=seed,
        target=target,
        restart=restart,
    )

    return SOURCE_ROOT / method / filename


def build_one(
    source_method,
    variant,
    seed,
    target,
    restart,
    observation_mode,
):
    source_path = source_config(
        source_method,
        seed,
        target,
        restart,
    )

    if not source_path.exists():
        raise FileNotFoundError(
            "Missing source config: {}".format(source_path)
        )

    cfg = load_json(source_path)

    # Oracle-label:
    # attacker is directly given the true class label.
    cfg["attack_name"] = "dlg"
    cfg["label_mode"] = "known"

    # Keep all other parameters exactly inherited from
    # the original privacy-main benchmark.
    cfg["observation_mode"] = observation_mode

    output_dir = (
        RAW_ROOT
        / variant
        / "seed{}".format(seed)
        / "T{:02d}".format(target)
        / "restart{:02d}".format(restart)
        / "oracle"
    )

    cfg["output_dir"] = output_dir.as_posix()

    output_filename = (
        "{variant}__m{seed}__T{target:02d}"
        "__oracle__r{restart:02d}.json"
    ).format(
        variant=variant,
        seed=seed,
        target=target,
        restart=restart,
    )

    output_path = (
        OUTPUT_ROOT
        / variant
        / output_filename
    )

    save_json(output_path, cfg)

    return output_path


def main():
    generated = []

    for seed in SEEDS:
        for target in TARGETS:
            for restart in RESTARTS:

                # --------------------------------------------------
                # 1. FedAvg raw + known label
                # --------------------------------------------------
                generated.append(
                    build_one(
                        source_method="fedavg",
                        variant="fedavg_raw",
                        seed=seed,
                        target=target,
                        restart=restart,
                        observation_mode="raw",
                    )
                )

                # --------------------------------------------------
                # 2. DRPE encoded upload + known label
                # Main Oracle experiment
                # --------------------------------------------------
                generated.append(
                    build_one(
                        source_method="drpe",
                        variant="drpe_encoded",
                        seed=seed,
                        target=target,
                        restart=restart,
                        observation_mode="encoded",
                    )
                )

                # --------------------------------------------------
                # 3. DRPE decoded representation + known label
                # Observation-space control
                # --------------------------------------------------
                generated.append(
                    build_one(
                        source_method="drpe",
                        variant="drpe_decoded",
                        seed=seed,
                        target=target,
                        restart=restart,
                        observation_mode="decoded",
                    )
                )

    print("=" * 80)
    print("Oracle-label configurations generated successfully.")
    print("Generated configs: {}".format(len(generated)))
    print("Expected: 180")
    print("Output root:")
    print(OUTPUT_ROOT)
    print("=" * 80)


if __name__ == "__main__":
    main()