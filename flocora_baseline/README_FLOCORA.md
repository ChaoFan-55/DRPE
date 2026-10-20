# FLoCoRA 独立基线（CIFAR-10 / ResNet18 / ~30% uplink）

本目录是给当前 DRPE 正式实验工程使用的**独立 FLoCoRA baseline**。它只依赖项目根目录已有的 `src/`（数据划分、ResNet18、工具函数），不会修改 FedAvg、DRPE、Random-k、Top-k、Dense-RP 或 PowerSGD 的任何文件。

## 1. 方法实现

依据 FLoCoRA（EUSIPCO 2024）的最终层配置：

- 原始随机初始化卷积权重始终冻结；
- 所有 `Conv2d` 使用 LoRA adapter；
- GroupNorm 参数正常训练；
- 最终 Linear/FC 参数正常训练；
- 每轮客户端只上传 trainable state（LoRA + GroupNorm + FC）；
- 服务端按客户端样本量做 FedAvg 加权平均；
- 主基线使用 FP32，不加量化、不裁剪、不加噪声。

卷积 adapter 使用论文记号：

`B: [r, I, K, K]`，`A: [O, r, 1, 1]`，缩放为 `alpha / r`。

当前正式配置使用 `alpha = 16r`，与论文对小 CNN / ResNet18 使用的增强缩放设置一致。

## 2. 为什么正式配置是 rank=93

当前工程的 CIFAR-10 ResNet18(GroupNorm) dense 参数量为 11,173,962。对所有卷积加 LoRA，同时训练 GroupNorm 和最终 FC 后：

- `rank=93`
- `alpha=1488=16*93`
- trainable / uplink 参数量 = 3,344,409
- uplink ratio = 0.299304x dense
- 两客户端每轮 FP32 uplink ≈ 25.52 MiB
- 对照 FedAvg 两客户端 raw uplink ≈ 85.25 MiB

因此它与当前约 30% uplink 的 Random-k / Top-k / Dense-RP / PowerSGD / DRPE 基本通信对齐。

注意：`rank=93` 是**当前论文统一 30% 通信预算下的通信匹配配置**，不是 FLoCoRA 原论文表格中固定使用的 rank。原论文的 ResNet18实验主要报告 r=16/32/64；本项目为了公平比较，使用 rank calibration 找到约 0.30x 的 r。

## 3. 先做校验

在项目根目录执行：

```powershell
cd .
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"

python ./flocora_baseline/validate_flocora.py
python ./flocora_baseline/calibrate_rank.py --target-ratio 0.30
```

正常应看到：初始 FLoCoRA 输出与原始 dense 初始化完全一致；冻结卷积在一次训练后不变化；rank=93 的通信比约 0.299304x。

## 4. Seed42 五轮 smoke

```powershell
python ./flocora_baseline/run_flocora.py `
  --config ./flocora_baseline/configs/cifar10_resnet18/flocora_commmatched_seed42_fixed200.json `
  --set rounds=5 output_dir=./outputs/cifar10/resnet18/flocora/smoke_seed42
```

重点确认日志开头：

```text
Method=FLoCoRA-FP32
LoRA rank=93, lora_alpha=1488
Trainable payload=3,344,409/11,173,962 (0.299304x dense)
Per-round uplink (2 clients)≈25.52 MiB
Quantization=None | clipping=None | noise=0.0
```

## 5. 正式运行顺序

先 Seed42 200轮 gate：

```powershell
python ./flocora_baseline/run_flocora.py `
  --config ./flocora_baseline/configs/cifar10_resnet18/flocora_commmatched_seed42_fixed200.json
```

Seed42 正常后再跑：

```powershell
python ./flocora_baseline/run_flocora.py `
  --config ./flocora_baseline/configs/cifar10_resnet18/flocora_commmatched_seed43_fixed200.json

python ./flocora_baseline/run_flocora.py `
  --config ./flocora_baseline/configs/cifar10_resnet18/flocora_commmatched_seed44_fixed200.json
```

也可以一次执行：

```powershell
powershell -ExecutionPolicy Bypass -File ./flocora_baseline/scripts/run_flocora_three_seeds.ps1
```

## 6. 输出目录

```text
outputs/cifar10/resnet18/flocora/
├── commmatched_seed42/
├── commmatched_seed43/
└── commmatched_seed44/
```

每组包含：

- `resolved_config.json`
- `environment.json`
- `partition_summary.json`
- `flocora_parameter_stats.json`
- `metrics.csv`
- `checkpoint_round_0020.pt` 等周期 checkpoint
- `best_model.pt`
- `last_model.pt`
- `summary.json`

## 7. 通信口径

`metrics.csv` 同时保留三类口径：

- `round_uplink_mib`：客户端实际上传 trainable FLoCoRA state；
- `round_downlink_payload_mib`：FLoCoRA 机制下每轮重复下发的 trainable state；
- `round_downlink_full_model_reference_mib`：完整 dense 模型下行参考值。

论文当前主通信公平性建议继续用 **uplink ratio / uplink MiB per round**，这样与已有约 30% baseline 的口径一致。

## 8. 论文来源与实现说明

FLoCoRA: *Federated learning compression with low-rank adaptation*, EUSIPCO 2024, arXiv:2406.14082。作者公开实现使用 Flower + PEFT；本目录为了与你现有 DRPE 工程兼容，采用 PyTorch 自包含重实现，不引入 Flower/PEFT，但保留论文的关键机制：冻结原始卷积、Conv-LoRA、训练 normalization/FC、仅聚合 trainable parameters。
