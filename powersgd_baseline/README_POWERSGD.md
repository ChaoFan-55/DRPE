# PowerSGD Baseline（独立根目录）

把整个 `powersgd_baseline/` 放到 `drpe_formal_experiments` 根目录：

```text
drpe_formal_experiments/
├─ src/
├─ random_k_baseline/
├─ top_k_baseline/
├─ dense_rp_baseline/
└─ powersgd_baseline/
   ├─ __init__.py
   ├─ powersgd.py
   ├─ run_powersgd.py
   ├─ run_powersgd_attack.py
   ├─ README_POWERSGD.md
   └─ configs/
      └─ cifar10_resnet18/
         ├─ powersgd_commmatched_seed42_fixed200.json
         ├─ powersgd_commmatched_seed43_fixed200.json
         └─ powersgd_commmatched_seed44_fixed200.json
```

## 正式 baseline 定义

当前实验使用 communication-matched PowerSGD-style low-rank compressor：

- CIFAR-10 + ResNet-18 + GroupNorm
- 10 clients，每轮 2 clients
- Dirichlet alpha = 0.5
- local_epochs = 5
- rounds = 200
- batch = 64
- SGD lr=0.01, momentum=0.9, weight_decay=1e-4
- lr step=60, gamma=0.5
- PowerSGD rank = 97
- one-step power iteration
- convolution / linear tensors reshape 为 `[shape[0], numel / shape[0]]`
- 1D tensors 保持 dense
- 若 `r(m+n) >= mn`，该 tensor 自动 fallback 到 dense
- no clipping
- no Gaussian noise
- error feedback = false
- float32 payload
- server reconstructs each client update and sample-weighted aggregates it

为什么 rank=97：

当前 CIFAR-10 ResNet18 一共有 11,173,962 个参数。
按上面的 PowerSGD payload 计数规则，rank=97 时每客户端发送约 3,364,746 个 float32，
即约 0.301124 × dense；每轮 2 clients 约 25.67 MiB。

这个预算与：
- Random-k ≈25.58 MiB
- Top-k ≈25.58 MiB
- Dense-RP ≈25.64 MiB
- DRPE ≈25.64 MiB

基本对齐。

三 seed：

| Train seed | Partition seed | PowerSGD seed |
|---:|---:|---:|
| 42 | 42 | 2026 |
| 43 | 43 | 2027 |
| 44 | 44 | 2028 |

## 先跑 1-round smoke

```powershell
cd .
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"

python ./powersgd_baseline/run_powersgd.py `
  --config ./powersgd_baseline/configs/cifar10_resnet18/powersgd_commmatched_seed42_fixed200.json `
  --set rounds=1 `
        output_dir=./outputs/cifar10/resnet18/powersgd/smoke_seed42
```

正常日志重点看：

```text
Method=POWERSGD
Rank=97
power_iterations=1
error_feedback=False
Predicted payload=3,364,746/11,173,962 (0.301124x)
Round 001 ... uplink≈25.67 MiB (≈0.301x raw)
```

## 正式训练

### Seed 42

```powershell
python ./powersgd_baseline/run_powersgd.py `
  --config ./powersgd_baseline/configs/cifar10_resnet18/powersgd_commmatched_seed42_fixed200.json
```

### Seed 43

```powershell
python ./powersgd_baseline/run_powersgd.py `
  --config ./powersgd_baseline/configs/cifar10_resnet18/powersgd_commmatched_seed43_fixed200.json
```

### Seed 44

```powershell
python ./powersgd_baseline/run_powersgd.py `
  --config ./powersgd_baseline/configs/cifar10_resnet18/powersgd_commmatched_seed44_fixed200.json
```

## Round-20 DLG / iDLG 协议

与 Random-k / Top-k / Dense-RP 的主攻击协议统一：

- checkpoint = checkpoint_round_0020.pt
- simulated upload round = 21
- client_id = 0
- sample_index = 1
- 2000 iterations
- Adam
- cosine matching
- attack_lr = 0.1
- tv_weight = 1e-4

攻击者知道 PowerSGD 机制、rank 和 deterministic Q0 seed，
并直接匹配实际 transmitted dense/P/Q payload。

### Seed 42 DLG

```powershell
python ./powersgd_baseline/run_powersgd_attack.py `
  --checkpoint ./outputs/cifar10/resnet18/powersgd/commmatched_seed42/checkpoint_round_0020.pt `
  --output-dir ./outputs/cifar10/resnet18/attacks/powersgd/seed42/dlg_round20 `
  --attack dlg `
  --seed 42 `
  --powersgd-seed 2026 `
  --round-index 21 `
  --client-id 0 `
  --sample-index 1 `
  --rank 97
```

### Seed 42 iDLG

```powershell
python ./powersgd_baseline/run_powersgd_attack.py `
  --checkpoint ./outputs/cifar10/resnet18/powersgd/commmatched_seed42/checkpoint_round_0020.pt `
  --output-dir ./outputs/cifar10/resnet18/attacks/powersgd/seed42/idlg_round20 `
  --attack idlg `
  --seed 42 `
  --powersgd-seed 2026 `
  --round-index 21 `
  --client-id 0 `
  --sample-index 1 `
  --rank 97
```

Seed43 使用 powersgd_seed=2027；Seed44 使用 powersgd_seed=2028。
