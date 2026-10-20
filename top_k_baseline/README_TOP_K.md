# Top-k Baseline（独立目录）

本目录直接放到 `drpe_formal_experiments` 项目根目录下：

```text
drpe_formal_experiments/
├─ src/
├─ scripts/
├─ configs/
├─ random_k_baseline/
└─ top_k_baseline/
   ├─ __init__.py
   ├─ top_k.py
   ├─ run_top_k.py
   ├─ run_top_k_attack.py
   └─ configs/
      └─ cifar10_resnet18/
         └─ top_k_commmatched_seed42_fixed200.json
```

## 方法定义

当前正式 Top-k 基线固定为：

- CIFAR-10 + ResNet-18 + GroupNorm
- 10 clients，client_fraction=0.2，每轮 2 clients
- Dirichlet alpha=0.5
- local_epochs=5
- rounds=200
- batch_size=64
- SGD：lr=0.01，momentum=0.9，weight_decay=1e-4
- LR 每 60 轮乘 0.5
- tensor-wise Top-k
- keep_ratio=0.15
- value=float32
- index=int32
- 无 clipping
- 无 noise
- 无 privacy mechanism
- 无 error feedback

通信匹配规则与 Random-k 相同：每个保留坐标发送 4-byte float32 value + 4-byte int32 index。
因此 keep_ratio=0.15 时通信约为原始 dense float32 更新的 0.30 倍，与 DRPE-30 对齐。

> 论文中建议写成 “tensor-wise Top-k (15% entries, explicit int32 indices, no error feedback)”，不要只写一个模糊的 Top-k。

## Seed 42 训练

```powershell
cd .
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"

python ./top_k_baseline/run_top_k.py `
  --config ./top_k_baseline/configs/cifar10_resnet18/top_k_commmatched_seed42_fixed200.json
```

## Seed 43

```powershell
python ./top_k_baseline/run_top_k.py `
  --config ./top_k_baseline/configs/cifar10_resnet18/top_k_commmatched_seed42_fixed200.json `
  --set seed=43 `
        partition_seed=43 `
        output_dir=./outputs/cifar10/resnet18/top_k/commmatched_seed43
```

## Seed 44

```powershell
python ./top_k_baseline/run_top_k.py `
  --config ./top_k_baseline/configs/cifar10_resnet18/top_k_commmatched_seed42_fixed200.json `
  --set seed=44 `
        partition_seed=44 `
        output_dir=./outputs/cifar10/resnet18/top_k/commmatched_seed44
```

## Round-20 攻击协议

与 FedAvg / Random-k / DRPE 主隐私实验保持一致：

- checkpoint：`checkpoint_round_0020.pt`
- 模拟第 21 轮上传
- sample_index=1
- DLG + iDLG
- 2000 iterations
- Adam
- cosine gradient matching
- attack_lr=0.1
- tv_weight=1e-4

Top-k 攻击者知道实际上传的 Top-k indices 和 values，并在这些已观测坐标上匹配候选梯度。

### Seed 42 DLG

```powershell
python ./top_k_baseline/run_top_k_attack.py `
  --checkpoint ./outputs/cifar10/resnet18/top_k/commmatched_seed42/checkpoint_round_0020.pt `
  --output-dir ./outputs/cifar10/resnet18/attacks/top_k/seed42/dlg_round20 `
  --attack dlg `
  --seed 42 `
  --round-index 21 `
  --client-id 0 `
  --sample-index 1 `
  --keep-ratio 0.15
```

### Seed 42 iDLG

```powershell
python ./top_k_baseline/run_top_k_attack.py `
  --checkpoint ./outputs/cifar10/resnet18/top_k/commmatched_seed42/checkpoint_round_0020.pt `
  --output-dir ./outputs/cifar10/resnet18/attacks/top_k/seed42/idlg_round20 `
  --attack idlg `
  --seed 42 `
  --round-index 21 `
  --client-id 0 `
  --sample-index 1 `
  --keep-ratio 0.15
```

Seed 43/44 只需要同步修改 checkpoint/output-dir/seed。
