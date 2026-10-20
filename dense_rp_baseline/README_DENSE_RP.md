# Dense Random Projection Baseline

把整个 `dense_rp_baseline/` 放到项目根目录：

```text
drpe_formal_experiments/
├─ src/
├─ scripts/
├─ random_k_baseline/
├─ top_k_baseline/
└─ dense_rp_baseline/
   ├─ __init__.py
   ├─ dense_rp.py
   ├─ run_dense_rp.py
   ├─ run_dense_rp_attack.py
   ├─ README_DENSE_RP.md
   └─ configs/
      └─ cifar10_resnet18/
         ├─ dense_rp_commmatched_seed42_fixed200.json
         ├─ dense_rp_commmatched_seed43_fixed200.json
         └─ dense_rp_commmatched_seed44_fixed200.json
```

## 正式定义

Dense-RP 用作 DRPE 的“投影族”对照：

- block-wise dense random projection
- block size = 512
- compression ratio = 0.3
- Gaussian matrix经过 reduced QR 得到 row-orthonormal P
- 每轮动态更新 P_t
- 同一轮所有客户端使用相同 P_t
- 参数张量使用确定性的 tensor-specific sign diagonal D
- 上传 `y = P_t D x`
- 服务器样本量加权聚合 encoded payload 后，用 `(P_t D)^T` 解码
- reconstruction_mode = projected
- no clipping
- no noise
- no privacy mechanism
- payload 只包含 float32 projected values，不发送 indices

这样 encoded 元素计数规则与 DRPE 的 block-wise ratio=0.3 保持一致，
理论上应接近 DRPE 的约 0.3008x 元素比 / 约 25.64 MiB 每轮（2 clients）。

## 训练设置

与 FedAvg / Random-k / Top-k / DRPE 保持一致：

- CIFAR-10
- ResNet-18 + GroupNorm
- clients = 10
- client_fraction = 0.2
- Dirichlet alpha = 0.5
- local_epochs = 5
- batch_size = 64
- rounds = 200
- SGD lr=0.01, momentum=0.9, weight_decay=1e-4
- lr step=60, gamma=0.5
- deterministic=true

三 seed 对应：

| Train seed | Partition seed | Dense-RP projection seed |
|---:|---:|---:|
| 42 | 42 | 2026 |
| 43 | 43 | 2027 |
| 44 | 44 | 2028 |

## 先做 1 轮 smoke

```powershell
cd .
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"

python ./dense_rp_baseline/run_dense_rp.py `
  --config ./dense_rp_baseline/configs/cifar10_resnet18/dense_rp_commmatched_seed42_fixed200.json `
  --set rounds=1 `
        output_dir=./outputs/cifar10/resnet18/dense_rp/smoke_seed42
```

重点确认：

```text
Method=DENSE-RP
Ratio=0.3
block=512
dynamic=True
Encoded parameters=... (...≈0.3008x)
Privacy extras: clipping=None, noise=0.0
Round 001 ... uplink≈25.64 MiB (≈0.301x raw)
```

## 正式 Seed 42

```powershell
python ./dense_rp_baseline/run_dense_rp.py `
  --config ./dense_rp_baseline/configs/cifar10_resnet18/dense_rp_commmatched_seed42_fixed200.json
```

## 正式 Seed 43

```powershell
python ./dense_rp_baseline/run_dense_rp.py `
  --config ./dense_rp_baseline/configs/cifar10_resnet18/dense_rp_commmatched_seed43_fixed200.json
```

## 正式 Seed 44

```powershell
python ./dense_rp_baseline/run_dense_rp.py `
  --config ./dense_rp_baseline/configs/cifar10_resnet18/dense_rp_commmatched_seed44_fixed200.json
```

## 论文命名建议

建议写：

> Block-wise Dense Orthogonal Random Projection (Dense-RP)

并在实验设置中明确：
`block_size=512`, `compression_ratio=0.3`, dynamic per-round projection,
transpose decoding, no clipping/noise.

不要把它描述成正式隐私机制或差分隐私方法。


## Round-20 攻击（训练完成后）

与前面的 Random-k / Top-k 主协议保持一致：

- checkpoint_round_0020.pt
- round_index=21
- sample=1
- 2000 iterations
- Adam + cosine
- DLG / iDLG
- 攻击者知道 Dense-RP 投影机制和当前轮投影
- 直接匹配 encoded upload
- iDLG 先从转置解码后的分类头梯度推断标签，再在 encoded 空间做白盒匹配

Seed 42 DLG 示例：

```powershell
python ./dense_rp_baseline/run_dense_rp_attack.py `
  --checkpoint ./outputs/cifar10/resnet18/dense_rp/commmatched_seed42/checkpoint_round_0020.pt `
  --output-dir ./outputs/cifar10/resnet18/attacks/dense_rp/seed42/dlg_round20 `
  --attack dlg `
  --seed 42 `
  --projection-seed 2026 `
  --round-index 21 `
  --sample-index 1
```
