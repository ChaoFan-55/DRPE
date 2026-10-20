# DRPE Formal Experiments

这是从探索性工程中重新整理出的正式论文实验项目。项目只保留两类训练方法：

- **FedAvg**：完整更新上传基线；
- **DRPE**：固定采用 `ratio=0.3 + P75全局裁剪 + eta=2.0编码噪声 + 动态投影`。

支持：

- 数据集：CIFAR-10、CIFAR-100、Tiny-ImageNet；
- 模型：ResNet-18、MobileNetV2；
- 随机种子：42、43、44；
- 攻击：FedAvg raw DLG/iDLG，DRPE adaptive encoded DLG 与 decoded iDLG；
- 自动汇总CSV与PDF/SVG/PNG收敛图。

## 1. 项目结构

```text
configs/<dataset>/<model>/
├── calibration/   # 生成该数据集-模型组合的P75
├── fedavg/        # 3个随机种子
├── drpe/          # 3个随机种子，自动读取P75并计算sigma
└── attacks/
    ├── fedavg/    # raw DLG/iDLG
    └── drpe/      # encoded DLG、decoded iDLG

outputs/<dataset>/<model>/...
figures/
tables/
scripts/launchers/<dataset>/<model>/
```

## 2. 安装与数据

```powershell
pip install -r requirements.txt
python -m pytest -q
```

数据统一放在 `data/`，详见 `data/README.md`。

## 3. 正式实验顺序

每个“数据集-模型”组合按以下顺序运行：

```powershell
# 1) 运行无隐私DRPE-30校准，生成 outputs/.../calibration/calibration.json
.\scripts\launchers\cifar100\resnet18\01_run_calibration.ps1

# 2) FedAvg三种子
.\scripts\launchers\cifar100\resnet18\02_run_fedavg_three_seeds.ps1

# 3) 完整DRPE三种子
.\scripts\launchers\cifar100\resnet18\03_run_drpe_three_seeds.ps1

# 4) DLG/iDLG攻击（计算量很大）
.\scripts\launchers\cifar100\resnet18\04_run_dlg_idlg_all_seeds.ps1
```

CIFAR-10+ResNet-18已经内置此前确定的P75文件，因此可以直接运行FedAvg和DRPE。其他组合必须先运行校准。

## 4. P75与eta=2.0的实现

校准运行使用：

- DRPE-30动态投影；
- 不裁剪；
- 不加噪；
- 独立校准种子2026；
- 200轮。

脚本对200个 `mean_raw_update_norm`（每轮参与客户端原始更新范数的均值）取P75。正式DRPE读取该P75作为C，并自动计算：

```text
sigma = 2.0 * C / sqrt(M)
```

其中M为当前模型在ratio=0.3、block=512下的实际编码元素总数。因此换数据集或模型时不复用CIFAR-10+ResNet-18的绝对阈值和sigma。

## 5. 随机种子协议

```text
seed42: partition=42, projection=2026, noise=314159
seed43: partition=43, projection=2027, noise=314160
seed44: partition=44, projection=2028, noise=314161
```

每个PowerShell脚本都在Python启动前设置：

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

## 6. 汇总和矢量图

所有训练完成后：

```powershell
python scripts/summarize_results.py
python scripts/plot_results.py
```

输出：

```text
tables/main_results_by_seed.csv
tables/main_results_summary.csv
figures/convergence_<dataset>_<model>.pdf
figures/convergence_<dataset>_<model>.svg
figures/convergence_<dataset>_<model>.png
```

## 7. 重要说明

1. DRPE中的eta是经验相对噪声强度，不是经过会计的差分隐私noise multiplier。
2. 投影种子不作为秘密；DRPE的DLG配置直接在编码空间匹配，是防御感知白盒攻击。
3. 首次跑新组合时，建议先用 `--set rounds=1 local_epochs=1` 做smoke test。
4. Tiny-ImageNet官方val作为最终测试集，官方train内部按类别分层划出10%验证集。
5. 本项目不包含此前ratio扫描、静态投影和噪声筛选等探索配置；它们已完成“确定方法”的作用，不进入正式主工程。
