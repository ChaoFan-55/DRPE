# privacy_main_benchmark

这是 DRPE 论文的统一多样本隐私主实验目录。它只负责**统一协议、统一目标样本、统一实验矩阵、调用已有攻击脚本、统一保存和统计**，不会覆盖或修改 `random_k_baseline`、`top_k_baseline`、`dense_rp_baseline`、`powersgd_baseline`、`flocora_baseline` 和 DRPE/FedAvg 原有实现。

## 目录放置

把整个 `privacy_main_benchmark` 文件夹放到工程根目录：

```text
./privacy_main_benchmark
```

所有新结果只写到：

```text
privacy_main_benchmark/results/
```

因此不会和前面各 baseline 的 `outputs/.../attacks` 冲突。

## v1 固定协议

- CIFAR-10 / ResNet18 / GroupNorm
- round-20 checkpoint，模拟 round-21 单客户端传输
- model seeds: 42 / 43 / 44
- attacks: DLG + iDLG
- 2000 iterations, Adam, cosine matching
- attack_lr=0.1, label_lr=0.05, TV=1e-4, label entropy=1e-3
- 10 个固定、类别均衡目标：CIFAR-10 每类 1 张训练样本
- pilot：3 targets × 1 restart
- main：10 targets × 2 independent attack base seeds
- methods: FedAvg / Random-k / Top-k / Dense-RP / PowerSGD / FLoCoRA / DRPE

`model_seed` 与 `attack_base_seed` 在 manifest 中完全分开。对于现有独立攻击脚本，`--seed` 被用于攻击初始化，而 checkpoint 与显式的 projection/random-k/PowerSGD seed 决定对应训练模型/传输机制。

## 第一次运行

在工程根目录执行：

```powershell
cd .
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

先验证已有攻击脚本、checkpoint 与 DRPE calibration 路径：

```powershell
python ./privacy_main_benchmark/scripts/validate_protocol.py
```

如果某个本地目录名和这里不一致，只修改：

```text
privacy_main_benchmark/configs/methods.json
```

不要改各 baseline 源码。

## 生成固定的 10 张攻击目标

```powershell
python ./privacy_main_benchmark/scripts/build_target_samples.py
```

首次会写入 `target_samples.csv`。之后默认不会覆盖，保证整个论文期间所有方法始终攻击完全相同的目标样本。

## Pilot：先验证 126 个任务矩阵

生成 manifest：

```powershell
python ./privacy_main_benchmark/scripts/build_manifest.py --mode pilot
```

先只打印命令，不真正运行：

```powershell
python ./privacy_main_benchmark/scripts/run_privacy_pilot.py --dry-run
```

建议先实际跑 6 个任务做 gate：

```powershell
python ./privacy_main_benchmark/scripts/run_privacy_pilot.py --max-runs 6
```

确认路径、日志、JSON/CSV 和重建图均正常后跑完 pilot：

```powershell
python ./privacy_main_benchmark/scripts/run_privacy_pilot.py
```

运行器默认支持 resume：已经成功的目录会自动 `SKIP`，中断后直接重跑同一条命令即可继续。

## Main：正式 840 次攻击

```powershell
python ./privacy_main_benchmark/scripts/build_manifest.py --mode main
```

先检查命令：

```powershell
python ./privacy_main_benchmark/scripts/run_privacy_main.py --dry-run --max-runs 20
```

正式执行：

```powershell
python ./privacy_main_benchmark/scripts/run_privacy_main.py
```

也可以按方法分批：

```powershell
python ./privacy_main_benchmark/scripts/run_privacy_main.py --method drpe
python ./privacy_main_benchmark/scripts/run_privacy_main.py --method fedavg
python ./privacy_main_benchmark/scripts/run_privacy_main.py --method random_k
python ./privacy_main_benchmark/scripts/run_privacy_main.py --method top_k
python ./privacy_main_benchmark/scripts/run_privacy_main.py --method dense_rp
python ./privacy_main_benchmark/scripts/run_privacy_main.py --method powersgd
python ./privacy_main_benchmark/scripts/run_privacy_main.py --method flocora
```

也可只跑某 seed/attack：

```powershell
python ./privacy_main_benchmark/scripts/run_privacy_main.py --method drpe --model-seed 42 --attack dlg
```

## 输出组织

例如：

```text
results/raw/flocora/seed42/T03/restart01/idlg/
```

唯一表示：FLoCoRA / trained model seed42 / target T03 / attack restart01 / iDLG。

每个任务还有：

```text
run_command.txt
console.log
run_meta.json
```

便于审计与复现。

## 汇总

正式实验跑完或跑到一部分都可以汇总：

```powershell
python ./privacy_main_benchmark/scripts/summarize_privacy_main.py
```

主要输出：

```text
results/summaries/per_run_metrics.csv
results/summaries/privacy_summary_by_method_attack.csv
results/summaries/missing_runs.txt
```

统计指标：PSNR、SSIM、LPIPS、label ASR。

生成矢量图：

```powershell
python ./privacy_main_benchmark/scripts/plot_privacy_main.py
```

保存 PDF / SVG / PNG 到 `results/figures`。

## 重要口径

1. 这是统一的 **single-example gradient-level / representation-aware empirical privacy benchmark**，不是重放历史 5 local epochs 后的真实客户端完整更新。
2. 不同方法看到的传输表示不同，这是设计的一部分；统一的是 target、checkpoint stage、攻击预算、攻击初始化集合和评价指标，而不是把所有方法强行改成 raw gradient。
3. PowerSGD 默认使用项目中已经验证的 `PQt decoded-product` 攻击脚本。
4. DRPE v1 主协议保持当前项目已经使用的口径：DLG 对 encoded observation，iDLG 对 decoded observation；若要增加 adaptive encoded iDLG，应作为单独 sensitivity/threat-model 实验，不要混入同一统计列。
5. `target_samples.csv` 一旦正式开始跑主实验就不要重新生成。
