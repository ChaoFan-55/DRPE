# PriPrune-Magnitude 防御基线运行说明

这是适配现有DRPE工程的独立实现。请先阅读 `METHOD_NOTES.md` 的机制对应、优化器选择与本地模型oracle差异，再决定是否用于论文实验。它不是官方源码的直接移植，未验证复现原论文结果。

## 放置与环境

将压缩包中的整个 `pri_prune_baseline` 文件夹放到：

```text
./pri_prune_baseline
```

代码只读旧工程的 `src/`、CIFAR数据、DRPE配置与既有结果。训练、攻击、汇总和临时文件全部写到新目录的 `results/`。禁用Python字节码生成，避免导入旧模块时在旧目录写入`__pycache__`。不会更新旧基线源码、配置、检查点或统计文件。

使用你旧攻击日志记录的环境（当前核对为Python3.10、torch1.13.1+cu116、torchvision0.14.1+cu116、RTX3050Ti Laptop）。根目录requirements.txt的torch>=2.1与实际环境不同，本包不要求按它升级。

```powershell
conda activate PTF-FSR
cd '.'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
```

既有环境已装lpips。若另一环境缺少它，只安装`lpips`，保持torch/torchvision版本不变。正式攻击要求LPIPS可用；不会悄悄把缺失指标当作有效结果。

## 1 检查公平性和已有数据

```powershell
./pri_prune_baseline/01_preflight.ps1
```

检查原配置、客户端划分、5000张验证集、10000张测试集、攻击协议、目标列表与上传字节计算。`results/preflight.json`保存配置和源文件指纹。

## 2 先跑功能检查

```powershell
./pri_prune_baseline/02_smoke.ps1 -Device cuda
```

只做2轮、每客户端1个批次，再做DLG与iDLG各3步。只验证训练/攻击连接和输出文件，**不代表方法收敛或防御有效**。所有smoke结果在单独目录，不能进入正式比较。

## 3 正式训练三个种子

```powershell
./pri_prune_baseline/03_train_three_seeds.ps1 -Device cuda
```

每个种子200轮，每轮2/10客户端，local_epochs=5，SGD lr=0.01、momentum=0.9、weight_decay=1e-4；lr每60轮乘0.5。保存验证集最佳模型，按该模型测试，不使用测试集挑选检查点。输出：

```text
results/formal/seed42/metrics.csv
results/formal/seed42/summary.json
results/formal/seed42/attack_checkpoint_round_0020.pt
results/formal/seed42/resume.pt
```

私有权重/掩码增加CPU内存与检查点大小；resume文件可能接近1GiB。它只保存一份可覆盖的新基线恢复状态，不触碰旧结果。中断后重跑脚本从最近20轮检查点继续；保存了抽样、加载器和随机数状态。保存前的未完成轮次重新计算。若在首次保存之前中断，没有可恢复检查点，请把**新包的**对应seed结果目录改名保留后重新运行，勿删除旧工程结果。

只训练某个种子：

```powershell
python -B ./pri_prune_baseline/train.py --seed 42 --device cuda
# 有恢复检查点时
python -B ./pri_prune_baseline/train.py --seed 42 --device cuda --resume
```

## 4 正式攻击120次

```powershell
python -B ./pri_prune_baseline/run_attacks.py --dry-run --max-runs 4
./pri_prune_baseline/04_attack_main.ps1 -Device cuda
```

每类攻击60次：3个模型种子×10个既有目标×2次初始化。攻击种子为3401/3402，实际图像初始化种子为`attack_base_seed + dataset_index*1009`，复用旧工程优化器与选图代码。中断后重跑自动跳过已成功任务。

分批执行：

```powershell
python -B ./pri_prune_baseline/run_attacks.py --seed 42 --attack dlg --device cuda
python -B ./pri_prune_baseline/run_attacks.py --seed 42 --attack idlg --device cuda
```

每次保留target.png、最佳重建图、progress.csv和run.json。PSNR/SSIM复用原实现；LPIPS统一从保存后的PNG计算，AlexNet、[-1,1]输入，与既有离线结果口径一致。首次LPIPS初始化可能下载预训练权重，沿用旧环境的缓存机制。

## 5 比较已有DRPE结果

```powershell
./pri_prune_baseline/05_compare.ps1
```

只读旧DRPE三种子的训练结果与`privacy_main_benchmark/results/summaries`中的逐次攻击和离线LPIPS，写出：

```text
results/comparison/training_by_seed.csv
results/comparison/training_summary.csv
results/comparison/attack_per_run.csv
results/comparison/attack_summary.csv
results/comparison/attack_by_model_seed.csv
results/comparison/comparison_scope.json
```

三种子训练或120次攻击缺失时，完整比较会报错。可以先看训练比较：

```powershell
python -B ./pri_prune_baseline/compare.py --training-only
```

不要只看攻击表决定优劣：必须同时读取准确率与实际上传量，且写明此方案是协议适配重实现。上传比例随防御掩码变化，不能把它默认写成30%。iDLG的标签恢复率也不等于图像重建成功率。

## 修改防御超参数

只有 `config.json` 中的防御参数是新增配置。若要调整，先把新包结果目录改名留存并启用一份新配置，避免混用不同版本结果；旧DRPE结果不需要改动。不得看测试准确率或攻击目标结果挑选配置。超参数选择应基于独立的验证流程，并如实报告搜索范围与选择规则。

## 可选单元测试

```powershell
python -B -m pytest -q -p no:cacheprovider ./pri_prune_baseline/tests
```

测试检查掩码梯度、私有状态保留、聚合、上传字节和代理模型组合。交付的测试/短流程记录只证明实现能运行；正式200轮训练和120次2000步攻击由你运行后才有可用于论文的结果。
