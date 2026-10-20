# PowerSGD PQ^T decoded-product attack（新增版，不覆盖原攻击）

这个补丁只在现有 `powersgd_baseline/` 目录中新增：

- `run_powersgd_attack_pqt_decoded.py`

不会覆盖：

- `powersgd_baseline/run_powersgd_attack.py`
- `powersgd_baseline/powersgd.py`
- PowerSGD 训练代码、配置和 checkpoint

因此两套攻击可以并存：

- 原攻击：`run_powersgd_attack.py` —— 直接匹配 transmitted dense/P/Q payload
- 新攻击：`run_powersgd_attack_pqt_decoded.py` —— 对低秩张量匹配 `P Q^T` decoded product；dense fallback 仍匹配实际传输 dense tensor

## Seed42 gate

```powershell
python .\powersgd_baseline\run_powersgd_attack_pqt_decoded.py `
  --checkpoint .\outputs\cifar10\resnet18\powersgd\commmatched_seed42\checkpoint_round_0020.pt `
  --output-dir .\outputs\cifar10\resnet18\attacks\powersgd_pqt_decoded\seed42\dlg_round20 `
  --attack dlg `
  --seed 42 `
  --powersgd-seed 2026 `
  --round-index 21 `
  --client-id 0 `
  --sample-index 1 `
  --rank 97
```

```powershell
python .\powersgd_baseline\run_powersgd_attack_pqt_decoded.py `
  --checkpoint .\outputs\cifar10\resnet18\powersgd\commmatched_seed42\checkpoint_round_0020.pt `
  --output-dir .\outputs\cifar10\resnet18\attacks\powersgd_pqt_decoded\seed42\idlg_round20 `
  --attack idlg `
  --seed 42 `
  --powersgd-seed 2026 `
  --round-index 21 `
  --client-id 0 `
  --sample-index 1 `
  --rank 97
```

新攻击日志会显示 `obs=pqt_decoded`，输出文件名也带 `powersgd_pqt_decoded`，因此不会和原攻击结果混在一起。
