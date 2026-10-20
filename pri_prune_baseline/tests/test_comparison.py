"""Synthetic fixtures test aggregation only; they are never empirical results."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.dont_write_bytecode=True


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def table(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def test_complete_comparison_and_checkpoint_mismatch_guard(tmp_path):
    package=tmp_path/'baseline';package.mkdir()
    original=Path(__file__).resolve().parents[1]
    for name in ['common.py','compare.py','config.json']:
        shutil.copy2(original/name,package/name)
    defense=json.loads((package/'config.json').read_text())
    root=tmp_path/'old_project';(root/'src').mkdir(parents=True)
    (root/'src/models.py').write_text('# fixture only')
    protocol=dict(dataset='cifar10',model='resnet18',num_classes=10,checkpoint_round=20,
                  transmission_round=21,observed_client_id=0,iterations=2000,optimizer='adam',
                  matching_loss='cosine',attack_lr=.1,label_lr=.05,tv_weight=1e-4,
                  label_entropy_weight=.001,model_seeds=[42,43,44],attacks=['dlg','idlg'],
                  main=dict(attack_base_seeds=[3401,3402]))
    dump(root/'privacy_main_benchmark/protocol.json',protocol)
    targets=[dict(target_id=f'T{i:02d}',dataset_index=100+i,class_id=i) for i in range(10)]
    table(root/'privacy_main_benchmark/target_samples.csv',targets)
    old,offline=[],[]
    for seed in [42,43,44]:
        existing=root/f'outputs/cifar10/resnet18/drpe/seed{seed}'
        new=package/f'results/formal/seed{seed}'
        for path,acc in [(existing,80+seed-42),(new,81+seed-42)]:
            dump(path/'summary.json',dict(test_accuracy=acc,best_round=200,smoke=False))
            table(path/'metrics.csv',[dict(round=i,round_uplink_mib=25.6,uplink_ratio_vs_raw_parameters=.3) for i in range(1,201)])
        dump(new/'run_signature.json',dict(defense=defense))
        ck=new/'attack_checkpoint_round_0020.pt';ck.write_bytes(b'synthetic checkpoint fingerprint only')
        fingerprint=hashlib.sha256(ck.read_bytes()).hexdigest()
        for t in targets:
            for attack in ['dlg','idlg']:
                for restart in [0,1]:
                    rid=f"drpe__{seed}__{t['target_id']}__{attack}__{restart}"
                    row=dict(run_id=rid,method='drpe',model_seed=seed,target_id=t['target_id'],attack=attack,
                             restart=restart,sample_index=t['dataset_index'],checkpoint_round=20,
                             transmission_round=21,attack_base_seed=3401+restart,psnr=10.,ssim=.2,
                             label_correct=1)
                    old.append(row);offline.append(dict(run_id=rid,lpips=.1))
                    value=dict(status='success',smoke=False,iterations=2000,lpips_source='saved_png_alex',
                               checkpoint_sha256=fingerprint,defense=defense,psnr=9.,ssim=.1,lpips=.2,
                               label_correct=0,**{k:row[k] for k in ['sample_index','checkpoint_round','transmission_round','attack_base_seed']})
                    dump(package/f"results/attacks/seed{seed}/{t['target_id']}/restart{restart:02d}/{attack}/run.json",value)
    table(root/'privacy_main_benchmark/results/summaries/per_run_metrics.csv',old)
    table(root/'privacy_main_benchmark/results/summaries/lpips_offline.csv',offline)
    cmd=[sys.executable,'-B',str(package/'compare.py'),'--project-root',str(root)]
    result=subprocess.run(cmd,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    with (package/'results/comparison/attack_summary.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==4 and all(int(r['n_runs'])==60 for r in rows)
    assert all(float(r['label_recovery_percent'])==0 for r in rows if r['method']=='pri_prune_magnitude')
    # A replaced checkpoint must invalidate already completed attack results.
    (package/'results/formal/seed42/attack_checkpoint_round_0020.pt').write_bytes(b'replaced')
    result=subprocess.run(cmd,capture_output=True,text=True)
    assert result.returncode!=0 and 'mismatch' in result.stderr.lower()
