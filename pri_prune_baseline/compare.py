from __future__ import annotations

import argparse
import math
import statistics as st

from common import PACKAGE,output_dir,project_root,read_csv,read_json,settings,sha256,validate_protocol,write_csv,write_json


def mean_std(values):
    values=list(values)
    if any(not math.isfinite(v) for v in values):raise ValueError('Non-finite metric; review raw runs before summarizing.')
    return st.mean(values),st.stdev(values) if len(values)>1 else 0.0


def main():
    p=argparse.ArgumentParser(description='Read old DRPE results and compare only complete formal new results.')
    p.add_argument('--project-root');p.add_argument('--training-only',action='store_true')
    a=p.parse_args();root=project_root(a.project_root);protocol,targets=validate_protocol(root)
    train_rows=[]
    for seed in protocol['model_seeds']:
        for label,path in [('drpe',root/f'outputs/cifar10/resnet18/drpe/seed{seed}'),
                           ('pri_prune_magnitude',output_dir(seed))]:
            summary=read_json(path/'summary.json')
            if summary.get('smoke',False):raise ValueError('Smoke results cannot enter comparison tables.')
            if label=='pri_prune_magnitude' and read_json(path/'run_signature.json')['defense']!=settings():
                raise ValueError('Current defense config differs from the trained experiment.')
            metrics=read_csv(path/'metrics.csv')
            if len(metrics)!=200 or [int(r['round']) for r in metrics]!=list(range(1,201)):
                raise ValueError(f'{path}: expected 200 complete rounds.')
            train_rows.append(dict(method=label,seed=seed,test_accuracy=float(summary['test_accuracy']),
                      best_round=int(summary['best_round']),
                      mean_uplink_mib=st.mean(float(r['round_uplink_mib']) for r in metrics),
                      mean_uplink_ratio=st.mean(float(r['uplink_ratio_vs_raw_parameters']) for r in metrics)))
    folder=PACKAGE/'results/comparison'
    write_csv(folder/'training_by_seed.csv',train_rows)
    summary=[]
    for method in ['drpe','pri_prune_magnitude']:
        rows=[r for r in train_rows if r['method']==method]
        acc,sd=mean_std(r['test_accuracy'] for r in rows)
        summary.append(dict(method=method,n_seeds=3,test_accuracy_mean=acc,test_accuracy_std=sd,
                       mean_uplink_mib=st.mean(r['mean_uplink_mib'] for r in rows),
                       mean_uplink_ratio=st.mean(r['mean_uplink_ratio'] for r in rows)))
    write_csv(folder/'training_summary.csv',summary)
    if a.training_only:
        print('Training comparison:',folder);return
    old=read_csv(root/'privacy_main_benchmark/results/summaries/per_run_metrics.csv')
    lpips_path=root/'privacy_main_benchmark/results/summaries/lpips_offline.csv'
    lpips={r['run_id']:float(r['lpips']) for r in read_csv(lpips_path)} if lpips_path.is_file() else {}
    if not lpips:raise FileNotFoundError('Existing offline LPIPS is required for the saved-PNG metric comparison.')
    by_key={}
    for r in old:
        if r['method']!='drpe':continue
        key=(int(r['model_seed']),r['target_id'],r['attack'],int(r['restart']))
        if key in by_key:raise ValueError('Duplicate old DRPE run key.')
        by_key[key]=r
    pairs=[]
    fingerprints={seed:sha256(output_dir(seed)/'attack_checkpoint_round_0020.pt') for seed in protocol['model_seeds']}
    for seed in protocol['model_seeds']:
        for t in targets:
            for attack in protocol['attacks']:
                for restart in range(2):
                    key=(seed,t['target_id'],attack,restart)
                    old_row=by_key[key]
                    path=PACKAGE/'results/attacks'/f'seed{seed}'/t['target_id']/f'restart{restart:02d}'/attack/'run.json'
                    new=read_json(path)
                    if new['status']!='success' or new['smoke'] or new['iterations']!=2000 or new['lpips_source']!='saved_png_alex':
                        raise ValueError(f'Incomplete or non-formal new result: {path}')
                    if new['checkpoint_sha256']!=fingerprints[seed] or new['defense']!=settings():
                        raise ValueError('Attack/training checkpoint or defense configuration mismatch.')
                    if any(int(old_row[k])!=new[k] for k in ['sample_index','checkpoint_round','transmission_round','attack_base_seed']):
                        raise ValueError('Paired old/new attack conditions differ.')
                    for method,r in [('drpe',old_row),('pri_prune_magnitude',new)]:
                        val_lpips=lpips.get(r['run_id']) if method=='drpe' else r['lpips']
                        if val_lpips is None:raise ValueError('Missing offline LPIPS.')
                        pairs.append(dict(method=method,model_seed=seed,target_id=t['target_id'],attack=attack,
                            restart=restart,sample_index=int(t['dataset_index']),psnr=float(r['psnr']),
                            ssim=float(r['ssim']),lpips=float(val_lpips),label_correct=int(r['label_correct'])))
    write_csv(folder/'attack_per_run.csv',pairs)
    attack_summary=[]
    for method in ['drpe','pri_prune_magnitude']:
        for attack in protocol['attacks']:
            rows=[r for r in pairs if r['method']==method and r['attack']==attack]
            if len(rows)!=60:raise ValueError('Every method/attack needs exactly 60 paired runs.')
            row=dict(method=method,attack=attack,n_runs=60,n_unique_targets=10)
            for metric in ['psnr','ssim','lpips']:
                row[metric+'_mean'],row[metric+'_std']=mean_std(r[metric] for r in rows)
            row['label_recovery_percent']=100*st.mean(r['label_correct'] for r in rows)
            attack_summary.append(row)
    write_csv(folder/'attack_summary.csv',attack_summary)
    # Summaries by model seed preserve the repeated-target experimental unit.
    seed_rows=[]
    for method in ['drpe','pri_prune_magnitude']:
        for attack in protocol['attacks']:
            for seed in protocol['model_seeds']:
                rows=[r for r in pairs if r['method']==method and r['attack']==attack and r['model_seed']==seed]
                seed_rows.append(dict(method=method,attack=attack,model_seed=seed,n_runs=len(rows),
                     psnr_mean=st.mean(r['psnr'] for r in rows),ssim_mean=st.mean(r['ssim'] for r in rows),
                     lpips_mean=st.mean(r['lpips'] for r in rows),
                     label_recovery_percent=100*st.mean(r['label_correct'] for r in rows)))
    write_csv(folder/'attack_by_model_seed.csv',seed_rows)
    write_json(folder/'comparison_scope.json',dict(
        training='Identical original global validation/test data, common model, seeds and optimizer schedule.',
        communication='Actual values+int32 indices+counts; native adaptive defense; no forced 30% label.',
        attack='Identical target list, round20 stage, seeds and 2000-step optimizer; full local-model oracle for PriPrune.',
        limitation='Protocol-adapted independent mechanism implementation, not author code or native SGI reproduction. Different learned model states remain a confounder.',
        statistical_unit='10 unique targets repeated across 3 model seeds and 2 attack initializations; no significance tests treating 60 runs as independent targets.'))
    print('Complete comparison saved:',folder)


if __name__=='__main__':main()
