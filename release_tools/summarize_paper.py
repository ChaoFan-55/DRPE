"""Rebuild paper tables from recorded results; standard library only, no training."""
from pathlib import Path
from collections import defaultdict
import argparse, csv, json, math, statistics

ROOT=Path(__file__).resolve().parents[1]
def read(p):
    with (ROOT/p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(out,name,rows):
    if not rows:raise ValueError(name+' has no rows')
    with (out/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(rows):
    result={'n_runs':len(rows)}
    for key in ['psnr','ssim','lpips']:
        vals=[float(r[key]) for r in rows]
        if not all(math.isfinite(v) for v in vals):raise ValueError('Invalid metric: '+key)
        result[key+'_mean']=statistics.mean(vals);result[key+'_sd']=statistics.stdev(vals)
    result['label_recovery_percent']=100*statistics.mean(float(r['label_correct']) for r in rows)
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='paper_data/derived');a=p.parse_args()
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    training=[];curves=[]
    for row in read('training_runs.csv'):
        folder=row['record_dir'];summary=json.loads((ROOT/folder/'summary.json').read_text(encoding='utf-8'))
        metrics=read(folder+'/metrics.csv');assert len(metrics)==200,(folder,len(metrics))
        training.append({**{k:row[k] for k in ['run_id','dataset','model','method','seed']},
                         'test_accuracy':float(summary['test_accuracy']),
                         'uplink_mib':statistics.mean(float(r['round_uplink_mib']) for r in metrics)})
        curves.extend({'run_id':row['run_id'],'method':row['method'],'dataset':row['dataset'],
                       'model':row['model'],'seed':row['seed'],'round':r['round'],
                       'val_accuracy':r['val_accuracy'],'round_uplink_mib':r['round_uplink_mib']} for r in metrics)
    write(out,'training_by_seed.csv',training);write(out,'training_curves.csv',curves)
    main_train=[r for r in training if r['dataset']=='cifar10' and r['model']=='resnet18']
    groups=defaultdict(list)
    for r in main_train:groups[r['method']].append(r)
    base=statistics.mean(r['test_accuracy'] for r in groups['fedavg'])
    base_up=statistics.mean(r['uplink_mib'] for r in groups['fedavg'])
    table2=[]
    methods=['fedavg','random_k','top_k','powersgd','flocora','dense_rp','random_k_clip_noise','pri_prune_magnitude','drpe']
    for method in methods:
        rows=groups[method];assert len(rows)==3
        acc=statistics.mean(r['test_accuracy'] for r in rows);up=statistics.mean(r['uplink_mib'] for r in rows)
        table2.append({'method':method,'accuracy_mean':acc,'accuracy_sd':statistics.stdev(r['test_accuracy'] for r in rows),
                       'uplink_mib':up,'bytes_ratio':up/base_up,'gap_pp':acc-base})
    write(out,'table2_training.csv',table2)
    table3=[]
    for method in ['static','clip_only','drpe']:
        rows=sorted(groups[method],key=lambda r:int(r['seed']));vals=[r['test_accuracy'] for r in rows]
        table3.append({'configuration':method,**{f'seed{s}':v for s,v in zip([42,43,44],vals)},'mean':statistics.mean(vals),'sd':statistics.stdev(vals)})
    write(out,'table3_components.csv',table3)

    # Join original per-run main results with the PriPrune-only rows, not duplicated DRPE rows.
    main=read('privacy_main_benchmark/results/summaries/per_run_metrics.csv')
    pri=[r for r in read('pri_prune_baseline/results/comparison/attack_per_run.csv') if r['method']=='pri_prune_magnitude']
    assert len(main)==840 and len(pri)==120
    attacks=main+pri;ag=defaultdict(list)
    for r in attacks:ag[(r['method'],r['attack'])].append(r)
    table4=[]
    for attack in ['dlg','idlg']:
        for method in [m for m in methods if m!='random_k_clip_noise']:
            rows=ag[(method,attack)];assert len(rows)==60,(method,attack,len(rows))
            table4.append({'method':method,'attack':attack,**stats(rows)})
    write(out,'table4_attacks.csv',table4)
    table5=[]
    for method in ['dense_rp','drpe']:
        for correct in [0,1]:
            rows=[r for r in ag[(method,'idlg')] if int(float(r['label_correct']))==correct]
            table5.append({'method':method,'label_correct':correct,**stats(rows)})
    write(out,'table5_label_groups.csv',table5)
    def raw(folder,attack):
        rows=[]
        for seed in [42,43,44]:
            for target in range(10):
                for restart in [0,1]:
                    rr=read(f'{folder}/seed{seed}/T{target:02}/restart{restart:02}/{attack}/classic_attack_metrics.csv')
                    assert len(rr)==1
                    rows.extend(rr)
        return rows
    oracle={m:raw('privacy_oracle_benchmark/results/raw/'+m,'oracle') for m in ['fedavg_raw','drpe_encoded','drpe_decoded']}
    write(out,'table6_known_labels.csv',[{'observation':m,**stats(r)} for m,r in oracle.items()])
    table7=[]
    for attack in ['dlg','idlg','oracle']:
        no=raw('privacy_component_benchmark/results/clip_only_model_eta0',attack)
        full=oracle['drpe_encoded'] if attack=='oracle' else ag[('drpe',attack)]
        for name,rr in [('no_noise',no),('full_drpe',full)]:table7.append({'attack':attack,'setting':name,**stats(rr)})
    write(out,'table7_noise_comparison.csv',table7)
    write(out,'table8_projection_overhead.csv',read('projection_efficiency_benchmark/results/benchmark_summary.csv'))
    table9=[]
    for dataset,model in [('cifar10','resnet18'),('cifar10','mobilenet_v2'),('cifar100','resnet18')]:
        gg={m:[r for r in training if r['dataset']==dataset and r['model']==model and r['method']==m] for m in ['fedavg','drpe']}
        means={m:statistics.mean(r['test_accuracy'] for r in rr) for m,rr in gg.items()}
        ups={m:statistics.mean(r['uplink_mib'] for r in rr) for m,rr in gg.items()}
        table9.append({'dataset':dataset,'model':model,'fedavg_accuracy':means['fedavg'],
                       'fedavg_sd':statistics.stdev(r['test_accuracy'] for r in gg['fedavg']),
                       'drpe_accuracy':means['drpe'],'drpe_sd':statistics.stdev(r['test_accuracy'] for r in gg['drpe']),
                       'gap_pp':means['drpe']-means['fedavg'],'uplink_reduction_percent':100*(1-ups['drpe']/ups['fedavg'])})
    write(out,'table9_generalization.csv',table9)

    # Independently compare the final figure snapshots with their run-level sources.
    snapshots=read('paper_figures/figure_sources/main_attacks.csv')
    key=lambda r:(r['method'],r['attack'],int(r['model_seed']),r['target_id'],int(r['restart']))
    actual={key(r):r for r in attacks};assert len(actual)==960
    for r in snapshots:
        match=actual[key(r)]
        for col in ['psnr','ssim','lpips','label_correct']:
            assert abs(float(r[col])-float(match[col]))<1e-10,(key(r),col)
    snapshot_train=read('paper_figures/figure_sources/training_summary.csv')
    train_index={(r['method'],int(r['seed'])):r for r in main_train}
    for r in snapshot_train:
        match=train_index[(r['method'],int(r['seed']))]
        for col in ['test_accuracy','uplink_mib']:assert abs(float(r[col])-float(match[col]))<1e-9
    extra_index={}
    for folder,methods_extra,attack_types in [
        ('privacy_oracle_benchmark/results/raw',['fedavg_raw','drpe_encoded','drpe_decoded'],['oracle']),
        ('privacy_component_benchmark/results',['clip_only_model_eta0'],['dlg','idlg','oracle'])]:
        for method in methods_extra:
            for seed in [42,43,44]:
                for target in range(10):
                    for restart in [0,1]:
                        for attack in attack_types:
                            rr=read(f'{folder}/{method}/seed{seed}/T{target:02}/restart{restart:02}/{attack}/classic_attack_metrics.csv')[0]
                            name='clip_only' if method=='clip_only_model_eta0' else method
                            extra_index[(name,attack,seed,f'T{target:02}',restart)]=rr
    extra_snapshot=read('paper_figures/figure_sources/label_noise_attacks.csv')
    for r in extra_snapshot:
        match=extra_index[key(r)]
        for col in ['psnr','ssim','lpips','label_correct']:assert abs(float(r[col])-float(match[col]))<1e-10
    checks={'45_complete_200_round_training_runs':len(training)==45,'960_main_attack_runs':len(attacks)==960,
            '180_known_label_runs':sum(map(len,oracle.values()))==180,'180_no_noise_attack_runs':True,
            '960_figure_snapshot_rows_match_original_metrics':len(snapshots)==960,
            '33_main_training_snapshot_rows_match_original_metrics':len(snapshot_train)==33,
            '360_extra_figure_snapshot_rows_match_original_metrics':len(extra_snapshot)==360}
    expected={('drpe','idlg','psnr_mean'):13.15,('drpe','idlg','ssim_mean'):0.364,('drpe','idlg','label_recovery_percent'):56.7}
    for (method,attack,metric),value in expected.items():
        row=next(r for r in table4 if r['method']==method and r['attack']==attack)
        digits=3 if metric=='ssim_mean' else (1 if metric=='label_recovery_percent' else 2)
        assert round(row[metric],digits)==value
    assert round(next(r['accuracy_mean'] for r in table2 if r['method']=='drpe'),2)==84.23
    assert round(next(r['accuracy_mean'] for r in table2 if r['method']=='pri_prune_magnitude'),2)==81.99
    (out/'verification.json').write_text(json.dumps({'checks':checks,'all_passed':all(checks.values()),
          'notes':'Statistics reproduce archived runs; no training or new attack experiments were performed. Sample SD uses n-1. Repeated runs are not independent targets.'},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'all_passed':all(checks.values()),'training_runs':len(training),'main_attack_runs':len(attacks),'output':str(out)},indent=2))
if __name__=='__main__':main()
