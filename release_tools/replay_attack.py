"""Replay a recorded single-sample attack into a fresh folder; default is dry-run."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['main','known-label','no-noise'],default='main')
    p.add_argument('--method',default='drpe');p.add_argument('--model-seed',type=int,choices=[42,43,44],default=42)
    p.add_argument('--target',choices=[f'T{i:02}' for i in range(10)],default='T00')
    p.add_argument('--attack',choices=['dlg','idlg','oracle'],default='idlg');p.add_argument('--restart',type=int,choices=[0,1],default=0)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    ident=f'{a.suite}_{a.method}_s{a.model_seed}_{a.target}_{a.attack}_r{a.restart}'
    dest=ROOT/'reproduced_attacks'/ident
    if dest.exists():raise FileExistsError('A reproduction folder already exists; use a fresh folder.')
    suffix=f'seed{a.model_seed}/{a.target}/restart{a.restart:02}/{a.attack}'
    pri=a.suite=='main' and a.method=='pri_prune_magnitude'
    if a.suite=='main':
        if a.attack=='oracle':p.error('Main attacks are dlg or idlg')
        folder=ROOT/('pri_prune_baseline/results/attacks' if pri else 'privacy_main_benchmark/results/raw/'+a.method)/suffix
    elif a.suite=='known-label':
        if a.method not in ['fedavg_raw','drpe_encoded','drpe_decoded'] or a.attack!='oracle':p.error('Known-label suite requires an oracle method and attack oracle')
        folder=ROOT/'privacy_oracle_benchmark/results/raw'/a.method/suffix
    else:
        if a.method!='clip_only':p.error('No-noise suite method is clip_only')
        folder=ROOT/'privacy_component_benchmark/results/clip_only_model_eta0'/suffix
    cfgfile=folder/'resolved_classic_attack_config.json'
    checkpoint=None
    if pri:
        checkpoint=ROOT/f'pri_prune_baseline/results/formal/seed{a.model_seed}/attack_checkpoint_round_0020.pt'
        package=dest/'pri_prune_baseline'
        cmd=[sys.executable,'-B',str(package/'attack.py'),'--project-root',str(ROOT),'--seed',str(a.model_seed),'--attack',a.attack,'--target',a.target,'--restart',str(a.restart)]
        if a.execute:
            if not checkpoint.exists():raise FileNotFoundError('Extract DRPE_Attack_Checkpoints.zip first: '+str(checkpoint))
            shutil.copytree(ROOT/'pri_prune_baseline',package,ignore=shutil.ignore_patterns('results','__pycache__','.pytest_cache'))
            cp=package/f'results/formal/seed{a.model_seed}'/checkpoint.name;cp.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(checkpoint,cp)
            except OSError:shutil.copy2(checkpoint,cp)
    elif cfgfile.exists():
        cfg=json.loads(cfgfile.read_text(encoding='utf-8'));cfg['output_dir']=str(dest/'results')
        checkpoint=ROOT/cfg['checkpoint_path'];generated=dest/'attack_config.json'
        cmd=[sys.executable,'-B',str(ROOT/'scripts/run_classic_attack_experiment.py'),'--config',str(generated)]
        if a.execute:
            if not checkpoint.exists():raise FileNotFoundError('Extract DRPE_Attack_Checkpoints.zip first: '+str(checkpoint))
            dest.mkdir(parents=True);generated.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    else:
        meta=folder/'run_meta.json'
        if not meta.exists():p.error('Recorded attack not found: '+str(folder))
        cmd=json.loads(meta.read_text(encoding='utf-8'))['command'];cmd[0]=sys.executable
        cmd[cmd.index('--output-dir')+1]=str(dest/'results')
        checkpoint=ROOT/cmd[cmd.index('--checkpoint')+1]
        if a.execute:
            if not checkpoint.exists():raise FileNotFoundError('Extract DRPE_Attack_Checkpoints.zip first: '+str(checkpoint))
            dest.mkdir(parents=True)
    print(subprocess.list2cmdline(cmd))
    if a.execute:
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    else:print('Dry-run only; add --execute to run the recorded attack into a fresh folder.')
if __name__=='__main__':main()
