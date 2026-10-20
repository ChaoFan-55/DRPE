"""Train one archived configuration into a fresh output directory. Default: dry-run."""
from pathlib import Path
import argparse,csv,json,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id');p.add_argument('--list',action='store_true');p.add_argument('--device',default='cuda');p.add_argument('--execute',action='store_true');a=p.parse_args()
    with (ROOT/'training_runs.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if a.list:
        for r in rows:print(r['run_id'])
        return
    if not a.run_id:p.error('--run-id or --list is required')
    row=next((r for r in rows if r['run_id']==a.run_id),None)
    if row is None:p.error('Unknown run-id')
    base=ROOT/'reproduced_training'/a.run_id
    if row['method']=='pri_prune_magnitude':
        script=base/'pri_prune_baseline/train.py'
        cmd=[sys.executable,'-B',str(script),'--project-root',str(ROOT),'--seed',row['seed'],'--device',a.device]
        if a.execute:
            if base.exists():raise FileExistsError('Choose a fresh location; existing reproduction is not overwritten.')
            shutil.copytree(ROOT/'pri_prune_baseline',base/'pri_prune_baseline',ignore=shutil.ignore_patterns('results','__pycache__','.pytest_cache'))
    else:
        cfg_path=ROOT/row['reproduction_config']
        cmd=[sys.executable,'-B',str(ROOT/row['entry_point']),'--config',str(cfg_path),'--set','device='+a.device]
        if a.execute and base.exists() and any(base.iterdir()):raise FileExistsError('Existing reproduction is not overwritten.')
    print(subprocess.list2cmdline(cmd))
    if a.execute:
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    else:print('Dry-run only. Add --execute to train; archived results are not replaced.')
if __name__=='__main__':main()
