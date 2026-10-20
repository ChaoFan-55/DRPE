from __future__ import annotations
import argparse
import subprocess
import sys
from common import PACKAGE,project_root,validate_protocol


def main():
    p=argparse.ArgumentParser(description='Run all 120 new attacks, with automatic completed-task skipping.')
    p.add_argument('--project-root');p.add_argument('--device',default='auto')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--max-runs',type=int)
    p.add_argument('--seed',type=int,choices=[42,43,44]);p.add_argument('--attack',choices=['dlg','idlg'])
    a=p.parse_args();root=project_root(a.project_root);protocol,targets=validate_protocol(root)
    count=0
    for seed in ([a.seed] if a.seed else protocol['model_seeds']):
        for target in targets:
            for attack in ([a.attack] if a.attack else protocol['attacks']):
                for restart in range(2):
                    if a.max_runs is not None and count>=a.max_runs:return
                    cmd=[sys.executable,'-B',str(PACKAGE/'attack.py'),'--project-root',str(root),
                         '--seed',str(seed),'--target',target['target_id'],'--attack',attack,
                         '--restart',str(restart),'--device',a.device]
                    print(subprocess.list2cmdline(cmd),flush=True)
                    if not a.dry_run:subprocess.run(cmd,check=True,cwd=PACKAGE)
                    count+=1


if __name__=='__main__':main()
