from __future__ import annotations
import argparse
from common import PACKAGE, project_root, read_csv, read_json, settings, sha256, source_config, validate_protocol, write_json


def main():
    p=argparse.ArgumentParser(description='Read-only inspection of the old project; writes its report only inside the new baseline.')
    p.add_argument('--project-root'); p.add_argument('--check-data',action='store_true')
    args=p.parse_args(); root=project_root(args.project_root)
    import inspect
    import torch
    import torchvision
    from src.models import build_model
    from src.data import build_federated_data
    from src.attacks.classic_attack import _run_single_attack
    from common import partition_digest
    from defense import base_mask, payload_bytes, sparse_payload
    from src.utils import set_global_seed
    protocol,targets=validate_protocol(root)
    report={'root':str(root),'torch':torch.__version__,'torchvision':torchvision.__version__,
            'defense':settings(),'attack_target_indices':[int(t['dataset_index']) for t in targets],
            'checks':[],'source_hashes':{}}
    expected=dict(dataset='cifar10',model_name='resnet18',norm='group',num_clients=10,
                  client_fraction=.2,dirichlet_alpha=.5,validation_ratio=.1,rounds=200,
                  local_epochs=5,batch_size=64,learning_rate=.01,momentum=.9,weight_decay=1e-4,
                  lr_decay_step=60,lr_decay_gamma=.5,server_lr=1.0,eval_every=1,save_every=20,
                  early_stop_patience=0,num_workers=0)
    if 'original_shapes' not in inspect.signature(_run_single_attack).parameters:
        raise ValueError('The existing attack helper interface changed.')
    for seed in [42,43,44]:
        cfg,path=source_config(root,seed)
        for key,value in expected.items():
            if getattr(cfg,key)!=value: raise ValueError(f'Seed {seed}: {key} does not match the audited protocol.')
        if cfg.seed!=seed or cfg.partition_seed!=seed: raise ValueError('Seed mismatch.')
        set_global_seed(seed,deterministic=True)
        model=build_model('resnet18',10,'group',32)
        parameters=dict(model.named_parameters())
        bm=base_mask(parameters,settings()['base_keep_ratio'],settings()['prune_scope'])
        upper=payload_bytes(sparse_payload(parameters,bm))
        raw=sum(x.numel()*x.element_size() for x in parameters.values())
        row={'seed':seed,'source':str(path),'num_parameters':sum(x.numel() for x in parameters.values()),
             'raw_bytes_per_client':raw,'maximum_base_upload_bytes_per_client':upper,
             'maximum_base_upload_ratio':upper/raw}
        if args.check_data:
            data=build_federated_data(cfg)
            row['partition_sha256']=partition_digest(data.client_indices)
            row['client_sizes']=data.client_sizes
            saved=root/f'outputs/cifar10/resnet18/drpe/seed{seed}/partition_summary.json'
            if saved.is_file() and read_json(saved)['client_sizes']!=data.client_sizes:
                raise ValueError('Actual partition does not match the previous DRPE partition.')
            row['validation_size']=len(data.val_loader.dataset)
            row['test_size']=len(data.test_loader.dataset)
        report['checks'].append(row)
        report['source_hashes'][str(path.relative_to(root))]=sha256(path)
    for name in ['src/models.py','src/data.py','src/config.py','src/runner.py','src/client.py',
                 'src/attacks/classic_attack.py','src/attacks/inversion.py','src/attacks/metrics.py',
                 'privacy_main_benchmark/protocol.json','privacy_main_benchmark/target_samples.csv',
                 'privacy_main_benchmark/results/summaries/per_run_metrics.csv']:
        report['source_hashes'][name]=sha256(root/name)
    write_json(PACKAGE/'results/preflight.json',report)
    print('PASS: training, attack protocol, target list and payload accounting inspected.')
    for r in report['checks']: print(r)
    print('Report:',PACKAGE/'results/preflight.json')


if __name__=='__main__': main()
