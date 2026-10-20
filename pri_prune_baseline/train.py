from __future__ import annotations

import argparse
import copy
import os
import platform
import time
from dataclasses import asdict

from common import PACKAGE, output_dir, partition_digest, project_root, read_json, settings, sha256, source_config, write_json


def main():
    parser = argparse.ArgumentParser(description='Train an independent PriPrune-Magnitude adaptation; never change existing DRPE outputs.')
    parser.add_argument('--project-root')
    parser.add_argument('--seed', type=int, choices=[42,43,44], required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--smoke', action='store_true', help='2 rounds, 1 batch/client; isolated and excluded from formal summaries.')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = project_root(args.project_root)
    import numpy as np
    import torch
    from defense import aggregate_sparse_weights, base_mask, new_client_state, train_client
    from src.data import build_federated_data, get_dataset_spec
    from src.models import build_model
    from src.runner import learning_rate_for_round
    from src.server import FederatedServer
    from src.utils import append_csv, resolve_device, set_global_seed

    cfg, config_path = source_config(root,args.seed)
    cfg.device = args.device
    out = output_dir(args.seed,args.smoke)
    cfg.output_dir = str(out)
    options = settings()
    if args.smoke:
        cfg.rounds, cfg.local_epochs, cfg.save_every = 2,1,1
    out.mkdir(parents=True,exist_ok=True)
    signature = {'training_config':asdict(cfg), 'defense':options, 'smoke':args.smoke,
                 'reference_sha256':sha256(config_path)}
    state_path = out/'resume.pt'
    if (out/'summary.json').exists():
        if args.resume:
            print(f'Already complete: {out}'); return
        raise FileExistsError('This run is already complete. Existing files are not overwritten.')
    if (out/'metrics.csv').exists() and not args.resume:
        raise FileExistsError('A partial run exists. Use --resume or choose a fresh package copy.')
    set_global_seed(cfg.seed,deterministic=cfg.deterministic)
    device = resolve_device(args.device)
    spec = get_dataset_spec(cfg.dataset)
    model = build_model(cfg.model_name,cfg.num_classes,cfg.norm,spec.image_size)
    initial = {n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    data = build_federated_data(cfg)
    if cfg.norm != 'group' or list(model.named_buffers()):
        raise ValueError('This delivered protocol uses GroupNorm with no running-stat buffers.')
    server = FederatedServer(model,cfg,device)
    selection = np.random.default_rng(cfg.seed+999)
    count = min(cfg.num_clients,max(1,int(round(cfg.num_clients*cfg.client_fraction))))
    client_states = {}
    start_round, best_round, best_val, cumulative = 1,0,float('-inf'),0
    partition = {'client_sizes':data.client_sizes, 'partition_seed':cfg.partition_seed,
                 'sha256_ordered_client_indices':partition_digest(data.client_indices)}

    if args.resume:
        if not state_path.exists():
            raise FileNotFoundError('No resume checkpoint. A new run requires omitting --resume.')
        state = torch.load(state_path,map_location='cpu')
        if state['signature'] != signature or state['partition'] != partition:
            raise ValueError('Resume configuration or data partition changed.')
        server.model.load_state_dict(state['global_state'])
        client_states = state['client_states']
        start_round = state['round']+1
        best_val,best_round,cumulative = state['best_val'],state['best_round'],state['cumulative']
        selection.bit_generator.state = state['selection_rng']
        for loader,rng in zip(data.client_loaders,state['loader_rng']):
            loader.generator.set_state(rng)
        torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(state['cuda_rng'])
        import random
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        # Interrupted rounds after the last persisted checkpoint are rerun;
        # discard only this package's corresponding incomplete metric rows.
        from common import read_csv,write_csv
        rows=[r for r in read_csv(out/'metrics.csv') if int(r['round']) < start_round]
        if rows: write_csv(out/'metrics.csv',rows)
        else: (out/'metrics.csv').unlink(missing_ok=True)
    else:
        write_json(out/'run_signature.json',signature)
        write_json(out/'partition_summary.json',partition)
        write_json(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,
                   cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                   reference_config=str(config_path),python_executable=os.sys.executable))

    def cpu_model():
        return {n:p.detach().cpu().clone() for n,p in server.model.state_dict().items()}

    def attack_checkpoint(round_index):
        cid=0
        if cid not in client_states:
            client_states[cid]=new_client_state(initial,options)
        torch.save({'global_state':cpu_model(),'client_state':client_states[cid],
                    'observed_client_id':cid,'round':round_index,'defense':options,
                    'smoke':args.smoke,'partition':partition,
                    'threat_model':'local-model-and-support oracle; gradient proxy'},
                   out/f'attack_checkpoint_round_{round_index:04d}.pt')

    for round_index in range(start_round,cfg.rounds+1):
        start=time.perf_counter()
        lr=learning_rate_for_round(cfg,round_index)
        selected=selection.choice(cfg.num_clients,size=count,replace=False).tolist()
        global_parameters={n:p.detach().cpu().clone() for n,p in server.model.named_parameters()}
        base=base_mask(global_parameters,options['base_keep_ratio'],options['prune_scope'])
        results=[]
        for cid in selected:
            previous=client_states.get(cid)
            if previous is None: previous=new_client_state(initial,options)
            local=copy.deepcopy(server.model).to(device)
            result,next_state=train_client(local,data.client_loaders[cid],cfg,options,previous,base,
                                           lr,cid,round_index,device,1 if args.smoke else None)
            results.append(result); client_states[cid]=next_state
            del local
        aggregated=aggregate_sparse_weights(global_parameters,results,cfg.server_lr)
        with torch.no_grad():
            for name,p in server.model.named_parameters(): p.copy_(aggregated[name].to(device))
        uplink=sum(r['uplink_bytes'] for r in results)
        cumulative+=uplink
        raw_bytes=server.parameter_bytes()*count
        val=server.evaluate(data.val_loader)
        if val.accuracy>best_val:
            best_val,best_round=val.accuracy,round_index
            torch.save(cpu_model(),out/'best_model.pt')
        append_csv({'round':round_index,'learning_rate':lr,'selected_clients':'|'.join(map(str,selected)),
                    'mean_client_loss':sum(r['local_loss'] for r in results)/count,
                    'val_loss':val.loss,'val_accuracy':val.accuracy,'round_uplink_mib':uplink/2**20,
                    'round_downlink_mib':raw_bytes/2**20,'uplink_ratio_vs_raw_parameters':uplink/raw_bytes,
                    'mean_nonsharing_probability':sum(r['alpha_mean'] for r in results)/count,
                    'cumulative_uplink_mib':cumulative/2**20,'round_seconds':time.perf_counter()-start},out/'metrics.csv')
        print(f'Round {round_index:03d}: val={val.accuracy:.2f}%, uplink={uplink/2**20:.3f} MiB, ratio={uplink/raw_bytes:.4f}',flush=True)
        if round_index%cfg.save_every==0 or round_index==cfg.rounds:
            torch.save(cpu_model(),out/f'checkpoint_round_{round_index:04d}.pt')
            if args.smoke or round_index==20: attack_checkpoint(round_index)
            import random
            checkpoint=dict(signature=signature,partition=partition,round=round_index,
                            global_state=cpu_model(),client_states=client_states,best_val=best_val,
                            best_round=best_round,cumulative=cumulative,selection_rng=selection.bit_generator.state,
                            loader_rng=[l.generator.get_state() for l in data.client_loaders],
                            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                            python_rng=random.getstate(),numpy_rng=np.random.get_state())
            temporary=out/'resume.pt.tmp'
            torch.save(checkpoint,temporary); temporary.replace(state_path)
    torch.save(cpu_model(),out/'last_model.pt')
    server.model.load_state_dict(torch.load(out/'best_model.pt',map_location=device))
    test=server.evaluate(data.test_loader)
    from common import read_csv
    rows=read_csv(out/'metrics.csv')
    write_json(out/'summary.json',{'method':options['method_label'],'seed':cfg.seed,'smoke':args.smoke,
               'rounds':cfg.rounds,'best_round':best_round,'best_validation_accuracy':best_val,
               'test_accuracy':test.accuracy,'test_loss':test.loss,'cumulative_uplink_mib':cumulative/2**20,
               'mean_round_uplink_mib':sum(float(r['round_uplink_mib']) for r in rows)/len(rows),
               'mean_uplink_ratio':sum(float(r['uplink_ratio_vs_raw_parameters']) for r in rows)/len(rows),
               'metric_scope':'global pruned model evaluated on the same global validation/test sets',
               'reference_config':str(config_path),'source_config_sha256':sha256(config_path)})
    print(f'Complete: test={test.accuracy:.2f}% at validation-selected round {best_round}. Results: {out}')


if __name__=='__main__':
    main()
