from __future__ import annotations

import argparse
import time
from contextlib import contextmanager

from common import PACKAGE, output_dir, project_root, read_json, sha256, validate_protocol, write_json


@contextmanager
def support_aware_adapter(classic, mask):
    """Temporarily adapt the existing optimizer in memory, never its source file."""
    original_candidate=classic._candidate_observation
    original_infer=classic.infer_idlg_label
    supports={n:m.reshape(-1).nonzero().reshape(-1) for n,m in mask.items()}
    supports={n:i for n,i in supports.items() if i.numel()}
    if not supports: raise ValueError('Empty upload: a gradient inversion comparison is undefined.')

    def candidate(raw, cfg, settings, shapes):
        return {n:raw[n].reshape(-1)[i.to(raw[n].device)] for n,i in supports.items()}

    def infer(observation, num_classes):
        dense={}
        for n,value in observation.items():
            tensor=value.new_zeros(mask[n].numel())
            tensor[supports[n].to(value.device)]=value
            dense[n]=tensor.reshape(mask[n].shape)
        # Uses only the transmitted support/values; no private logits or label.
        try:
            label,source=original_infer(dense,num_classes)
            return label,source+':public_zero_filled_support'
        except ValueError:
            return 0,'no_classifier_observation:deterministic_blind_class0'

    classic._candidate_observation=candidate
    classic.infer_idlg_label=infer
    try: yield supports
    finally:
        classic._candidate_observation=original_candidate
        classic.infer_idlg_label=original_infer


def main():
    p=argparse.ArgumentParser(description='Support-aware DLG/iDLG under the existing single-sample gradient proxy protocol.')
    p.add_argument('--project-root'); p.add_argument('--seed',type=int,choices=[42,43,44],required=True)
    p.add_argument('--attack',choices=['dlg','idlg'],required=True)
    p.add_argument('--target',choices=[f'T{i:02d}' for i in range(10)],required=True)
    p.add_argument('--restart',type=int,choices=[0,1],required=True)
    p.add_argument('--device',default='auto');p.add_argument('--smoke',action='store_true')
    p.add_argument('--skip-lpips',action='store_true',help='Only permitted in smoke mode.')
    args=p.parse_args();root=project_root(args.project_root)
    if args.skip_lpips and not args.smoke: raise ValueError('Formal comparisons require saved-PNG LPIPS.')
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.utils import save_image
    from defense import proxy_model_state
    from src.attacks import classic_attack as classic
    from src.attacks.classic_config import ClassicAttackConfig
    from src.attacks.inversion import compute_named_gradients
    from src.attacks.metrics import psnr_01,ssim_01
    from src.data import build_attack_dataset,get_dataset_spec
    from src.models import build_model
    from src.utils import resolve_device,set_global_seed

    protocol,targets=validate_protocol(root)
    target=next(r for r in targets if r['target_id']==args.target)
    index=int(target['dataset_index'])
    base_seed=protocol['main']['attack_base_seeds'][args.restart]
    iterations=3 if args.smoke else protocol['iterations']
    round_index=1 if args.smoke else protocol['checkpoint_round']
    checkpoint=output_dir(args.seed,args.smoke)/f'attack_checkpoint_round_{round_index:04d}.pt'
    out=PACKAGE/'results'/('smoke_attacks' if args.smoke else 'attacks')/f'seed{args.seed}'/args.target/f'restart{args.restart:02d}'/args.attack
    if (out/'run.json').exists():
        previous=read_json(out/'run.json')
        if (previous.get('status')=='success' and previous.get('checkpoint_sha256')==sha256(checkpoint)
            and previous.get('defense')==read_json(PACKAGE/'config.json')):
            print('SKIP complete:',out);return
        raise ValueError('Completed attack uses a different checkpoint. Use a fresh result folder.')
    out.mkdir(parents=True,exist_ok=True)
    set_global_seed(base_seed,deterministic=True)
    device=resolve_device(args.device)
    saved=torch.load(checkpoint,map_location='cpu')
    if saved['round']!=round_index or saved['smoke']!=args.smoke:
        raise ValueError('Wrong attack checkpoint stage or smoke/formal mode.')
    if saved['defense'] != read_json(PACKAGE/'config.json'):
        raise ValueError('Defense config changed after training; do not mix experiment variants.')
    model=build_model('resnet18',10,'group',32)
    params={n:saved['global_state'][n] for n,_ in model.named_parameters()}
    effective,mask=proxy_model_state(saved['global_state'],saved['client_state'],params,saved['defense'])
    model.load_state_dict(effective);model.to(device).eval()
    dataset=build_attack_dataset('cifar10',str(root/'data'),'train',download=False)
    image,label=dataset[index]
    if label!=int(target['class_id']):raise ValueError('Target label differs from the existing target list.')
    image=image.unsqueeze(0).to(device)
    y=torch.tensor([label],device=device)
    spec=get_dataset_spec('cifar10')
    raw=compute_named_gradients(model,image,y,'all',False,spec.mean,spec.std)
    cfg=ClassicAttackConfig(seed=base_seed+index*1009,dataset='cifar10',data_root=str(root/'data'),
          dataset_download=False,output_dir=str(out),device=args.device,model_name='resnet18',num_classes=10,
          norm='group',sample_indices=[index],attack_name=args.attack,label_mode='optimize',
          observation_mode='raw',parameter_scope='all',iterations=iterations,
          attack_lr=protocol['attack_lr'],label_lr=protocol['label_lr'],optimizer=protocol['optimizer'],
          matching_loss=protocol['matching_loss'],tv_weight=protocol['tv_weight'],
          label_entropy_weight=protocol['label_entropy_weight'],init='uniform',image_parameterization='sigmoid',
          label_init='zeros',print_every=25,save_every=25,compute_optional_lpips=False)
    # Fail before expensive inversion if a formal LPIPS dependency is missing.
    lpips_model=None
    if not args.skip_lpips:
        try:
            import lpips
            lpips_model=lpips.LPIPS(net='alex').to(device).eval()
        except Exception as e:
            raise RuntimeError('LPIPS unavailable. Install lpips in the existing experiment environment; do not change torch.') from e
    save_image(image.cpu(),out/'target.png')
    start=time.perf_counter()
    with support_aware_adapter(classic,mask) as supports:
        observed={n:raw[n].reshape(-1)[i.to(device)].detach().cpu() for n,i in supports.items()}
        result=classic._run_single_attack(model,observed,label,cfg,None,
                     {n:m.shape for n,m in mask.items()},device,out/'progress.csv')
    elapsed=time.perf_counter()-start
    best=result.best_image.to(device)
    save_image(best.cpu(),out/f'reconstruction_{args.attack}_best.png')
    lpips_value=None
    if lpips_model is not None:
        def png_tensor(path):
            array=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255
            return torch.from_numpy(array).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            lpips_value=float(lpips_model(png_tensor(out/'target.png')*2-1,
                            png_tensor(out/f'reconstruction_{args.attack}_best.png')*2-1).mean())
    write_json(out/'run.json',dict(status='success',method='pri_prune_magnitude',model_seed=args.seed,
         target_id=args.target,sample_index=index,class_id=label,attack=args.attack,restart=args.restart,
         attack_base_seed=base_seed,checkpoint_round=round_index,transmission_round=round_index+1,
         iterations=iterations,smoke=args.smoke,checkpoint_sha256=sha256(checkpoint),
         psnr=psnr_01(image,best),ssim=ssim_01(image,best),lpips=lpips_value,
         lpips_source='saved_png_alex' if lpips_value is not None else 'smoke_skipped',
         predicted_label=result.predicted_label,label_correct=int(result.predicted_label==label),
         label_source=result.label_source,best_iteration=result.best_iteration,attack_seconds=elapsed,
         observation_numel=sum(v.numel() for v in observed.values()),
         threat_model='Complete effective local model and transmitted support disclosed to attacker; stronger oracle proxy than native PriPrune.',
         extra_information='No true label unless inferred from public observation. No hidden alpha disclosed.',
         model_state_sha256=sha256(checkpoint),defense=saved['defense']))
    print(f'Saved {out}: PSNR={psnr_01(image,best):.4f}, SSIM={ssim_01(image,best):.4f}, label_correct={int(result.predicted_label==label)}')


if __name__=='__main__': main()
