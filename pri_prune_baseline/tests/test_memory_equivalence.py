"""Regression: memory optimization must preserve the original first-order updates."""
import copy
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.stateless import functional_call
from torch.utils.data import DataLoader, TensorDataset
from common import settings
from defense import base_mask, deterministic_share, new_client_state, privacy_loss, sample_share, train_client


def reference_updates(model, loader, cfg, opt, previous, base):
    """Original retained-graph computation, used only on a small CPU model."""
    shared = OrderedDict((n, nn.Parameter(p.detach().clone())) for n,p in model.named_parameters())
    private = OrderedDict((n, nn.Parameter(previous['private'][n].clone())) for n in shared)
    logits = OrderedDict((n, nn.Parameter(previous['logits'][n].clone())) for n in shared)
    weights = list(shared.values()) + list(private.values())
    weight_opt = torch.optim.SGD(weights, lr=.01, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    mask_opt = torch.optim.Adam(list(logits.values()), lr=opt['mask_learning_rate'])
    generator = torch.Generator().manual_seed(opt['mask_seed'] + cfg.seed*100003 + 101)
    total, examples = 0., 0
    for _ in range(cfg.local_epochs):
        for x,y in loader:
            masks = {n: sample_share(a, opt['gumbel_temperature'], generator) for n,a in logits.items()}
            effective = OrderedDict((n, base[n]*(masks[n]*shared[n] + (1-masks[n])*private[n])) for n in shared)
            loss = F.cross_entropy(functional_call(model, effective, (x,)), y)
            g = torch.autograd.grad(loss, tuple(effective.values()), retain_graph=True)
            pri = privacy_loss(logits, {n:grad*base[n] for n,grad in zip(effective,g)})
            penalty = sum(torch.sigmoid(a).sum() for a in logits.values())
            objective = opt['lambda_accuracy']*loss + opt['lambda_privacy']*pri + opt['lambda_nonsharing']*penalty
            weight_opt.zero_grad(set_to_none=True)
            mask_opt.zero_grad(set_to_none=True)
            objective.backward()
            for p in weights: p.grad.div_(opt['lambda_accuracy'])
            weight_opt.step()
            mask_opt.step()
            with torch.no_grad():
                for a in logits.values(): a.clamp_(-12,12)
            total += float(loss.detach()) * len(x)
            examples += len(x)
    final = {n:torch.where(deterministic_share(logits[n]), shared[n], private[n]).detach()*base[n] for n in shared}
    return {'private': final, 'logits': {n:a.detach() for n,a in logits.items()}}, total/examples


def test_memory_optimization_preserves_multibatch_optimizer_updates():
    torch.manual_seed(13)
    model = nn.Sequential(nn.Conv2d(3,4,3,padding=1), nn.GroupNorm(2,4), nn.ReLU(inplace=True),
                          nn.Flatten(), nn.Linear(4*8*8,3))
    dataset = TensorDataset(torch.randn(19,3,8,8), torch.randint(0,3,(19,)))
    loader = DataLoader(dataset,batch_size=4,shuffle=False)
    cfg = SimpleNamespace(local_epochs=3,momentum=.9,weight_decay=1e-4,seed=42)
    opt = settings()
    opt.update(initial_nonsharing_probability=.45,base_keep_ratio=.5)
    params = dict(model.named_parameters())
    previous = new_client_state(params,opt)
    base = base_mask(params,.5)
    expected, loss = reference_updates(copy.deepcopy(model),loader,cfg,opt,previous,base)
    result, actual = train_client(copy.deepcopy(model),loader,cfg,opt,previous,base,.01,0,1,torch.device('cpu'))
    for group in ['private','logits']:
        for name in expected[group]:
            torch.testing.assert_close(expected[group][name],actual[group][name],rtol=1e-5,atol=1e-7)
    assert abs(result['local_loss']-loss)<1e-6
