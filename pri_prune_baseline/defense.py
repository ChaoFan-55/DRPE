"""Independent implementation of PriPrune's adaptive pseudo-pruning mechanism.

See METHOD_NOTES.md for the explicit benchmark adaptations. This is not author
code and not a reproduction of their reported accuracy or SGI attack results.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.stateless import functional_call


def base_mask(parameters, keep_ratio, scope='matrix_weights'):
    """Tensor-wise magnitude pruning; normalization and bias stay unpruned."""
    if not 0 < keep_ratio <= 1:
        raise ValueError('base_keep_ratio must be in (0,1].')
    result = OrderedDict()
    for name, value in parameters.items():
        mask = torch.ones_like(value, dtype=torch.bool)
        if scope == 'all' or (value.ndim >= 2 and name.endswith('weight')):
            n = value.numel()
            k = max(1, int(math.ceil(n * keep_ratio)))
            mask = torch.zeros(n, device=value.device, dtype=torch.bool)
            mask[torch.topk(value.detach().abs().reshape(-1), k, sorted=False).indices] = True
            mask = mask.reshape_as(value)
        result[name] = mask
    return result


def sample_share(logits, temperature, generator):
    """logits parameterize alpha=P(not sharing). Independent random stream."""
    if temperature <= 0:
        raise ValueError('Temperature must be positive.')
    u0 = torch.rand(logits.shape, generator=generator, device=logits.device).clamp(1e-6, 1-1e-6)
    u1 = torch.rand(logits.shape, generator=generator, device=logits.device).clamp(1e-6, 1-1e-6)
    # Difference of two Gumbel variables implements the two-category softmax.
    noise = -torch.log(-torch.log(u0)) + torch.log(-torch.log(u1))
    share_soft = torch.sigmoid((-logits + noise) / temperature)
    share_hard = (share_soft >= 0.5).to(share_soft.dtype)
    return share_hard + share_soft - share_soft.detach()


def deterministic_share(logits):
    # argmax([alpha, 1-alpha]); a tie is non-sharing.
    return logits.detach() < 0


def privacy_loss(logits, gradients):
    """Paper's layer-size weighted, gradient-magnitude normalized -log(alpha)."""
    total = sum(g.numel() for g in gradients.values())
    first = next(iter(logits.values()))
    value = first.new_zeros(())
    for name, g in gradients.items():
        score = g.detach().abs()
        score = score / score.sum().clamp_min(1e-12)
        value = value - (g.numel() / total) * (score * F.logsigmoid(logits[name])).sum()
    return value


def sparse_payload(parameters, mask):
    """Actual representation: FP32 values + int32 flattened indices + counts."""
    payload = OrderedDict()
    for name, p in parameters.items():
        indices = torch.where(mask[name].reshape(-1))[0]
        if p.numel() >= 2**31:
            raise ValueError('Tensor indices do not fit int32.')
        payload[name] = (indices.to(device='cpu', dtype=torch.int32),
                         p.detach().reshape(-1)[indices].to(device='cpu', dtype=torch.float32))
    return payload


def payload_bytes(payload):
    # A 4-byte count per tensor separates variable-size payloads. Shapes and
    # names are agreed in advance; no secret masks or free indices are assumed.
    return sum(i.numel()*i.element_size() + v.numel()*v.element_size() + 4
               for i, v in payload.values())


def aggregate_sparse_weights(global_parameters, results, server_lr=1.0):
    """Sample-weighted pruned local models; absent weights are zero.

    This retains the paper's pruned-model aggregation instead of silently
    introducing coordinate-wise renormalization or error feedback.
    """
    total = sum(r['num_samples'] for r in results)
    if total <= 0:
        raise ValueError('No client samples.')
    out = {name: torch.zeros_like(p, device='cpu') for name, p in global_parameters.items()}
    for r in results:
        for name, (indices, values) in r['payload'].items():
            out[name].reshape(-1).index_add_(0, indices.long(), values * (r['num_samples']/total))
    return {name: p.detach().cpu() + server_lr*(out[name] - p.detach().cpu())
            for name, p in global_parameters.items()}


def proxy_model_state(global_state, client_state, parameters, options):
    """Round-21 initialization: shared global weights + retained private ones."""
    bm = base_mask(parameters, options['base_keep_ratio'], options['prune_scope'])
    effective, upload_mask = {}, {}
    for name, p in parameters.items():
        share = deterministic_share(client_state['logits'][name])
        effective[name] = torch.where(share, p.detach().cpu(), client_state['private'][name]) * bm[name].cpu()
        upload_mask[name] = share.cpu() & bm[name].cpu()
    state = {name: value.detach().cpu().clone() for name, value in global_state.items()}
    state.update(effective)
    return state, upload_mask


def new_client_state(parameters, options):
    a = options['initial_nonsharing_probability']
    if not 0 < a < 1:
        raise ValueError('Initial alpha must be in (0,1).')
    logit = math.log(a/(1-a))
    return {'private': {n:p.detach().cpu().clone() for n,p in parameters.items()},
            'logits': {n:torch.full_like(p.detach().cpu(), logit) for n,p in parameters.items()}}


def train_client(model, loader, cfg, options, previous, base, lr, client_id, round_index,
                 device, max_batches=None):
    """Hard-forward ST-Gumbel composition with trainable shared/private branches.

    Two branches implement the global/private composition without overwriting
    nn.Parameter.data, which would sever the mask's accuracy-loss gradients.
    Optimizers reset per client participation; alpha and private weights persist.
    """
    model.train()
    shared = OrderedDict((n, nn.Parameter(p.detach().to(device).clone())) for n,p in model.named_parameters())
    private = OrderedDict((n, nn.Parameter(previous['private'][n].to(device).clone())) for n in shared)
    logits = OrderedDict((n, nn.Parameter(previous['logits'][n].to(device).clone())) for n in shared)
    bm = {n:m.to(device) for n,m in base.items()}
    weight_opt = torch.optim.SGD(list(shared.values()) + list(private.values()), lr=lr,
                                 momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    mask_opt = torch.optim.Adam(list(logits.values()), lr=options['mask_learning_rate'])
    generator = torch.Generator(device=device).manual_seed(options['mask_seed'] + cfg.seed*100003 + round_index*101 + client_id)
    loss_total, examples, batches = 0.0, 0, 0
    for epoch_index in range(cfg.local_epochs):
        for x, y in loader:
            if max_batches is not None and batches >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            # Release the previous batch's gradients before constructing a new
            # graph. Optimizer moments remain unchanged.
            weight_opt.zero_grad(set_to_none=True)
            mask_opt.zero_grad(set_to_none=True)
            masks = {n:sample_share(a, options['gumbel_temperature'], generator) for n,a in logits.items()}
            effective = OrderedDict((n, bm[n]*(masks[n]*shared[n] + (1-masks[n])*private[n])) for n in shared)
            # The privacy score already treats g as constant. Compute g through
            # detached effective-weight leaves, releasing the convolution graph
            # immediately, then apply the same accuracy gradient through the ST
            # composition using the chain rule. This avoids retaining the whole
            # network graph for a second convolution backward on a 4-GB GPU.
            leaves = OrderedDict((n, p.detach().requires_grad_(True)) for n,p in effective.items())
            loss_acc = F.cross_entropy(functional_call(model, leaves, (x,)), y)
            g = torch.autograd.grad(loss_acc, tuple(leaves.values()), retain_graph=False, create_graph=False)
            accuracy_value = loss_acc.detach()
            accuracy_surrogate = sum((p * gradient.detach()).sum() for p,gradient in zip(effective.values(),g))
            # Gradients of the base-pruned parameterization include the base
            # mask. Already removed coordinates must not get privacy weight.
            loss_pri = privacy_loss(logits, {n:gradient*bm[n] for n,gradient in zip(effective,g)})
            penalty = sum(torch.sigmoid(a).sum() for a in logits.values())
            objective = options['lambda_accuracy']*accuracy_surrogate + options['lambda_privacy']*loss_pri + options['lambda_nonsharing']*penalty
            if not torch.isfinite(accuracy_value) or not torch.isfinite(objective):
                raise FloatingPointError('Non-finite PriPrune objective. Stop instead of producing a comparison result.')
            objective.backward()
            # Lambda_acc changes the mask trade-off, but must not silently make
            # the common model learning rate five times larger than DRPE's.
            for p in list(shared.values()) + list(private.values()):
                if p.grad is not None:
                    p.grad.div_(options['lambda_accuracy'])
            weight_opt.step()
            mask_opt.step()
            with torch.no_grad():
                for a in logits.values():
                    a.clamp_(-12,12)
            loss_total += float(accuracy_value) * x.size(0)
            examples += x.size(0)
            batches += 1
            # Do not keep graph references and full parameter-sized tensors
            # alive while the next batch's forward pass is being evaluated.
            del masks, effective, leaves, loss_acc, g, accuracy_surrogate
            del accuracy_value, loss_pri, penalty, objective, x, y
        print(f'  Round {round_index:03d} client {client_id}: '
              f'epoch {epoch_index+1}/{cfg.local_epochs}, batches={batches}, '
              f'loss={loss_total/examples:.4f}' if examples else '  Empty client epoch', flush=True)
        if max_batches is not None and batches >= max_batches:
            break
    if not examples:
        raise ValueError('Empty local training loader.')
    final = {n:torch.where(deterministic_share(logits[n]), shared[n], private[n]).detach().cpu()*base[n].cpu() for n in shared}
    mask = {n:deterministic_share(logits[n]).cpu() & base[n].cpu() for n in shared}
    upload = sparse_payload(final, mask)
    next_state = {'private': final, 'logits': {n:a.detach().cpu().clone() for n,a in logits.items()}}
    return {'client_id': client_id, 'num_samples': len(loader.dataset), 'payload': upload,
            'local_loss': loss_total/examples, 'uplink_bytes': payload_bytes(upload),
            'shared_count': sum(int(m.sum()) for m in mask.values()),
            'alpha_mean': sum(float(torch.sigmoid(a.detach()).sum()) for a in logits.values())/sum(a.numel() for a in logits.values())}, next_state
