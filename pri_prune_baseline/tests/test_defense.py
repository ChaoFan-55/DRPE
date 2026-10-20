import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import copy
from types import SimpleNamespace
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from defense import (aggregate_sparse_weights,base_mask,deterministic_share,new_client_state,
                     payload_bytes,privacy_loss,proxy_model_state,sample_share,sparse_payload,train_client)
from common import settings


def test_gumbel_sampling_and_st_gradient():
    a=torch.full((10000,),-2.197224577,requires_grad=True)
    m=sample_share(a,1.0,torch.Generator().manual_seed(1))
    assert abs(float(m.detach().mean())-.9)<.025
    assert set(m.detach().round().tolist())=={0,1}
    (m.square().sum()).backward()
    assert torch.isfinite(a.grad).all() and torch.count_nonzero(a.grad)>0


def test_privacy_loss_discourages_sharing_high_gradient():
    a=torch.zeros(2,requires_grad=True)
    loss=privacy_loss({'weight':a},{'weight':torch.tensor([9.,1.])})
    loss.backward()
    assert a.grad[0]<a.grad[1]<0


def test_byte_count_and_zero_fill_aggregation():
    params={'weight':torch.tensor([10.,20.])}
    first=sparse_payload({'weight':torch.tensor([2.,100.])},{'weight':torch.tensor([1,0],dtype=torch.bool)})
    second=sparse_payload({'weight':torch.tensor([100.,4.])},{'weight':torch.tensor([0,1],dtype=torch.bool)})
    assert payload_bytes(first)==12  # float32 value, int32 index, uint32 count
    out=aggregate_sparse_weights(params,[{'num_samples':1,'payload':first},{'num_samples':3,'payload':second}])
    torch.testing.assert_close(out['weight'],torch.tensor([.5,3.]))


def test_base_pruning_does_not_prune_bias_or_normalization():
    ps={'conv.weight':torch.arange(12.).reshape(3,4),'norm.weight':torch.ones(3),'head.bias':torch.zeros(3)}
    bm=base_mask(ps,.25)
    assert bm['conv.weight'].sum()==3 and bm['norm.weight'].all() and bm['head.bias'].all()


def test_client_training_preserves_server_and_updates_mask_and_private_state():
    torch.manual_seed(5)
    model=nn.Sequential(nn.Flatten(),nn.Linear(12,3))
    params=dict(model.named_parameters())
    original=copy.deepcopy(model.state_dict())
    opt=settings();opt.update(base_keep_ratio=1.0,lambda_accuracy=1.0,initial_nonsharing_probability=.55)
    previous=new_client_state(params,opt)
    loader=DataLoader(TensorDataset(torch.randn(4,3,2,2),torch.tensor([0,1,2,1])),batch_size=2)
    cfg=SimpleNamespace(local_epochs=1,momentum=.9,weight_decay=1e-4,seed=42)
    result,state=train_client(model,loader,cfg,opt,previous,base_mask(params,1),.01,0,1,torch.device('cpu'))
    assert result['uplink_bytes']==payload_bytes(result['payload'])
    assert result['num_samples']==4
    assert any(not torch.equal(state['logits'][n],previous['logits'][n]) for n in params)
    assert any(not torch.equal(state['private'][n],previous['private'][n]) for n in params)
    for n,p in model.state_dict().items():torch.testing.assert_close(p,original[n])


def test_proxy_composition_retains_private_weights_and_public_support():
    ps={'weight':torch.tensor([1.,2.])}
    client={'private':{'weight':torch.tensor([8.,9.])},'logits':{'weight':torch.tensor([-1.,1.])}}
    opt={'base_keep_ratio':1,'prune_scope':'all'}
    state,mask=proxy_model_state(ps,client,ps,opt)
    torch.testing.assert_close(state['weight'],torch.tensor([1.,9.]))
    assert mask['weight'].tolist()==[True,False]
    assert deterministic_share(torch.tensor([0.])).item() is False


def test_attack_adapter_gathers_public_indices_and_does_not_get_true_label():
    import os
    import pytest
    root=os.environ.get('DRPE_TEST_ROOT')
    if not root: pytest.skip('Set DRPE_TEST_ROOT to validate the original attack adapter.')
    sys.path.insert(0,root)
    from src.attacks import classic_attack as classic
    from attack import support_aware_adapter
    original=classic._candidate_observation
    mask={'head.bias':torch.tensor([1,0,0,0,0,1,0,0,0,0],dtype=torch.bool)}
    raw={'head.bias':torch.tensor([.2,.1,.1,.1,.1,-.9,.1,.1,.1,.1],requires_grad=True)}
    with support_aware_adapter(classic,mask):
        observation=classic._candidate_observation(raw,None,None,None)
        assert observation['head.bias'].numel()==2
        label,_=classic.infer_idlg_label(observation,10)
        assert label==5
        observation['head.bias'].sum().backward()
        torch.testing.assert_close(raw['head.bias'].grad,mask['head.bias'].float())
    assert classic._candidate_observation is original


