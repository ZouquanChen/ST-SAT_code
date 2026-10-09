import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SpikeModule(nn.Module):

    def __init__(self):
        super().__init__()
        self._spiking = False

    def set_spike_state(self, use_spike=True):
        self._spiking = use_spike

def spike_activation(x, ste=False, temp=1.0, grad_scale=1.0):
    out_s = torch.gt(x, 0.5)
    if not torch.is_grad_enabled():
        return out_s.float()
    if ste:
        out_bp = torch.clamp(x, 0, 1)
    else:
        out_bp = torch.clamp(x, 0, 1)
        tanh_half = torch.tanh(temp * 0.5) if isinstance(temp, torch.Tensor) else np.tanh(temp * 0.5)
        out_bp = (torch.tanh(temp * (out_bp - 0.5)) + tanh_half) / (2 * tanh_half + 1e-08)
    if grad_scale is None:
        grad_scale = 1.0
    out_bp = out_bp * grad_scale
    return (out_s.float() - out_bp).detach() + out_bp

def mem_update(bn, x_in, mem, V_th, decay, grad_scale=1.0, temp=1.0, use_nam=True):
    mem = mem * decay + x_in
    if use_nam:
        mem_spatial = bn(mem)
    elif bn.training and mem.numel() // mem.shape[1] == 1:
        mem_spatial = F.batch_norm(mem, bn.running_mean, bn.running_var, bn.weight, bn.bias, training=False, momentum=0.0, eps=bn.eps)
    else:
        mem_spatial = bn(mem)
    spike = spike_activation(mem_spatial / V_th, temp=temp, grad_scale=grad_scale)
    mem = mem * (1 - spike)
    return (mem, spike)

class NAM(nn.Module):

    def __init__(self, channels, t=16):
        super(NAM, self).__init__()
        self.channels = channels
        self.my_NAM = nn.BatchNorm2d(self.channels, affine=True)
        nn.init.constant_(self.my_NAM.weight, 1)

    def forward(self, x):
        residual = x
        bn = self.my_NAM
        if bn.training and x.numel() // x.shape[1] == 1:
            x = F.batch_norm(x, bn.running_mean, bn.running_var, bn.weight, bn.bias, training=False, momentum=0.0, eps=bn.eps)
        else:
            x = bn(x)
        weight = self.my_NAM.weight
        weight_bn = weight.abs() / (weight.abs().sum() + 1e-08)
        if not torch.is_grad_enabled():
            return torch.sigmoid(x * weight_bn.view(1, -1, 1, 1)) * residual
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual
        return x

class LIFAct(SpikeModule):

    def __init__(self, step, channel, decay=0.25, V_th=1.0, temp=3.0, grad_scale=0.1, learnable_decay=True, use_ltf=True, use_nam=True):
        super(LIFAct, self).__init__()
        self.step = step
        self.channel = channel
        self.V_th = V_th
        self.decay_init = decay
        self.temp = temp
        self.grad_scale = grad_scale
        self.learnable_decay = learnable_decay
        self.use_ltf = use_ltf
        self.use_nam = use_nam
        if use_ltf:
            self.temporal = nn.Parameter(torch.zeros(self.step))
        else:
            self.register_buffer('temporal', torch.zeros(self.step))
        if learnable_decay:
            tau_init = math.log(decay / (1 - decay + 1e-08))
            self.tau = nn.Parameter(torch.full((channel,), tau_init))
        else:
            self.register_buffer('tau', None)
            self.decay = decay
        if use_nam:
            self.bn = NAM(channel)
        else:
            self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        if self._spiking is not True:
            return F.relu(x)
        if self.grad_scale is None:
            self.grad_scale = 1 / math.sqrt(x[0].numel() * self.step)
        u = torch.zeros_like(x[0])
        out = []
        if self.learnable_decay:
            decay = torch.sigmoid(self.tau).view(1, -1, 1, 1)
        else:
            decay = self.decay
        temp = self.temp
        self.weight = torch.exp(self.temporal)
        for i in range(self.step):
            V_th_dynamic = self.V_th / self.weight[i]
            (u, out_i) = mem_update(bn=self.bn, x_in=x[i], mem=u, V_th=V_th_dynamic, grad_scale=self.grad_scale, decay=decay, temp=temp, use_nam=self.use_nam)
            out.append(out_i)
        out = torch.stack(out)
        return out

    def get_decay(self):
        if self.learnable_decay:
            return torch.sigmoid(self.tau).detach()
        else:
            return torch.tensor(self.decay)

    def get_decay_stats(self):
        decay = self.get_decay()
        if self.learnable_decay:
            return {'mean': decay.mean().item(), 'std': decay.std().item(), 'min': decay.min().item(), 'max': decay.max().item()}
        else:
            return {'mean': decay.item(), 'std': 0.0, 'min': decay.item(), 'max': decay.item()}

    def get_temporal_weights(self):
        return torch.exp(self.temporal).detach().cpu().numpy()

    def get_dynamic_thresholds(self):
        weights = torch.exp(self.temporal)
        thresholds = self.V_th / weights
        return thresholds.detach().cpu().numpy()

def validate_st_snn_model(model, input_shape, device=None):
    lif_modules = [module for module in model.modules() if isinstance(module, LIFAct)]
    if not lif_modules:
        raise RuntimeError('ST-SAT model contains no LIFAct modules.')
    inactive = [module.__class__.__name__ for module in lif_modules if module._spiking is not True]
    if inactive:
        raise RuntimeError(f'ST-SAT has {len(inactive)} inactive custom LIF modules; call set_spike_state(True) during model construction.')
    model_device = device
    if model_device is None:
        model_device = next(model.parameters()).device
    was_training = model.training
    hooks = []
    outputs = []

    def collect_output(module, inputs, output):
        if isinstance(output, torch.Tensor):
            outputs.append(output.detach())
    for module in lif_modules:
        hooks.append(module.register_forward_hook(collect_output))
    try:
        model.eval()
        numel = int(np.prod(input_shape))
        sample = torch.linspace(-1.0, 1.0, steps=numel, device=model_device).reshape(input_shape)
        with torch.no_grad():
            model(sample)
        if len(outputs) != len(lif_modules):
            raise RuntimeError(f'ST-SAT forward check observed {len(outputs)} LIF outputs, but the model contains {len(lif_modules)} LIF modules.')
        non_binary = [output for output in outputs if not torch.all((output == 0) | (output == 1)).item()]
        if non_binary:
            raise RuntimeError('ST-SAT forward check failed: at least one LIF output is not binary; a LIF layer may still be using ReLU mode.')
        return {'num_lif_modules': len(lif_modules), 'num_spiking_modules': len(lif_modules), 'num_checked_outputs': len(outputs), 'binary_outputs': True}
    finally:
        for hook in hooks:
            hook.remove()
        try:
            from spikingjelly.activation_based import functional
            functional.reset_net(model)
        except Exception:
            pass
        model.train(was_training)
