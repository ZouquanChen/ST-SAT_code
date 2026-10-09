import re
import torch
import torch.nn as nn
from utils import INPUT_SIZE
from models.st_lif import LIFAct
vgg_cfg = {'VGG5': [64, 'P', 128, 128, 'P'], 'VGG9': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P'], 'VGG11': [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'], 'VGG13': [64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'], 'VGG16': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'], 'VGG19': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P']}

class VGG_STLIF(nn.Module):

    def __init__(self, args):
        super(VGG_STLIF, self).__init__()
        (C, H, W) = INPUT_SIZE[args.dataset]
        self.bias_flag = False
        self.snn = True
        self.T = args.T
        self.decay = getattr(args, 'st_decay', 0.25)
        self.V_th = getattr(args, 'st_v_th', 1.0)
        self.temp = getattr(args, 'st_temp', 3.0)
        self.grad_scale = getattr(args, 'st_grad_scale', 0.1)
        self.learnable_decay = getattr(args, 'st_learnable_decay', True)
        self.use_ltf = getattr(args, 'st_use_ltf', True)
        self.use_nam = getattr(args, 'st_use_nam', True)
        pattern = re.compile('\\d+')
        layer_num = int(pattern.findall(args.model)[0])
        assert int(layer_num) in [5, 9, 11, 13, 16, 19], f'VGG layer {layer_num} not supported!'
        conv = []
        in_channel = C
        for x in vgg_cfg[f'VGG{layer_num}']:
            if x == 'P':
                conv.append(nn.MaxPool2d(kernel_size=2))
                H //= 2
                W //= 2
            else:
                out_channel = x
                conv.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=self.bias_flag))
                conv.append(nn.BatchNorm2d(out_channel))
                conv.append(LIFAct(step=args.T, channel=out_channel, decay=self.decay, V_th=self.V_th, temp=self.temp, grad_scale=self.grad_scale, learnable_decay=self.learnable_decay, use_ltf=self.use_ltf, use_nam=self.use_nam))
                in_channel = out_channel
        self.features = nn.Sequential(*conv)
        if int(layer_num) in [5, 9]:
            fc_hidden = 1024
        else:
            fc_hidden = 4096
        fc_layers = []
        fc_layers.append(nn.Flatten())
        fc_layers.append(nn.Linear(in_channel * H * W, fc_hidden, bias=self.bias_flag))
        fc_layers.append(LIFAct(step=args.T, channel=fc_hidden, decay=self.decay, V_th=self.V_th, temp=self.temp, grad_scale=self.grad_scale, learnable_decay=self.learnable_decay, use_ltf=self.use_ltf, use_nam=self.use_nam))
        fc_layers.append(nn.Linear(fc_hidden, args.num_classes, bias=self.bias_flag))
        self.fc = nn.Sequential(*fc_layers)
        self.set_spike_state(True)

    def set_spike_state(self, use_spike=True):
        for module in self.modules():
            if isinstance(module, LIFAct):
                module.set_spike_state(use_spike)

    def forward(self, x):
        (T, B, C, H, W) = x.shape
        out = self._forward_features_snn(x)
        out = self._forward_fc_snn(out)
        return out.mean(0)

    def _forward_features_snn(self, x):
        (T, B, C, H, W) = x.shape
        for layer in self.features:
            if isinstance(layer, LIFAct):
                x = layer(x)
            elif isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                out = []
                for t in range(T):
                    out.append(layer(x[t]))
                x = torch.stack(out, dim=0)
            elif isinstance(layer, nn.MaxPool2d):
                out = []
                for t in range(T):
                    out.append(layer(x[t]))
                x = torch.stack(out, dim=0)
        return x

    def _forward_fc_snn(self, x):
        T = x.shape[0]
        for layer in self.fc:
            if isinstance(layer, LIFAct):
                if len(x.shape) == 3:
                    x = x.unsqueeze(-1).unsqueeze(-1)
                x = layer(x)
                x = x.squeeze(-1).squeeze(-1)
            elif isinstance(layer, nn.Flatten):
                out = []
                for t in range(T):
                    out.append(layer(x[t]))
                x = torch.stack(out, dim=0)
            elif isinstance(layer, nn.Linear):
                out = []
                for t in range(T):
                    out.append(layer(x[t]))
                x = torch.stack(out, dim=0)
        return x

    def forward_with_features(self, x):
        (T, B, C, H, W) = x.shape
        feat = self._forward_features_snn(x)
        (out, feature_rates) = self._forward_fc_snn_with_features(feat)
        return (out.mean(0), feature_rates)

    def _forward_fc_snn_with_features(self, x):
        T = x.shape[0]
        feature_spikes = None
        for (i, layer) in enumerate(self.fc):
            if isinstance(layer, LIFAct):
                if len(x.shape) == 3:
                    x = x.unsqueeze(-1).unsqueeze(-1)
                x = layer(x)
                x = x.squeeze(-1).squeeze(-1)
                if i == len(self.fc) - 2:
                    feature_spikes = x
            elif isinstance(layer, nn.Flatten):
                out = []
                for t in range(T):
                    out.append(layer(x[t]))
                x = torch.stack(out, dim=0)
            elif isinstance(layer, nn.Linear):
                out = []
                for t in range(T):
                    out.append(layer(x[t]))
                x = torch.stack(out, dim=0)
        if feature_spikes is not None:
            feature_rates = feature_spikes.mean(0)
        else:
            feature_rates = None
        return (x, feature_rates)

def get_all_lifact_params(model):
    lifact_params = []
    other_params = []
    lifact_modules = set()
    for (name, module) in model.named_modules():
        if isinstance(module, LIFAct):
            for (param_name, param) in module.named_parameters():
                lifact_params.append(param)
            lifact_modules.add(module)
    for (name, param) in model.named_parameters():
        if param.requires_grad:
            is_lifact = False
            for lifact_mod in lifact_modules:
                for (p_name, p) in lifact_mod.named_parameters():
                    if param is p:
                        is_lifact = True
                        break
                if is_lifact:
                    break
            if not is_lifact:
                other_params.append(param)
    return (lifact_params, other_params)

def get_lifact_temporal_info(model):
    info_list = []
    for (name, module) in model.named_modules():
        if isinstance(module, LIFAct):
            layer_info = {'name': name, 'type': 'LIFAct', 'step': module.step, 'channel': module.channel, 'temporal_params': module.temporal.detach().cpu().numpy(), 'temporal_weights': module.get_temporal_weights(), 'dynamic_thresholds': module.get_dynamic_thresholds(), 'learnable_decay': module.learnable_decay, 'temp': module.temp}
            if module.learnable_decay:
                layer_info['decay_stats'] = module.get_decay_stats()
            else:
                layer_info['decay'] = module.decay
            info_list.append(layer_info)
    return info_list

def compute_st_regularization(model, lambda_temporal=0.01, lambda_nam=0.001, lambda_decay=0.01):
    device = next(model.parameters()).device
    temporal_reg = torch.tensor(0.0, device=device)
    nam_reg = torch.tensor(0.0, device=device)
    decay_reg = torch.tensor(0.0, device=device)
    count_temporal = 0
    count_nam = 0
    count_decay = 0
    for module in model.modules():
        if isinstance(module, LIFAct):
            temporal_reg = temporal_reg + torch.mean(module.temporal ** 2)
            count_temporal += 1
            if module.use_nam and hasattr(module.bn, 'my_NAM'):
                bn_weight = module.bn.my_NAM.weight
                weight_var = torch.var(bn_weight)
                nam_reg = nam_reg + weight_var
                count_nam += 1
            if module.learnable_decay and module.tau is not None:
                decay = torch.sigmoid(module.tau)
                decay_reg = decay_reg + torch.mean((decay - module.decay_init) ** 2)
                count_decay += 1
    total_reg = torch.tensor(0.0, device=device)
    if count_temporal > 0:
        total_reg = total_reg + lambda_temporal * temporal_reg / count_temporal
    if count_nam > 0:
        total_reg = total_reg + lambda_nam * nam_reg / count_nam
    if count_decay > 0:
        total_reg = total_reg + lambda_decay * decay_reg / count_decay
    return total_reg
