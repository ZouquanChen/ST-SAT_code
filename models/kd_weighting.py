"""CE-referenced, detached cosine gates for ST-SATH distillation."""
import math

import torch


def ce_aligned_kd_weights(ce_loss, objectives, parameters, mode='cosine'):
    """Return scalar gates in [0, 1] without modifying parameter .grad buffers.

    Each objective is the already-scaled KD loss. The reference is pure CE,
    over ALL trainable parameters, with unused gradients treated as zero.
    w_i = stop_gradient(clamp(cos(g_CE, g_i), 0, 1)). Therefore
    <g_CE, w_i*g_i> >= 0; this is a global first-order gradient constraint,
    not a per-layer constraint or a guarantee about an Adam/momentum step.
    Zero CE/KD gradients receive zero weight. 'none' restores fixed weights.
    """
    if mode not in ('cosine', 'none'):
        raise ValueError('KD gradient weighting must be cosine or none')
    weights = {name: ce_loss.new_zeros(()) for name in objectives}
    active = {name: loss for name, loss in objectives.items() if loss.requires_grad}
    if not active:
        return weights
    if mode == 'none':
        return {name: ce_loss.new_tensor(float(name in active)) for name in objectives}

    params = tuple(p for p in parameters if p.requires_grad)
    if not params:
        raise ValueError('CE gradient weighting requires trainable parameters')
    ce_grads = torch.autograd.grad(ce_loss, params, retain_graph=True, allow_unused=True)

    # Accumulate per tensor in float64, without flattening/copying the model.
    # Detached gradients keep the gate out of the backward graph (no Hessian).
    def dot(left, right):
        value = ce_loss.new_zeros((), dtype=torch.float64)
        for a, b in zip(left, right):
            if a is not None and b is not None:
                value = value + (a.detach().double() * b.detach().double()).sum()
        return value

    ce_sq = dot(ce_grads, ce_grads)
    if not torch.isfinite(ce_sq):
        raise RuntimeError('Non-finite CE gradient in KD weighting')
    for name, loss in active.items():
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        kd_sq = dot(grads, grads)
        cross = dot(ce_grads, grads)
        if not torch.isfinite(torch.stack((kd_sq, cross))).all():
            raise RuntimeError(f'Non-finite {name} gradient in KD weighting')
        denominator = ce_sq.sqrt() * kd_sq.sqrt()
        # A small denominator only attenuates the weight; it cannot flip its sign.
        weights[name] = (cross / denominator.clamp_min(1e-12)).clamp(0, 1).to(ce_loss).detach()
        del grads
    return weights


def stsat_distillation_loss(ce_loss, kl_per_class, missing, existing, feature,
                            parameters, kd_weight, missing_weight, existing_weight,
                            feature_weight, mode='cosine'):
    """Keep missing-class KD fixed; gate existing-class KD and soft-feature KD.

    The original KL uses a full-class softmax, sliced only after computing KL.
    Static nonnegative coefficients remain separate from the [0, 1] gates.
    Disabled/empty terms are omitted, including their gradient probes.
    Returns (effective KD loss, {'existing': gate, 'feature': gate}).
    """
    coefficients = (kd_weight, missing_weight, existing_weight, feature_weight)
    if any(not math.isfinite(value) or value < 0 for value in coefficients):
        raise ValueError('KD coefficients must be finite and nonnegative')
    loss = ce_loss.new_zeros(())
    if kd_weight * missing_weight > 0 and missing.numel():
        loss = loss + kd_weight * missing_weight * kl_per_class[:, missing.long()].sum(1).mean()
    objectives = {}
    if kd_weight * existing_weight > 0 and existing.numel():
        objectives['existing'] = kd_weight * existing_weight * kl_per_class[:, existing.long()].sum(1).mean()
    if feature_weight > 0 and feature is not None and feature.requires_grad:
        objectives['feature'] = feature_weight * feature
    weights = ce_aligned_kd_weights(ce_loss, objectives, parameters, mode)
    for name, objective in objectives.items():
        loss = loss + weights[name] * objective
    return loss, {name: weights.get(name, ce_loss.new_zeros(()))
                  for name in ('existing', 'feature')}
