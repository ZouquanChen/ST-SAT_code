"""CE-referenced, detached cosine gates for ST-SATH distillation."""
import torch


def ce_aligned_kd_weights(ce_loss, objectives, parameters):
    """Return scalar gates in [0, 1] without modifying parameter .grad buffers.

    Each objective is an unscaled KD loss. The caller supplies its reference,
    over ALL trainable parameters, with unused gradients treated as zero.
    w_i = stop_gradient(clamp(cos(g_CE, g_i), 0, 1)). Therefore
    <g_CE, w_i*g_i> >= 0; this is a global first-order gradient constraint,
    not a per-layer constraint or a guarantee about an Adam/momentum step.
    Zero reference/KD gradients receive zero weight.
    """
    weights = {name: ce_loss.new_zeros(()) for name in objectives}
    active = {name: loss for name, loss in objectives.items() if loss.requires_grad}
    if not active:
        return weights

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
                            parameters):
    """Original dynamic B: unit missing KL and detached existing/feature gates.

    All available KD terms participate, with no coefficient arguments or
    enable switches. Both KL terms use the full-class softmax. Existing
    KL is gated against pure CE. Feature MSE is gated against CE + missing KL
    + detached-existing-gate * existing KL, excluding regularization and the
    feature term itself. The caller adds regularization exactly once.
    """
    parameters = tuple(parameters)
    loss = ce_loss.new_zeros(())
    weights = {'existing': ce_loss.new_zeros(()), 'feature': ce_loss.new_zeros(())}
    if missing.numel():
        loss = loss + kl_per_class[:, missing.long()].sum(1).mean()
    if existing.numel():
        existing_loss = kl_per_class[:, existing.long()].sum(1).mean()
        weights['existing'] = ce_aligned_kd_weights(
            ce_loss, {'existing': existing_loss}, parameters)['existing']
        loss = loss + weights['existing'] * existing_loss
    if feature is not None and feature.requires_grad:
        reference = ce_loss + loss
        weights['feature'] = ce_aligned_kd_weights(
            reference, {'feature': feature}, parameters)['feature']
        loss = loss + weights['feature'] * feature
    return loss, weights
