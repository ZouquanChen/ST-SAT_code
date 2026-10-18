"""CE-referenced, detached cosine gates for ST-SATH distillation."""
import math

import torch


def ce_aligned_kd_weights(ce_loss, objectives, parameters, mode='cosine'):
    """Return scalar gates in [0, 1] without modifying parameter .grad buffers.

    Each objective is the unscaled KD loss. The reference is pure CE,
    over ALL trainable parameters, with unused gradients treated as zero.
    w_i = stop_gradient(clamp(cos(g_CE, g_i), 0, 1)). Therefore
    <g_CE, w_i*g_i> >= 0; this is a global first-order gradient constraint,
    not a per-layer constraint or a guarantee about an Adam/momentum step.
    Zero CE/KD gradients receive zero weight. 'none' uses unit gates.
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
                            feature_weight, mode='cosine', regularization=None, feature_reference='ce', diagnostics=False, fixed_feature_weight=0.01):
    """Use unit missing-class KD and cosine-only existing/feature KD weights.

    The original KL uses a full-class softmax, sliced only after computing KL.
    Legacy nonnegative coefficient arguments are enable switches: zero disables
    a term, while any positive value enables it without scaling its loss.
    kd_weight enables class KD; feature KD is enabled independently.
    Disabled/empty terms are omitted, including their gradient probes.
    'dual' takes min(CE cosine, complete-base cosine) for feature KD only.
    Complete base = CE + missing KD + CE-gated existing KD + regularization.
    All gates are detached. Regularization is referenced but NOT added to the
    returned KD loss; callers add it exactly once. 'none' uses unit gates.
    'fixed' uses fixed_feature_weight directly (no static outer coefficient
    and no feature-gradient probes); feature_weight remains an enable switch.
    Fixed-mode CE/base feature diagnostics are zero placeholders, not cosines.
    'ce_missing' uses CE + unit missing KD; 'ce_missing_existing' adds the
    actual detached CE-gated existing KD. Both use the reference cosine
    directly, without a CE minimum and without referencing regularization.
    In these modes feature_base records that selected reference gate.
    Extra diagnostics report clipped cosines, reductions/amplifications
    relative to CE, and detached raw/effective feature loss.
    """
    if feature_reference not in ('ce', 'dual', 'fixed', 'ce_missing', 'ce_missing_existing'):
        raise ValueError('feature_reference must be ce, dual, fixed, ce_missing or ce_missing_existing')
    parameters = tuple(parameters)
    coefficients = (kd_weight, missing_weight, existing_weight, feature_weight, fixed_feature_weight)
    if any(not math.isfinite(value) or value < 0 for value in coefficients):
        raise ValueError('KD enable values must be finite and nonnegative')
    loss = ce_loss.new_zeros(())
    if kd_weight > 0 and missing_weight > 0 and missing.numel():
        loss = loss + kl_per_class[:, missing.long()].sum(1).mean()
    missing_loss = loss
    objectives = {}
    if kd_weight > 0 and existing_weight > 0 and existing.numel():
        objectives['existing'] = kl_per_class[:, existing.long()].sum(1).mean()
    if feature_weight > 0 and feature is not None and feature.requires_grad:
        objectives['feature'] = feature
    # Fixed feature KD never probes feature gradients or applies a cosine gate.
    gated_objectives = {k:v for k,v in objectives.items()
                        if not (feature_reference == 'fixed' and k == 'feature')}
    weights = ce_aligned_kd_weights(ce_loss, gated_objectives, parameters, mode)
    if feature_reference == 'fixed' and 'feature' in objectives:
        weights['feature'] = ce_loss.new_tensor(fixed_feature_weight).detach()
    zero = ce_loss.new_zeros(())
    # Existing-class KD stays CE-gated. Its detached gate belongs in the
    # complete base reference, which excludes feature KD to avoid circularity.
    if 'existing' in objectives:
        loss = loss + weights['existing'] * objectives['existing']
    feature_ce = weights.get('feature', zero) if feature_reference != 'fixed' else zero
    feature_base = feature_ce
    if feature_reference == 'dual' and mode == 'cosine' and 'feature' in objectives:
        base = ce_loss + loss
        if regularization is not None:
            base = base + regularization
        feature_base = ce_aligned_kd_weights(
            base, {'feature': objectives['feature']}, parameters, mode)['feature']
        weights['feature'] = torch.minimum(feature_ce, feature_base).detach()
    if feature_reference in ('ce_missing', 'ce_missing_existing') and mode == 'cosine' and 'feature' in objectives:
        # These references target classification and output KD only. The
        # existing gate is CE-derived and detached; feature and regularizers
        # must not be included in the reference (no self-alignment).
        reference = ce_loss + (missing_loss if feature_reference == 'ce_missing' else loss)
        feature_base = ce_aligned_kd_weights(
            reference, {'feature': objectives['feature']}, parameters, mode)['feature']
        weights['feature'] = feature_base.detach()  # DIRECT gate, no min with CE
    if 'feature' in objectives:
        loss = loss + weights['feature'] * objectives['feature']
    feature_effective = weights.get('feature', zero)
    result = {
        'existing': weights.get('existing', zero), 'feature': feature_effective,
        'feature_ce': feature_ce, 'feature_base': feature_base,
        'feature_reduced': (feature_effective < feature_ce).to(ce_loss).detach(),
        'feature_blocked_by_base': ((feature_ce > 0) & (feature_effective == 0)).to(ce_loss).detach(),
        'feature_amplified': ((feature_effective > feature_ce) & (feature_reference != 'fixed')).to(ce_loss).detach(),
        'feature_raw_loss': feature.detach() if 'feature' in objectives else zero,
        'feature_weighted_loss': feature_effective * feature.detach() if 'feature' in objectives else zero,
    }
    if not diagnostics:
        result = {name: result[name] for name in ('existing', 'feature')}
    return loss, result
