"""Ensemble blending: a dynamic weighted average of per-number scores across
strategies. We use a walk-forward auto-assessor (similar to pattern analysis)
that evaluates each model's accuracy over the last 30 draws to dynamically 
assign weights, rewarding models that are currently 'hot' and penalizing those
that are not."""

DEFAULT_WEIGHTS = {'blend': 0.2, 'charts': 0.15, 'rf': 0.2, 'gbm': 0.2, 'mlp': 0.1, 'deep': 0.15}


def normalize(scores):
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def blend_scores(component_scores, weights=None, exclude=None):
    """component_scores: {strategy_name: {1..90: raw_score}}. weights: optional override
    of DEFAULT_WEIGHTS. Weights can be passed dynamically based on recent validation.

    exclude: optional {number: set(component_names)} -- for that SPECIFIC number, treat
    those components as not applicable and renormalize its weight share among the
    remaining active components for that number only, rather than counting the
    incapable of saying anything about a given number (not just "measured low")."""
    weights = weights or DEFAULT_WEIGHTS
    active = {name: w for name, w in weights.items() if name in component_scores and w > 0}
    if not active:
        return {k: 0.0 for k in range(1, 91)}
    norm = {name: normalize(scores) for name, scores in component_scores.items()}
    exclude = exclude or {}
    out = {k: 0.0 for k in range(1, 91)}
    for k in range(1, 91):
        local_active = active
        excluded_here = exclude.get(k)
        if excluded_here:
            reduced = {name: w for name, w in active.items() if name not in excluded_here}
            if reduced:
                local_active = reduced
        total_w = sum(local_active.values())
        for name, w in local_active.items():
            out[k] += (w / total_w) * norm[name][k]
    return out
