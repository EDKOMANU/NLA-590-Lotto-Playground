"""Classic heuristic strategies: hot/recent/overdue/blend + chart-relationship scoring,
plus the strategy dispatch table. Moved verbatim (in behavior) from the original predictor.py."""
from collections import Counter, defaultdict

from .charts import CHARTS
from .config import EXP_GAP


def chart_scores(history):
    """Score numbers by how strongly the last draw 'points' at them through the charts,
    weighted by each chart's historical transfer rate in this game.

    A handful of chart entries are fixed points (the number maps to itself -- e.g. the
    'turning' chart has 16: 11, 19, 22, 29, 33, 39, 44, 49, 55, 59, 66, 69, 77, 79, 88,
    89). Those are skipped here: crediting a number for 'pointing at itself' isn't a
    cross-number relationship, it's just re-crediting a number for having been drawn
    last time -- see pattern_analysis's honest caveat on this. Still visible via
    pattern_analysis.chart_pointers_for_last_draw() (flagged 'self_pointer'), just not
    counted toward the score."""
    sc = {k: 0.0 for k in range(1, 91)}
    if not history: return sc
    last = history[-1]
    for name, mp in CHARTS.items():
        hits = trials = 0
        recent_history = history[-51:] if len(history) > 50 else history
        for a, b in zip(recent_history[:-1], recent_history[1:]):
            nxt = set(b['win'])
            for n in a['win']:
                if n in mp:
                    trials += 1
                    hits += mp[n] in nxt
        rate = hits / trials if trials else 0.0
        w = max(rate, 0.0)
        for n in last['win'] + last['mach']:
            if n in mp and mp[n] != n:
                sc[mp[n]] += w * (1.0 if n in last['win'] else 0.5)
    return sc


def chart_transfer_rates(history):
    """Measured transfer rate per chart (how often a chart-partner of a drawn number
    appears in the next draw), for display alongside the 5.56% chance rate."""
    rates = {}
    for name, mp in CHARTS.items():
        hits = trials = 0
        recent_history = history[-51:] if len(history) > 50 else history
        for a, b in zip(recent_history[:-1], recent_history[1:]):
            nxt = set(b['win'])
            for x in a['win']:
                if x in mp:
                    trials += 1
                    hits += mp[x] in nxt
        rates[name] = hits / trials if trials else 0.0
    return rates


def stat_scores(history, mode='blend', half_life=60):
    n = len(history)
    sc = {k: 1.0 for k in range(1, 91)}
    if n == 0: return sc
    freq = Counter(); wfreq = defaultdict(float); last_seen = {}
    for i, d in enumerate(history):
        w = 0.5 ** ((n - 1 - i) / half_life)
        for x in d['win']:
            freq[x] += 1; wfreq[x] += w; last_seen[x] = i
    for k in range(1, 91):
        gap = n - last_seen.get(k, -1) - 1
        if mode == 'hot': sc[k] = freq.get(k, 0)
        elif mode == 'recent': sc[k] = wfreq.get(k, 0.0)
        elif mode == 'overdue': sc[k] = gap / EXP_GAP
        else: sc[k] = wfreq.get(k, 0.0) * (1.0 + 0.15 * min(gap / EXP_GAP, 3.0))
    return sc


def explain(history, number, mode):
    """Per-number evidence behind hot/recent/overdue/blend -- the literal frequency/
    recency/gap facts that feed stat_scores()'s formula for this one candidate, laid
    out the same way pattern_analysis.explain() does for the richer pattern-analysis
    system, so these simpler strategies get 'show your work' too rather than an opaque
    number."""
    n = len(history)
    half_life = 60
    freq = 0; freq_30 = 0; wfreq = 0.0; last_seen_idx = None
    for i, d in enumerate(history):
        if number in d['win']:
            freq += 1
            wfreq += 0.5 ** ((n - 1 - i) / half_life)
            last_seen_idx = i
            if i >= n - 30:
                freq_30 += 1
    gap = (n - 1 - last_seen_idx) if last_seen_idx is not None else n
    gap_ratio = gap / EXP_GAP
    last_seen_date = history[last_seen_idx]['date'] if last_seen_idx is not None else None

    if mode == 'hot':
        formula = f"hot score = raw frequency = {freq}"
        narrative = (f"{number} has been drawn {freq} times in {n} draws"
                     + (f" ({freq_30} of the last 30)" if freq_30 else "")
                     + ". 'hot' ranks purely by this count -- no recency or gap weighting at all.")
    elif mode == 'recent':
        formula = f"recent score = recency-weighted frequency (half-life {half_life} draws) = {wfreq:.3f}"
        narrative = (f"{number}'s recency-weighted frequency is {wfreq:.3f} (drawn {freq} times total, "
                     f"each occurrence discounted by 0.5^(draws-ago/{half_life})) -- older wins count for "
                     f"progressively less than recent ones.")
    elif mode == 'overdue':
        formula = f"overdue score = gap / expected_gap = {gap}/{EXP_GAP:.0f} = {gap_ratio:.2f}"
        narrative = (f"{number} was last drawn {gap} draws ago"
                     + (f" (on {last_seen_date})" if last_seen_date else " (never seen in this history)")
                     + f", vs. an expected gap of {EXP_GAP:.0f} draws for a 5/90 game (90/5) -- 'overdue' "
                       f"ranks purely by how far past that expectation a number is.")
    else:  # blend
        bonus = 1.0 + 0.15 * min(gap_ratio, 3.0)
        formula = (f"blend score = recency-weighted frequency x (1 + 0.15 x min(gap_ratio, 3)) "
                   f"= {wfreq:.3f} x {bonus:.3f}")
        narrative = (f"{number}'s recency-weighted frequency ({wfreq:.3f}) is boosted {(bonus-1)*100:.1f}% "
                     f"for being {gap} draws overdue vs. an expected {EXP_GAP:.0f} (the boost caps at 45% "
                     f"however overdue a number gets) -- 'blend' combines recency and overdueness into one "
                     f"score.")

    return {
        'freq_all': freq, 'freq_30': freq_30, 'wfreq': wfreq,
        'gap': gap, 'gap_ratio': gap_ratio, 'expected_gap': EXP_GAP,
        'last_seen_date': last_seen_date, 'n_draws': n,
        'formula': formula,
        'narrative': narrative + " None of this is causal for an independent random draw -- see the Methodology tab.",
    }


def picks(scores, k):
    return [n for n, _ in sorted(scores.items(), key=lambda t: (-t[1], t[0]))[:k]]


def get_scores(history, mode):
    """Dispatch for the legacy modes (hot/recent/overdue/blend/charts/ml). New modes
    (rf/gbm/mlp/deep/ensemble) are dispatched by lottery_core's higher-level modules
    since they need trained artifacts, not just history."""
    if mode == 'charts':
        base = stat_scores(history, 'recent')
        ch = chart_scores(history)
        mx = max(ch.values()) or 1.0
        mb = max(base.values()) or 1.0
        return {k: 0.5 * base[k] / mb + ch[k] / mx for k in range(1, 91)}
    if mode == 'ml':
        from .features import ml_scores
        return ml_scores(history)
    return stat_scores(history, mode)
