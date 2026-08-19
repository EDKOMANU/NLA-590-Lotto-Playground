"""Upgraded chart strategy ('charts2') -- the same 9 traditional charts, measured at a
finer grain and with the assumptions of the legacy 'charts' mode replaced by
measurements. The legacy mode (classic.chart_scores) is deliberately left untouched as
the baseline, so the walk-forward backtest can show exactly what each upgrade is worth.

What changes vs the legacy 'charts' mode, point by point:

1. PER-ENTRY rates, not one rate per chart. A chart is a bundle of ~90 individual
   "a points to b" claims; the legacy mode measures only the bundle average, so entry
   3->17 and entry 88->89 of the same chart always score identically. Here every entry's
   own transfer record (hits/trials) is measured, then EMPIRICAL-BAYES-shrunk toward
   its chart's overall rate with M pseudo-trials: shrunk = (hits + M*chart_rate) /
   (trials + M). A well-sampled entry keeps its own rate; a thin entry collapses to the
   chart average (which, at ~13k pooled trials, is precisely estimated) -- never to an
   overclaimed outlier. Both raw and shrunk values are disclosed.
2. MEASURED machine-source weighting. Legacy hard-codes machine pointers at 0.5x the
   win rate -- an assumption. Here machine-sourced transfers get their own measured
   per-chart/per-entry tables, and machine pointers score at those measured rates.
3. POOLED cross-game trials. The charts are game-agnostic numbering traditions, so
   transfer is measured over every game's own consecutive-draw pairs and pooled
   (~13,400 trials per chart instead of ~2,000), tightening every entry estimate.
   Pooling stays walk-forward pure: callers pass the same date-sliced `all_draws`
   the cross-game component already uses.
4. LAG CURVES as evidence: P(partner appears exactly L draws after the source), per
   chart, L=1..5 -- the "within a few weeks" reading measured, not assumed. Scoring
   stays lag-1 (same prediction target as legacy) so the backtest comparison is
   apples-to-apples; the curve is for the evidence panel.
5. NOISE CONTROL for entry-mining: with ~810 entries measured, the best-looking
   entries are guaranteed to look impressive by selection alone. best_entry_report()
   gives each top entry's exact binomial p-value against the 5.56% chance rate, and
   bootstrap_best_entry_pvalue() answers the family-wise question -- does the real
   chart's best entry beat the best entry a structure-destroyed synthetic history
   produces? -- exactly like the spatial engine's key validation.

Self-pointers (turning's 16 fixed points etc.) stay excluded from scoring and flagged
in evidence, same as legacy. Like every strategy in this project, none of this is
expected to beat chance -- the point is to measure the tradition properly and let the
walk-forward backtest judge the upgrade against its baseline.
"""
import math
from collections import defaultdict

from .charts import CHARTS
from .classic import stat_scores
from .spatial_engine import _synthetic_history, _binom_tail_p

EB_PSEUDO_TRIALS = 30      # M: pseudo-trials pulling a thin entry toward its chart mean
CHANCE = 5 / 90            # 5.56% -- P(a specific partner appears in a 5/90 draw)
LAG_MAX = 5


def _per_game(draws):
    by_game = defaultdict(list)
    for d in draws:
        by_game[d['code']].append(d)
    return by_game


def entry_stats(draws, source_field='win', lag=1):
    """Per chart, per source number: (hits, trials) for "source drawn (in
    `source_field`) at draw i -> chart partner in the WIN numbers of draw i+lag",
    measured over every game's own consecutive-draw sequence in `draws` and pooled.
    Also returns each chart's aggregate (hits, trials)."""
    per_entry = {name: defaultdict(lambda: [0, 0]) for name in CHARTS}
    per_chart = {name: [0, 0] for name in CHARTS}
    for seq in _per_game(draws).values():
        for i in range(len(seq) - lag):
            src_nums = seq[i].get(source_field) or []
            nxt = set(seq[i + lag]['win'])
            for name, mp in CHARTS.items():
                for a in src_nums:
                    b = mp.get(a)
                    if b is None or b == a:  # unmapped or self-pointer: never scored
                        continue
                    e = per_entry[name][a]
                    e[1] += 1
                    per_chart[name][1] += 1
                    if b in nxt:
                        e[0] += 1
                        per_chart[name][0] += 1
    entries = {name: {a: (h, t) for a, (h, t) in d.items()} for name, d in per_entry.items()}
    chart_totals = {name: (h, t) for name, (h, t) in per_chart.items()}
    return entries, chart_totals


def shrunk_entry_rate(hits, trials, chart_rate, m=EB_PSEUDO_TRIALS):
    """Empirical-Bayes shrink of one entry's rate toward its chart's overall rate:
    (hits + m*chart_rate) / (trials + m). trials=0 returns the chart rate exactly."""
    return (hits + m * chart_rate) / (trials + m)


def chart_tables(draws):
    """Everything charts2 scoring needs, measured from `draws` (pass the pooled,
    date-sliced all_draws): win-source and machine-source entry tables with their
    chart aggregates."""
    win_entries, win_totals = entry_stats(draws, 'win')
    mach_entries, mach_totals = entry_stats(draws, 'mach')
    return {'win_entries': win_entries, 'win_totals': win_totals,
            'mach_entries': mach_entries, 'mach_totals': mach_totals}


def _pointer_weight(tables, kind, name, a):
    entries = tables[f'{kind}_entries'][name]
    ch_h, ch_t = tables[f'{kind}_totals'][name]
    chart_rate = ch_h / ch_t if ch_t else 0.0
    h, t = entries.get(a, (0, 0))
    return shrunk_entry_rate(h, t, chart_rate), h, t, chart_rate


def chart2_pointer_hits(history, tables):
    """Every non-self chart pointer from the current draw (win AND machine sources),
    each with its own measured entry record, chart rate, and the EB-shrunk weight that
    actually scores -- the full evidence trail behind chart2_scores."""
    if not history:
        return []
    last = history[-1]
    out = []
    for kind, nums in (('win', last['win']), ('mach', last.get('mach') or [])):
        for name, mp in CHARTS.items():
            for a in nums:
                b = mp.get(a)
                if b is None:
                    continue
                if b == a:
                    out.append({'chart': name, 'from': a, 'from_kind': kind, 'to': b,
                                'self_pointer': True, 'weight': 0.0,
                                'entry_hits': 0, 'entry_trials': 0, 'chart_rate': None})
                    continue
                w, h, t, cr = _pointer_weight(tables, kind, name, a)
                out.append({'chart': name, 'from': a, 'from_kind': kind, 'to': b,
                            'self_pointer': False, 'weight': w,
                            'entry_hits': h, 'entry_trials': t, 'chart_rate': cr})
    out.sort(key=lambda e: -e['weight'])
    return out


def chart2_scores(history, all_draws=None, tables=None):
    """The upgraded chart score: every non-self pointer from the current draw credits
    its target at the pointer's own measured, EB-shrunk entry rate -- win-sourced and
    machine-sourced pointers each at their own measured rates (no assumed 0.5x).
    Blended 50/50 with the same recency baseline as the legacy 'charts' mode, so a
    backtest difference between the two modes isolates the chart-scoring upgrade.
    Returns the blended {1..90} score dict."""
    if tables is None:
        tables = chart_tables(all_draws if all_draws else history)
    sc = {k: 0.0 for k in range(1, 91)}
    for p in chart2_pointer_hits(history, tables):
        if not p['self_pointer']:
            sc[p['to']] += p['weight']
    base = stat_scores(history, 'recent')
    mx = max(sc.values()) or 1.0
    mb = max(base.values()) or 1.0
    return {k: 0.5 * base[k] / mb + sc[k] / mx for k in range(1, 91)}


def lag_curves(draws, max_lag=LAG_MAX, source_field='win'):
    """Per chart: the transfer rate at each exact lag 1..max_lag -- 'the partner tends
    to come within a few weeks' as a measured curve (chance = 5.56% at every lag, since
    each lag is one specific future draw)."""
    curves = {}
    for lag in range(1, max_lag + 1):
        _, totals = entry_stats(draws, source_field, lag=lag)
        for name, (h, t) in totals.items():
            curves.setdefault(name, {})[lag] = (h / t if t else 0.0, t)
    return curves


def best_entry_report(draws, top_n=10, min_trials=25):
    """The individually best-looking entries across all charts (win-source), each with
    its exact binomial p-value vs the 5.56% chance rate -- PLUS the count of entries
    tested, because with ~800 entries the best few are guaranteed to look impressive
    by selection alone (that is what bootstrap_best_entry_pvalue is for)."""
    entries, totals = entry_stats(draws, 'win')
    rows = []
    tested = 0
    for name, d in entries.items():
        for a, (h, t) in d.items():
            if t < min_trials:
                continue
            tested += 1
            rows.append({'chart': name, 'from': a, 'to': CHARTS[name][a], 'hits': h,
                         'trials': t, 'rate': h / t, 'p_value': _binom_tail_p(h, t, CHANCE)})
    rows.sort(key=lambda r: r['p_value'])
    return {'top': rows[:top_n], 'tested': tested,
            'best_p': rows[0]['p_value'] if rows else 1.0}


def bootstrap_best_entry_pvalue(draws, iterations=100, seed=0, min_trials=25):
    """Family-wise noise control for entry-mining, exactly like the spatial engine's
    key validation: re-run the full best-entry search on structure-destroyed synthetic
    histories (fresh uniform draws, same shape) and report how often noise produces a
    best entry at least as impressive as the real one. > 0.05 means the real best
    entries are indistinguishable from apophenia and must be read as chart-average
    performers, not special relationships."""
    import random
    rng = random.Random(seed)
    real_best = best_entry_report(draws, top_n=1, min_trials=min_trials)['best_p']
    ge = 0
    for _ in range(iterations):
        fake = best_entry_report(_synthetic_history(draws, rng), top_n=1, min_trials=min_trials)
        ge += fake['best_p'] <= real_best
    return {'real_best_p': real_best, 'bootstrap_p': ge / iterations, 'iterations': iterations}


def explain_number(history, number, all_draws=None, tables=None):
    """charts2's per-number evidence: every pointer landing on `number` with its own
    measured entry record (and the chart-average fallback a thin entry is shrunk
    toward), plus the measured machine-source rates that replace the legacy assumed
    0.5x weighting."""
    if tables is None:
        tables = chart_tables(all_draws if all_draws else history)
    hits = [p for p in chart2_pointer_hits(history, tables) if p['to'] == number]
    non_self = [p for p in hits if not p['self_pointer']]
    parts = []
    if non_self:
        best = non_self[0]
        parts.append(
            f"{best['from']} ({best['from_kind']} number, current draw) points to {number} via the "
            f"{best['chart']} chart. This ENTRY's own record: {best['entry_hits']}/{best['entry_trials']} "
            f"({(best['entry_hits'] / best['entry_trials']) if best['entry_trials'] else 0:.2%}) vs the "
            f"chart's pooled average {best['chart_rate']:.2%} (chance 5.56%); EB-shrunk to a "
            f"{best['weight']:.2%} score weight"
            + (f"; {len(non_self) - 1} other pointer(s) also land here." if len(non_self) > 1 else "."))
    elif hits:
        parts.append(f"{number} only has SELF chart pointer(s) here -- disclosed, never scored.")
    else:
        parts.append(f"No chart pointer from the current draw lands on {number}.")
    parts.append("Machine-sourced pointers score at their own MEASURED rates here (legacy 'charts' "
                 "assumed 0.5x the win rate instead). None of this is causal for an independent "
                 "random draw -- see the Methodology tab.")
    return {'chart_hits': hits, 'narrative': " ".join(parts)}
