"""Pattern Analysis: a standalone, no-training-required prediction system.

This is the second of the two systems in this project: it composes the lottery-paper
reading strategy discussed with the user directly from history on every call --
chart relationships, machine-number affinity, and the pattern-transformation engine in
transform_engine.py (literal-carryover pair/triplet tracing with position tracking, a
registry of measured arithmetic transforms, and positional-stickiness weighting).
Unlike sklearn_models.py/deep_model.py it holds no trained artifact and never goes
stale: run it again after `predictor.py update` and it immediately reflects the new
draw, no `train.py` needed.

This is explicitly a rule-*discovery* system, not a frequency counter: rather than
assuming "the same number comes back" or "addition is the technique," it identifies
recurring group structures (pairs and triplets) and their sorted-rank positions, looks
at how those groups transitioned into winning numbers historically (direct carry-over,
or via one of several measured arithmetic transforms, or via positional stickiness),
and replicates whichever transformation paths have actually shown reliability -- while
being explicit that with ~2,700 independent random draws, "reliability" here is
measured honestly and is expected to land near chance (see the Methodology tab).

Every component is fully transparent: `explain()` returns the concrete historical
evidence behind a number's score -- which groups repeated and what followed, at which
positions, which transform rules derive it and each rule's own measured reliability,
how many independent transformation paths agree -- so the reasoning can be audited the
way a human reader manually tracing patterns would, rather than trusting an opaque
score. This is the tradeoff against the ML Models tab: interpretable and always fresh,
versus statistically richer but requiring periodic retraining.

Honest caveat about one of the seven scored components: 'recent' (see DEFAULT_WEIGHTS
below) is plain recency-weighted frequency (classic.stat_scores(history, 'recent')) --
exactly the kind of signal this module's structural components are meant to go beyond.
It is kept, at a disclosed fixed weight, as the simplest possible baseline the other six
genuinely structural/transformation components (charts, mach_to_win, pattern_trace,
transform, terminal, group) can be judged against via `_contribution_breakdown()` -- not
as something the "rule-discovery, not a frequency counter" framing above is meant to
hide.

DIAGNOSTIC-ONLY COMPONENTS -- computed, disclosed, but excluded from the score: three
named lottery-paper strategies live in transform_engine.py but do NOT feed
component_scores()/blend_scores(): positional_carryover_score ('does a number in this
sorted-rank position tend to carry over') and lap_score ('does a number in this
as-drawn slot carry over specifically into the very next draw') are structurally
nonzero ONLY for the 5 numbers in the immediately preceding draw -- by construction,
not incidentally. Measured on real history, including them in the blend made the
top-5 picks overlap LAST WEEK'S OWN 5 numbers at ~2.4/5 on average (8x the ~0.28/5
chance level two random 5-number sets would share), while overlap with the ACTUAL next
draw stayed flat at chance -- i.e. the system was substantially echoing last week's
draw back out, not forecasting, directly contradicting this module's own "not to bring
down numbers directly" framing above. They're kept fully computed and shown in
explain()'s evidence trail (via te.positional_carryover_rates/lap_carryover_rates
directly, not through component_scores()) because "does this exact number repeat" is a
legitimate thing to disclose evidence about -- it just isn't allowed to drive which
numbers get picked. 'terminal' and 'group' are similarly affected in a narrower way (a
number is always trivially a member of its own class), so class_carryover_score
(transform_engine.py) excludes that one SELF-credited number from the score while still
crediting every other member of a touched class, and still discloses the self-fact via
`detail`'s `self_credit` flag for explain() to show. 'charts' has the same issue for the
16 numbers where the 'turning' chart maps to itself (11, 19, 22, ..., 89) --
classic.chart_scores() skips those specific self-pointers when scoring, while
chart_pointers_for_last_draw() still discloses them (flagged `self_pointer`).

Because a number can only ever belong to ONE terminal and ONE digital-root group (its
own), the self-credit exclusion above means last week's own 5 numbers can NEVER reach
terminal/group credit through any path -- a structural zero, not a probabilistic one, on
two of the most heavily-weighted components. The first version of this fix overcorrected
exactly that way: re-measured, top-5/last-week overlap dropped to 0.000/5 (ranked
consistently below median -- disfavored, not neutral). See _self_exclusion() and
ensemble.blend_scores()'s `exclude` parameter: for a number that IS one of the seed
draw's own, 'terminal'/'group' are treated as not-applicable and the remaining
components' weights are renormalized around them, rather than counting the structural
silence as a literal (worst-possible, post-normalization) zero. Final measured result:
0.245/5 overlap with last week, 0.323/5 with the actual next draw -- both within normal
sampling noise of the 0.278/5 chance level (375 walk-forward test points, all 7 games).

Two more named lottery-paper strategies ARE scored (see transform_engine.py): 'terminal'
(does the last-digit terminal group -- e.g. the "7s": 7/17/27/.../87 -- recur in OTHER
members, crediting any member but itself) and 'group' (the digital-root grouping:
repeatedly sum a number's digits to a single 1-9 value, e.g. 67 -> 13 -> 4, same logic).
Both kept at a modest weight since they're narrower/newer signals than the original five.

FOUR further scored components (see trend_analysis.py and transform_engine.py):
- 'trend' -- trend-window similarity: profile the last few draws as a trend (sums,
  spans, parity, decade/terminal distribution, internal repeats, and the literal
  numbers), find the most similar NON-overlapping historical windows, and credit what
  won immediately after each, weighted by similarity. This is the "trace the new trend
  back through the old papers" strategy made explicit. Overlap with the present is
  excluded by construction so similarity can't degenerate into self-similarity (the
  same echo failure mode 'positional'/'lap' exhibited, see below).
- 'conditional' -- the same patterns measured under different CONDITIONS: every
  historical draw is classified along four fixed, pre-declared dimensions (sum band,
  parity profile, span band, repeated-a-number-from-its-predecessor), each number's
  next-draw appearance rate is measured per condition, and the current draw's own
  conditions select which measured rates apply. Band boundaries come from the passed
  history only (walk-forward safe), and the dimensions are declared, not mined.
- 'cross_game' -- the "today's results point at tomorrow's game" reading: numbers
  drawn in OTHER games since this game's last draw, weighted by each source game's own
  measured, Wilson-shrunk transfer rate into this game. Only computed when the caller
  passes `all_draws` (the pattern tab, predictor.py and backtest.py all do); with
  per-game history alone the component is simply absent and the remaining weights
  renormalize.
- 'mach_trace' -- machine-seeded pattern tracing: the current draw's MACHINE pairs/
  triplets traced against historical MACHINE draws, crediting what won after each
  repeat (pattern_trace_score with field='mach').

DEFAULT_WEIGHTS is a static fallback; `dynamic_weights()` is the actual default used by
`pattern_scores()` -- a walk-forward auto-assessor that reweights the components by how
much score-mass each one actually placed on the real winners over the last 30 draws.
Note this is a deliberate departure from ensemble.py's stated anti-auto-tuning stance
for the ML ensemble (fixed weights there specifically to avoid fitting backtest noise);
here it's an explicit tradeoff the project owner chose to try. Measured caveats worth
knowing before reading too much into any resulting weight: (1) it's ~30x the cost of a
single component_scores() call (mitigated here via a small memoization cache, and in
backtest.py via a throttled recompute), and (2) it structurally rewards components whose
scoring touches many numbers (less sparse) over components that only touch a few,
regardless of real skill, since the accuracy metric is raw score-mass landed on winners,
not a sparsity-adjusted one -- a component's dynamic weight is not on its own evidence
of how good a signal it is.
"""
from . import features as feat_mod
from . import transform_engine as te
from . import trend_analysis as ta
from .classic import chart_scores, chart_transfer_rates, stat_scores, explain as stat_explain
from .charts import CHARTS
from .ensemble import blend_scores, normalize

DEFAULT_WEIGHTS = {
    'recent': 0.13,       # plain recency-weighted frequency -- the baseline, see module docstring
    'charts': 0.09, 'mach_to_win': 0.09,
    'pattern_trace': 0.13, 'transform': 0.13,
    # 'terminal'/'group' are the classic lottery-paper strategies of the same name --
    # see transform_engine's terminal_score/group_score docstrings. Kept at a modest
    # weight since they're narrower/newer signals than the original five.
    'terminal': 0.04, 'group': 0.04,
    # The trend-similarity, conditional, cross-game, and machine-trace strategies --
    # see the module docstring and trend_analysis.py. 'trend' and 'conditional' carry
    # real weight since they're the broadest of the new signals; 'cross_game' and
    # 'mach_trace' start modest like terminal/group.
    'trend': 0.10, 'conditional': 0.10,
    'cross_game': 0.04, 'mach_trace': 0.04,
    # 'yearly' -- anniversary recurrence in the same calendar window across years
    # (see trend_analysis.yearly_rates; strategy studied from the Kaigee forecasting
    # app, re-implemented with measured rates instead of heuristic confidences).
    # Finally meaningful now that the archive reaches 1962.
    'yearly': 0.07,
    # NOTE: 'positional' and 'lap' are deliberately NOT here -- see the module docstring
    # ("diagnostic-only components") for why. They're still computed and shown in
    # explain()'s evidence trail via te.positional_carryover_rates/lap_carryover_rates
    # directly, just excluded from the blended score.
}


def component_scores(history, all_draws=None):
    """The scored components of the pattern-analysis blend, each on its own raw scale
    (see ensemble.blend_scores for the normalization applied before blending).
    'positional' and 'lap' are intentionally absent -- see the module docstring.
    'cross_game' is present only when `all_draws` (every game's draws, date-sorted,
    same cutoff as `history`) is provided; blend_scores/dynamic_weights renormalize
    around its absence otherwise."""
    last = history[-1] if history else None
    mach_rates, _ = feat_mod.mach_to_win_affinity(history)
    pattern_trace, _ = te.pattern_trace_score(history)
    transform, _ = te.transform_score(history, last)
    terminal, _ = te.terminal_score(history, last)
    group, _ = te.group_score(history, last)
    trend, _ = ta.trend_match_score(history)
    conditional, _ = ta.conditional_score(history, last)
    mach_trace, _ = te.pattern_trace_score(history, field='mach')
    yearly, _ = ta.yearly_score(history)
    comps = {
        'yearly': yearly,
        'recent': stat_scores(history, 'recent'),
        'charts': chart_scores(history),
        'mach_to_win': mach_rates,
        'pattern_trace': pattern_trace,
        'transform': transform,
        'terminal': terminal,
        'group': group,
        'trend': trend,
        'conditional': conditional,
        'mach_trace': mach_trace,
    }
    if all_draws and last:
        cross, _ = ta.cross_game_score(all_draws, last['code'], last['date'])
        comps['cross_game'] = cross
    return comps


_dynamic_weights_cache = {}
_DYNAMIC_WEIGHTS_CACHE_MAX = 8  # small: only needs to survive one page render / explain() burst


def _history_fingerprint(history, all_draws=None):
    """Cheap, sufficient identity check for `history` in this codebase: every caller
    passes a chronological prefix of one game's draws, so (length, last draw's date and
    game code) uniquely identifies the content without hashing every draw. `all_draws`
    (when provided, for cross-game scoring) is identified the same way."""
    if not history:
        return (0, None, None, 0)
    last = history[-1]
    return (len(history), last['date'], last.get('code'),
            len(all_draws) if all_draws else 0)


def dynamic_weights(history, window=30, all_draws=None):
    """Walk-forward auto-assessor. Measures the actual accuracy of each component over the
    last `window` draws (by summing the normalized scores it gave to the actual winning
    numbers) and returns a proportionally-weighted dictionary to use instead of
    DEFAULT_WEIGHTS.

    Expensive (~30x a single component_scores() call, since it recomputes every
    component for each of the last `window` draws) -- memoized per distinct `history` so
    that a single prediction/explain() burst against the same history (e.g. the "Show
    your work" panel, which calls this indirectly once per displayed number) only pays
    that cost once, not once per number. This does NOT help a walk-forward backtest,
    where every test point's history is genuinely new by construction -- see
    backtest.py's own throttled recompute for that case.

    When `all_draws` is provided, each simulated test point sees only the draws (of
    every game) strictly before that point's own date -- the same walk-forward
    discipline as everything else here, applied to the cross-game component."""
    if len(history) <= window:
        return DEFAULT_WEIGHTS

    key = _history_fingerprint(history, all_draws)
    cached = _dynamic_weights_cache.get(key)
    if cached is not None:
        return cached

    accuracy = {}
    all_dates = [d['date'] for d in all_draws] if all_draws else None

    # Simulate predicting the last `window` draws
    for i in range(len(history) - window, len(history)):
        sub_history = history[:i]
        actual_winners = history[i]['win']
        sub_all = (ta.slice_before(all_draws, history[i]['date'], all_dates)
                   if all_draws else None)

        comps = component_scores(sub_history, all_draws=sub_all)
        for name, scores in comps.items():
            norm = normalize(scores)
            # Add the probability mass this component placed on the winning numbers
            accuracy[name] = accuracy.get(name, 0.0) + sum(norm.get(w, 0.0) for w in actual_winners)

    total = sum(accuracy.values())
    result = DEFAULT_WEIGHTS if total <= 0 else {name: (val / total) for name, val in accuracy.items()}

    if len(_dynamic_weights_cache) >= _DYNAMIC_WEIGHTS_CACHE_MAX:
        _dynamic_weights_cache.pop(next(iter(_dynamic_weights_cache)))
    _dynamic_weights_cache[key] = result
    return result


# 'terminal'/'group' can only ever credit a number for sharing a class with a DIFFERENT
# number in the seed draw -- a number that IS itself one of the seed draw's own 5 has no
# other class to be reached through (it can only ever be a member of its own class,
# where it's excluded as self-credit, see transform_engine.class_carryover_score). So
# these two components are structurally silent for exactly those 5 numbers, not "low" --
# treating that silence as a literal 0 in a min-max-normalized blend would read as "worst
# possible" and systematically rank them last, which is just last week's-own-numbers bias
# in the OPPOSITE direction. blend_scores' `exclude` renormalizes around that silence.
_SELF_EXCLUDED_COMPONENTS = {'terminal', 'group'}


def _self_exclusion(history):
    last = history[-1] if history else None
    return {x: _SELF_EXCLUDED_COMPONENTS for x in last['win']} if last else {}


def pattern_scores(history, weights=None, all_draws=None):
    """The blended pattern-analysis score: {1..90: score}. No trained model involved --
    every call recomputes directly from `history` (and `all_draws`, when provided, for
    the cross-game component)."""
    comp = component_scores(history, all_draws=all_draws)
    w = weights if weights is not None else dynamic_weights(history, all_draws=all_draws)
    return blend_scores(comp, w, exclude=_self_exclusion(history)), comp


def pattern_trace_events(history, group_sizes=(2, 3), lookahead=None, top_n=15):
    """Concrete evidence for the pattern-trace component: every earlier repeat (pair or
    triplet) of the current draw's groups, with what won afterward and at what
    positions. Delegates to transform_engine (see there for the full structure)."""
    events = te.pattern_trace_events(history, group_sizes=group_sizes,
                                      lookahead=lookahead or te.TRACE_LOOKAHEAD)
    return events[:top_n]


def chart_pointers_for_last_draw(history):
    """Which chart points from the most recent draw's numbers to which candidates,
    with each chart's measured transfer rate -- the concrete evidence behind the
    'charts' component. `self_pointer` flags entries where the chart maps a number to
    itself (e.g. 16 of 'turning's entries) -- disclosed here, but classic.chart_scores()
    excludes these from the actual score (see its docstring)."""
    if not history:
        return []
    last = history[-1]
    rates = chart_transfer_rates(history)
    out = []
    for name, mp in CHARTS.items():
        for x in last['win']:
            if x in mp:
                out.append({'chart': name, 'from': x, 'from_kind': 'win', 'to': mp[x],
                             'transfer_rate': rates[name], 'self_pointer': mp[x] == x})
        for x in last['mach']:
            if x in mp:
                out.append({'chart': name, 'from': x, 'from_kind': 'machine', 'to': mp[x],
                             'transfer_rate': rates[name], 'self_pointer': mp[x] == x})
    out.sort(key=lambda e: -e['transfer_rate'])
    return out


def explain_charts(history, number):
    """Per-number evidence for the 'charts' strategy (see classic.get_scores): which
    chart(s) point to `number` from the last draw's numbers (with each chart's measured
    transfer rate), plus the recency-weighted-frequency baseline that formula blends in
    alongside the chart score (50/50, via classic.get_scores's 'charts' branch)."""
    hits = [p for p in chart_pointers_for_last_draw(history) if p['to'] == number]
    stat_facts = stat_explain(history, number, 'recent')
    parts = []
    non_self = [p for p in hits if not p['self_pointer']]
    if non_self:
        best = max(non_self, key=lambda p: p['transfer_rate'])
        parts.append(f"{best['from']} ({best['from_kind']} number, current draw) points to {number} via "
                      f"the {best['chart']} chart, measured at {best['transfer_rate']:.2%} historically "
                      f"(chance = 5.56%)"
                      + (f"; {len(non_self)-1} other chart pointer(s) also land here." if len(non_self) > 1 else "."))
    elif hits:
        parts.append(f"{number} only has a SELF chart pointer here (a chart maps it to itself) -- disclosed "
                      f"but excluded from its score, since crediting a number for pointing at itself isn't a "
                      f"cross-number relationship.")
    else:
        parts.append(f"No chart in the current draw points to {number}.")
    parts.append(f"Recency-weighted frequency baseline: {stat_facts['wfreq']:.3f} (drawn "
                  f"{stat_facts['freq_all']} times total) -- 'charts' blends this baseline 50/50 with the "
                  f"chart-pointer score above.")
    parts.append("None of this is causal for an independent random draw -- see the Methodology tab.")
    return {'chart_hits': hits, 'stat_facts': stat_facts, 'narrative': " ".join(parts)}


def confidence_label(trials):
    """A plain-language reliability tag for a trial count -- lottery-paper 'patterns'
    are frequently asserted from a handful of coincidences; this makes the actual
    sample size behind any given number impossible to miss."""
    if trials == 0:
        return "no historical evidence"
    if trials < 5:
        return f"very low confidence ({trials} historical instance{'s' if trials != 1 else ''})"
    if trials < 15:
        return f"low confidence ({trials} instances)"
    if trials < 40:
        return f"moderate confidence ({trials} instances)"
    return f"more data ({trials} instances) -- still not a causal signal for independent draws"


def _number_mach_events(history, number, lookahead=None):
    """Every historical instance where `number` was drawn as a machine number, with
    whether/when it then appeared as a winning number -- the literal occurrences behind
    mach_to_win_affinity's aggregate rate for this one number. An occurrence in the most
    recent draw is excluded: it has no follow-up window yet, so it isn't a resolved
    "miss" (see features.mach_to_win_affinity's matching fix)."""
    lookahead = lookahead or feat_mod.MACH_LOOKAHEAD
    win_sets = [set(d['win']) for d in history]
    n = len(history)
    events = []
    for i, d in enumerate(history):
        if number in d['mach'] and i + 1 < n:
            hit_lag, hit_date = None, None
            for lag, j in enumerate(range(i + 1, min(i + 1 + lookahead, n)), start=1):
                if number in win_sets[j]:
                    hit_lag, hit_date = lag, history[j]['date']
                    break
            events.append({'date': d['date'], 'hit': hit_lag is not None, 'lag': hit_lag, 'hit_date': hit_date})
    return events


def _number_pattern_trace_hits(history, number, group_sizes=(2, 3), lookahead=None, field='win'):
    """Every historical group-repeat (pair or triplet, of the current draw's groups)
    that was followed by `number` winning, with the positions involved, plus the total
    repeat count found (for confidence context). `groups` lists every seed group that
    matched at that historical repeat (a single historical draw overlapping the current
    one by 3+ numbers can satisfy several seed pairs/triplets at once -- that's still
    one historical repeat, not several, see transform_engine.pattern_trace_events).
    `field='mach'` gives the machine-seeded variant."""
    all_events = te.pattern_trace_events(history, group_sizes=group_sizes,
                                          lookahead=lookahead or te.TRACE_LOOKAHEAD, field=field)
    hits = []
    for e in all_events:
        for f in e['followups']:
            if number in f['win']:
                hits.append({
                    'groups': e['groups'], 'group_sizes': [len(g) for g in e['groups']],
                    'seed_positions_then': e['seed_positions_then'], 'seed_positions_now': e['seed_positions_now'],
                    'repeated_on': e['repeat_date'], 'lag': f['lag'], 'followup_date': f['date'],
                    'followup_win': f['win'], 'landed_at_position': f['hit_positions'].get(number),
                })
    return hits, len(all_events)


def _number_trend_hits(history, number):
    """The most-similar historical trend windows whose immediate follow-up draw
    contained `number` -- the literal evidence behind its 'trend' score -- plus the
    full top-match list length and total windows considered, for confidence context."""
    events, total_windows = ta.trend_match_events(history)
    hits = [e for e in events if number in e['followup_win']]
    return hits, len(events), total_windows


def _number_conditional_hits(history, last, number):
    """Which of the current draw's conditions (sum band, parity, span band,
    repeat-from-previous) credit `number`, each with its measured next-draw rate under
    that condition and the trial count behind it."""
    if not last:
        return []
    _, detail = ta.conditional_score(history, last)
    return detail.get(number, [])


def _number_yearly_hits(history, number):
    """The years in which `number` appeared inside the same calendar window as the
    upcoming draw, with the eligible-year count and the per-window chance rate --
    the anniversary-recurrence evidence."""
    _, meta = ta.yearly_score(history)
    h, e = meta['rates'].get(number, (0, 0))
    return {'hit_years': meta['detail'].get(number, []), 'hits': h, 'eligible_years': e,
            'mean_window_draws': meta['mean_window_draws'],
            'chance_per_window': meta['chance_per_window']}


def _number_cross_game_hits(history, number, all_draws):
    """Which other games' draws since this game's last draw contain `number`, with each
    source game's measured transfer rate into this game -- the cross-game evidence.
    Empty when no all_draws was provided (component absent)."""
    if not all_draws or not history:
        return []
    last = history[-1]
    _, detail = ta.cross_game_score(all_draws, last['code'], last['date'])
    return detail.get(number, [])


def _number_transform_hits(history, last, number, rule_rates=None):
    """Which transform rules, applied to which groups of the current draw, derive
    `number` -- and how many *independent* rules agree (the 'consistency across
    multiple transformation paths' signal)."""
    if not last:
        return [], 0
    rates = rule_rates if rule_rates is not None else te.measure_rule_rates(history)
    _, detail = te.transform_score(history, last, rates)
    hits = detail.get(number, [])
    distinct_rules = len({h['rule'] for h in hits})
    return hits, distinct_rules


def _number_class_hit(history, last, number, classify, rates=None):
    """Which class (terminal digit or digital-root group) the current draw shares with
    `number`, and that class's own measured (Wilson-shrunk) reliability -- the evidence
    behind a terminal/group score. A number belongs to exactly one class at a time, so
    at most one hit is possible (unlike transform, where several rules can agree)."""
    if not last:
        return None
    _, detail = te.class_carryover_score(history, last, classify, rates)
    hits = detail.get(number, [])
    return hits[0] if hits else None


def _contribution_breakdown(history, number, weights=None, all_draws=None):
    """How much each component actually contributed to `number`'s final blended score,
    as a percentage of the total -- literally which piece of evidence is driving the
    recommendation, not just each component's raw (differently-scaled) value. Mirrors
    pattern_scores()'s per-number component exclusion (see _self_exclusion) so this
    breakdown matches what actually happened in the score, not the un-excluded weights."""
    weights = weights if weights is not None else dynamic_weights(history, all_draws=all_draws)
    comp = component_scores(history, all_draws=all_draws)
    norm = {name: normalize(scores) for name, scores in comp.items()}
    excluded_here = _self_exclusion(history).get(number, set())
    # Mirror blend_scores: only weights for components actually present participate
    # (e.g. 'cross_game' is absent entirely when no all_draws was provided).
    present = {name: w for name, w in weights.items() if name in comp}
    local_weights = {name: w for name, w in present.items() if name not in excluded_here} or present
    total_w = sum(local_weights.values()) or 1.0
    contrib = {name: ((local_weights[name] / total_w) * norm[name].get(number, 0.0) if name in local_weights else 0.0)
               for name in comp}
    total = sum(contrib.values()) or 1e-9
    return {name: {'raw': comp[name].get(number, 0.0), 'normalized': norm[name].get(number, 0.0),
                    'weight': (local_weights[name] / total_w) if name in local_weights else 0.0,
                    'contribution_pct': contrib[name] / total * 100,
                    'excluded': name in excluded_here}
            for name in comp}


def _fmt_group(g):
    return str(g[0]) if len(g) == 1 else str(tuple(g))


def _build_narrative(number, n_draws, freq_info, chart_hits, mach_events,
                      trace_hits, trace_trials, tform_hits, distinct_rules,
                      pos_rate, pos_trials, pos_landing,
                      lap_rate, lap_trials, terminal_hit, group_hit,
                      trend_hits=(), trend_top=0, trend_total=0,
                      cond_hits=(), cross_hits=(),
                      mach_trace_hits=(), mach_trace_trials=0, yearly_info=None):
    parts = []
    if freq_info['last_seen_gap'] is not None:
        parts.append(f"{number} has been drawn {freq_info['freq_all']} times in this game's "
                      f"{n_draws}-draw history ({freq_info['freq_30']} of the last 30), last seen "
                      f"{freq_info['last_seen_gap']} draws ago on {freq_info['last_seen_date']}.")
    else:
        parts.append(f"{number} has never been drawn in this game's {n_draws}-draw recorded history.")

    if mach_events:
        hits = sum(1 for e in mach_events if e['hit'])
        chance = 1 - (85 / 90) ** feat_mod.MACH_LOOKAHEAD
        parts.append(f"It has appeared as a MACHINE number {len(mach_events)} time(s); "
                      f"{hits}/{len(mach_events)} were followed by it winning within "
                      f"{feat_mod.MACH_LOOKAHEAD} draws ({hits/len(mach_events):.0%} vs. {chance:.1%} "
                      f"expected by chance -- {confidence_label(len(mach_events))}).")
    else:
        parts.append("It has never appeared as a machine number in this history.")

    chart_non_self = [p for p in chart_hits if not p['self_pointer']]
    if chart_non_self:
        best = max(chart_non_self, key=lambda p: p['transfer_rate'])
        parts.append(f"{best['from']} ({best['from_kind']} number, current draw) points to {number} via "
                      f"the {best['chart']} chart, which transfers at a measured {best['transfer_rate']:.2%} "
                      f"historically (chance = 5.56%)"
                      + (f"; {len(chart_non_self)-1} other chart pointer(s) also land here." if len(chart_non_self) > 1 else "."))
    elif chart_hits:
        parts.append(f"{number} only has a SELF chart pointer here (a chart maps it to itself) -- disclosed "
                      f"but excluded from its score (see the Methodology tab).")
    else:
        parts.append("No chart in the current draw points to this number.")

    if trace_hits:
        ex = trace_hits[0]
        pos_note = ""
        if ex['seed_positions_then'] == ex['seed_positions_now']:
            pos_note = " (the group(s) held the *same* sorted-rank position(s) then as now)"
        groups_str = " & ".join(_fmt_group(g) for g in ex['groups'])
        parts.append(f"Pattern-tracing (pairs + triplets) found {len(trace_hits)} historical case(s) "
                      f"(out of {trace_trials} total historical repeats of this draw's pairs/triplets, each "
                      f"repeated historical draw counted once even if it satisfies more than one seed group) "
                      f"where a repeat was followed by {number} winning -- e.g. {groups_str} repeated together "
                      f"on {ex['repeated_on']}{pos_note}, and {number} appeared {ex['lag']} draw(s) later on "
                      f"{ex['followup_date']} at position {ex['landed_at_position']} ({confidence_label(trace_trials)}).")
    else:
        parts.append(f"Pattern-tracing found no historical case (across {trace_trials} group-repeats found) "
                      f"where a repeat of this draw's pairs/triplets was followed by {number} winning.")

    if tform_hits:
        ex = max(tform_hits, key=lambda h: h['weight'])
        rule_names = ", ".join(sorted({h['rule'] for h in tform_hits}))
        trans_conf = confidence_label(ex['trials'])
        parts.append(f"{distinct_rules} independent transform rule(s) derive {number} from the current "
                      f"draw's numbers ({rule_names}) -- best evidence: '{ex['rule']}' on {_fmt_group(ex['group'])} "
                      f"at a measured {ex['rate']:.2%} (n={ex['trials']}, chance ~5.56%, {trans_conf}), shrunk to a "
                      f"{ex['weight']:.2%} score weight to account for sample size. Agreement across multiple "
                      f"transform paths is a weak consistency signal, not proof.")
    else:
        parts.append("No transform rule in the registry (doubling/mirror/one-up/one-down over "
                      "individual numbers, sum/difference over pairs, sum/mean over triplets) derives "
                      "this number from the current draw's numbers.")

    if trend_hits:
        ex = trend_hits[0]
        shared = f" (sharing {len(ex['shared_numbers'])} literal number(s) with the current trend)" \
            if ex['shared_numbers'] else ""
        parts.append(f"Trend-similarity: of the {trend_top} historical windows most similar to the "
                      f"current trend (out of {trend_total} considered), {len(trend_hits)} were followed "
                      f"immediately by {number} winning -- most similar: the window ending "
                      f"{ex['window_end']} (similarity {ex['similarity']:.2f}: structural "
                      f"{ex['struct_sim']:.2f}, shared-numbers {ex['jaccard']:.2f}){shared}, followed by "
                      f"{number} on {ex['followup_date']}. Similarity credits what followed similar past "
                      f"trends; it is not evidence the next draw depends on them.")
    elif trend_top:
        parts.append(f"Trend-similarity: none of the {trend_top} most similar historical windows "
                      f"(of {trend_total} considered) was followed by {number} winning.")

    if cond_hits:
        best = max(cond_hits, key=lambda h: h['weight'])
        conds_str = ", ".join(f"{h['dim']}={h['value']}" for h in cond_hits)
        chance = ta.CHANCE_PER_NUMBER
        parts.append(f"Conditional rates: under the current draw's conditions ({conds_str}), {number} "
                      f"has historically appeared in the NEXT draw at best {best['rate']:.1%} "
                      f"(n={best['trials']} draws under {best['dim']}={best['value']}, chance = "
                      f"{chance:.1%}, {confidence_label(best['trials'])}), Wilson-shrunk to "
                      f"{best['weight']:.2%} before scoring.")

    if cross_hits:
        best = max(cross_hits, key=lambda h: h['weight'])
        srcs = ", ".join(sorted({h['source_game'] for h in cross_hits}))
        parts.append(f"Cross-game: {number} was drawn this week in {srcs} -- numbers from "
                      f"{best['source_game']} draws have historically transferred into this game's next "
                      f"draw at {best['rate']:.2%} (n={best['trials']}, chance = 5.56%, "
                      f"{confidence_label(best['trials'])}).")

    if mach_trace_hits:
        ex = mach_trace_hits[0]
        groups_str = " & ".join(_fmt_group(g) for g in ex['groups'])
        parts.append(f"Machine-trace: {len(mach_trace_hits)} historical repeat(s) of the current draw's "
                      f"MACHINE pairs/triplets (out of {mach_trace_trials} found) were followed by "
                      f"{number} winning -- e.g. machine group {groups_str} repeated on "
                      f"{ex['repeated_on']}, and {number} won {ex['lag']} draw(s) later "
                      f"({confidence_label(mach_trace_trials)}).")

    if yearly_info and yearly_info['eligible_years']:
        h, e = yearly_info['hits'], yearly_info['eligible_years']
        recent_years = ", ".join(str(y) for y, _ in yearly_info['hit_years'][-4:])
        parts.append(f"Yearly (anniversary window): {number} has appeared within a week of this same "
                      f"calendar date in {h} of the {e} year(s) with data ({h/e:.0%} vs "
                      f"{yearly_info['chance_per_window']:.0%} expected by chance for a "
                      f"{yearly_info['mean_window_draws']:.1f}-draw window"
                      + (f"; most recently {recent_years}" if h else "")
                      + f") -- {confidence_label(e)}.")

    if pos_trials:
        pos_chance = 1 - (85 / 90) ** te.TRACE_LOOKAHEAD
        landing_note = ""
        if pos_landing:
            modal_pos, modal_n = max(pos_landing.items(), key=lambda kv: kv[1])
            landing_total = sum(pos_landing.values())
            landing_note = (f" When it has carried over, it most often landed at sorted-rank position "
                             f"{modal_pos} ({modal_n}/{landing_total} time(s)).")
        parts.append(f"Positional stickiness (diagnostic only, excluded from the score -- see the Methodology "
                      f"tab): numbers landing in this candidate's most relevant sorted-rank position have "
                      f"historically carried over (to any later draw within {te.TRACE_LOOKAHEAD} weeks) at "
                      f"{pos_rate:.1%} (n={pos_trials}) vs. {pos_chance:.1%} expected by chance "
                      f"({confidence_label(pos_trials)}).{landing_note}")

    if lap_trials:
        lap_chance = 1 - (85 / 90) ** te.LAP_LOOKAHEAD
        parts.append(f"Lap (diagnostic only, excluded from the score): the physical draw-slot {number} "
                      f"occupied in the last draw has, specifically in the very NEXT draw (lag=1), carried "
                      f"over at {lap_rate:.1%} (n={lap_trials}) vs. {lap_chance:.1%} expected by chance "
                      f"({confidence_label(lap_trials)}).")

    if terminal_hit:
        if terminal_hit['self_credit']:
            parts.append(f"Terminal: {number} is itself one of the current draw's numbers, so its own "
                          f"'{terminal_hit['class']}s' terminal membership is disclosed here but excluded "
                          f"from its score -- crediting a number for sharing a group with itself isn't a "
                          f"cross-number relationship.")
        else:
            t_chance = 1 - (85 / 90) ** (terminal_hit['class_size'] * te.TRACE_LOOKAHEAD)
            parts.append(f"Terminal: {number} shares the '{terminal_hit['class']}s' terminal (last digit) "
                          f"with one of the current draw's numbers -- some OTHER member of that "
                          f"{terminal_hit['class_size']}-number terminal has historically reappeared within "
                          f"{te.TRACE_LOOKAHEAD} draws at {terminal_hit['rate']:.1%} (n={terminal_hit['trials']}) "
                          f"vs. {t_chance:.1%} expected by chance, shrunk to a {terminal_hit['weight']:.1%} "
                          f"score weight ({confidence_label(terminal_hit['trials'])}).")

    if group_hit:
        if group_hit['self_credit']:
            parts.append(f"Group (digital root): {number} is itself one of the current draw's numbers, so "
                          f"its own digital-root group {group_hit['class']} membership is disclosed here but "
                          f"excluded from its score, for the same reason as terminal above.")
        else:
            g_chance = 1 - (85 / 90) ** (group_hit['class_size'] * te.TRACE_LOOKAHEAD)
            parts.append(f"Group (digital root): {number} shares digital-root group {group_hit['class']} with "
                          f"one of the current draw's numbers -- some OTHER member of that "
                          f"{group_hit['class_size']}-number group has historically reappeared within "
                          f"{te.TRACE_LOOKAHEAD} draws at {group_hit['rate']:.1%} (n={group_hit['trials']}) vs. "
                          f"{g_chance:.1%} expected by chance, shrunk to a {group_hit['weight']:.1%} score "
                          f"weight ({confidence_label(group_hit['trials'])}).")

    parts.append("None of this is causal for an independent random draw -- see the Methodology tab.")
    return " ".join(parts)


def explain(history, number, lookahead=None, all_draws=None):
    """Rich, fully-auditable breakdown of why `number` is (or isn't) favored by the
    pattern analysis: the literal historical instances behind every component
    (including which sorted-rank positions were involved and which of several
    transform rules agree), a contribution breakdown showing which evidence actually
    drives the blended score, and a plain-language narrative stitching it together.
    Pass `all_draws` (as pattern_scores' callers do) so the cross-game evidence and
    contribution breakdown match the actual score."""
    last = history[-1] if history else None
    n = len(history)

    win_seen = [i for i, d in enumerate(history) if number in d['win']]
    freq_info = {
        'freq_all': len(win_seen),
        'freq_30': sum(1 for i in win_seen if i >= n - 30),
        'last_seen_gap': (n - 1 - win_seen[-1]) if win_seen else None,
        'last_seen_date': history[win_seen[-1]]['date'] if win_seen else None,
    }

    chart_hits = [p for p in chart_pointers_for_last_draw(history) if p['to'] == number]
    mach_events = _number_mach_events(history, number, lookahead)
    trace_hits, trace_trials = _number_pattern_trace_hits(history, number, lookahead=lookahead)
    rule_rates = te.measure_rule_rates(history) if last else {}
    tform_hits, distinct_rules = _number_transform_hits(history, last, number, rule_rates)
    trend_hits, trend_top, trend_total = _number_trend_hits(history, number)
    cond_hits = _number_conditional_hits(history, last, number)
    cross_hits = _number_cross_game_hits(history, number, all_draws)
    mach_trace_hits, mach_trace_trials = _number_pattern_trace_hits(history, number,
                                                                    lookahead=lookahead, field='mach')
    yearly_info = _number_yearly_hits(history, number)
    contribution = _contribution_breakdown(history, number, all_draws=all_draws)

    pos_rate, pos_trials, pos_landing = 0.0, 0, {}
    draw_order_rate, draw_order_trials = 0.0, 0
    lap_rate, lap_trials = 0.0, 0
    if last and number in last['win']:
        pos_rates, landing = te.positional_carryover_rates(history)
        p = te.positions_of(last['win'])[number]
        pos_rate, pos_trials = pos_rates.get(p, (0.0, 0))
        pos_landing = dict(landing.get(p, {}))

        # Second, independent positional lens: as-drawn slot instead of sorted-rank
        # (diagnostic only -- not blended into any score, see transform_engine's
        # draw_order_carryover_rates docstring for why).
        do_rates, _do_landing = te.draw_order_carryover_rates(history)
        dp = te.draw_order_positions(last['win'])[number]
        draw_order_rate, draw_order_trials = do_rates.get(dp, (0.0, 0))

        # 'Lap': the lag=1-only special case of the as-drawn-order lens.
        lap_rates, _lap_landing = te.lap_carryover_rates(history)
        lap_rate, lap_trials = lap_rates.get(dp, (0.0, 0))

    # Terminal (last-digit) and group (digital-root) hits -- unlike positional/lap,
    # these credit ANY number sharing a class with the current draw, not just the 5
    # numbers just drawn, so they're computed for every candidate, not gated on
    # `number in last['win']`.
    terminal_hit = _number_class_hit(history, last, number, te.terminal_of)
    group_hit = _number_class_hit(history, last, number, te.group_of)

    narrative = _build_narrative(number, n, freq_info, chart_hits, mach_events,
                                  trace_hits, trace_trials, tform_hits, distinct_rules,
                                  pos_rate, pos_trials, pos_landing,
                                  lap_rate, lap_trials, terminal_hit, group_hit,
                                  trend_hits=trend_hits, trend_top=trend_top, trend_total=trend_total,
                                  cond_hits=cond_hits, cross_hits=cross_hits,
                                  mach_trace_hits=mach_trace_hits, mach_trace_trials=mach_trace_trials,
                                  yearly_info=yearly_info)

    return {
        'number': number,
        'narrative': narrative,
        # Trend-similarity, conditional, cross-game, and machine-trace evidence -- see
        # trend_analysis.py and transform_engine.pattern_trace_events(field='mach').
        'trend_hits': trend_hits,
        'trend_top_matches': trend_top,
        'trend_total_windows': trend_total,
        'conditional_hits': cond_hits,
        'cross_game_hits': cross_hits,
        'mach_trace_hits': mach_trace_hits,
        'mach_trace_total_repeats': mach_trace_trials,
        'mach_trace_confidence': confidence_label(mach_trace_trials),
        'yearly_info': yearly_info,
        'frequency': freq_info,
        'contribution': contribution,
        'chart_hits': chart_hits,
        'machine_number_events': mach_events,
        'machine_number_confidence': confidence_label(len(mach_events)),
        'pattern_trace_hits': trace_hits,
        'pattern_trace_total_repeats': trace_trials,
        'pattern_trace_confidence': confidence_label(trace_trials),
        'transform_hits': tform_hits,
        'transform_distinct_rules': distinct_rules,
        'positional_rate': pos_rate,
        'positional_trials': pos_trials,
        'positional_confidence': confidence_label(pos_trials),
        'positional_chance': 1 - (85 / 90) ** te.TRACE_LOOKAHEAD,
        'positional_landing': pos_landing,
        # Diagnostic-only as-drawn-order lens (see module docstring) -- not part of the
        # blended score, surfaced purely so it can be inspected/compared to sorted-rank.
        'draw_order_rate': draw_order_rate,
        'draw_order_trials': draw_order_trials,
        'draw_order_confidence': confidence_label(draw_order_trials),
        # 'Lap' (lag=1 as-drawn-slot carryover), and terminal/group (digital-root)
        # class-carryover hits -- see transform_engine's lap_score/terminal_score/
        # group_score for what each measures.
        'lap_rate': lap_rate,
        'lap_trials': lap_trials,
        'lap_confidence': confidence_label(lap_trials),
        'terminal_hit': terminal_hit,
        'group_hit': group_hit,
    }


def combination_diagnostics(history, picks):
    """Purely descriptive: how the *combination* actually picked (as opposed to each
    number's own individual history, which every component above already covers)
    compares to what real 5-number draws in this game's history look like -- sum,
    odd/even split, and spacing profile.

    This never feeds back into any score. For a genuinely random 5/90 draw, no
    combination is more or less likely than any other, so nudging picks toward "looking
    typical" would manufacture a false edge -- exactly the overfitting tradeoff this
    project's fixed-weight, never-auto-tuned design (see ensemble.py) exists to avoid.
    It exists only so a reader can see whether the picked combination is a structural
    outlier relative to real history, not to imply that being 'typical' predicts a win."""
    sums = [sum(d['win']) for d in history]
    odd_counts = [sum(1 for x in d['win'] if x % 2 == 1) for d in history]
    spans = [max(d['win']) - min(d['win']) for d in history]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    profile = te.spacing_profile(picks)
    return {
        'picks': sorted(picks),
        'sum': sum(picks), 'history_mean_sum': mean(sums),
        'odd_count': sum(1 for x in picks if x % 2 == 1), 'history_mean_odd_count': mean(odd_counts),
        'span': profile['span'], 'history_mean_span': mean(spans),
        'gaps': profile['gaps'], 'decade_buckets_used': profile['decade_buckets_used'],
        'n_historical_draws': len(history),
    }
