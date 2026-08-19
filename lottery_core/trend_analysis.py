"""Trend-similarity, conditional, and cross-game analysis -- three further
lottery-paper reading strategies, formalized the same way transform_engine.py
formalizes the others: measured from history, Wilson-shrunk where a rate drives a
score, and fully disclosable via evidence structures for pattern_analysis.explain().

1. TREND-WINDOW SIMILARITY (trend_match_events / trend_match_score) -- the "trace the
   new trend back through the old papers" strategy: profile the current run of recent
   draws (their sums, spans, parity, decade/terminal distribution, internal repeats AND
   the literal numbers involved), find the historical windows that looked most similar,
   and see what actually won immediately after each of those windows. Scoring credits
   the numbers that followed similar windows, weighted by how similar each window was.
   Historical windows are required NOT to overlap the current window, and their
   follow-up draw must itself fall strictly before the current window -- otherwise the
   most similar windows are trivially the ones sharing draws with the present, and the
   component degenerates into echoing the current draws back out (the exact bias
   documented and fixed for positional/lap, see pattern_analysis's module docstring).

2. CONDITIONAL NUMBER RATES (conditional_number_rates / conditional_score) -- "the same
   pattern behaves differently under different conditions": classify every historical
   draw along four named, pre-declared condition dimensions (sum band, parity profile,
   span band, whether it repeated a number from its predecessor), and measure each
   number's appearance rate in the draw that followed each condition. Scoring looks up
   the CURRENT draw's condition on each dimension and credits numbers by their measured
   next-draw rate under that same condition, Wilson-shrunk by the condition's trial
   count. Band boundaries (sum/span terciles) are derived from the passed history only,
   so walk-forward evaluation never peeks ahead. The dimensions are fixed and declared
   here rather than searched over -- mining many candidate conditions for whichever
   looked best would manufacture an edge out of noise (see ensemble.py's stance).

3. CROSS-GAME TRANSFER (cross_game_rates / cross_game_score) -- the classic "today's
   results point at tomorrow's game" reading: for a target game, measure per source
   game how often a number from the source's most recent draw (within the last 7 days)
   appeared in the target's next draw, then credit the numbers in the OTHER games'
   draws since the target game's last draw, weighted by each source game's own
   measured, shrunk transfer rate. Chance level per number is 5/90 = 5.56%, same as the
   charts -- and measured the same honest way.

None of these is expected to beat chance for independent random draws -- see the
Methodology tab. They exist so the reading strategies people actually use are encoded,
measured, and auditable instead of asserted.
"""
from bisect import bisect_left
from collections import Counter, defaultdict

from .transform_engine import _wilson_lower_bound

TREND_WINDOW = 3       # draws per window ("the recent trend")
TREND_TOP_N = 25       # how many most-similar historical windows get a vote
STRUCT_SIM_WEIGHT = 0.6  # structural-profile similarity vs...
JACCARD_WEIGHT = 0.4     # ...literal shared-numbers similarity (disclosed split)
CROSS_GAME_WINDOW_DAYS = 7  # a source draw older than this can't "point at" a target draw
CHANCE_PER_NUMBER = 5 / 90  # 5.56%: P(a specific number appears in a 5/90 draw)


# ---------------------------------------------------------------- condition dimensions
def _tercile_bounds(values):
    """(lower, upper) tercile cut points of `values` -- band boundaries derived from
    the data actually passed in (a walk-forward caller passes only past draws, so
    these never leak future information)."""
    if not values:
        return (0, 0)
    s = sorted(values)
    return (s[len(s) // 3], s[(2 * len(s)) // 3])


def _band(value, bounds, labels=('low', 'mid', 'high')):
    lo, hi = bounds
    if value <= lo:
        return labels[0]
    if value <= hi:
        return labels[1]
    return labels[2]


def _parity_band(win):
    odd = sum(1 for x in win if x % 2 == 1)
    if odd <= 1:
        return 'even_heavy'
    if odd >= 4:
        return 'odd_heavy'
    return 'balanced'


def draw_conditions(draw, prev_draw, sum_bounds, span_bounds):
    """The named condition profile of one draw: four fixed, pre-declared dimensions.
    `repeat_prev` is None when there's no predecessor to compare against (first draw),
    and that dimension is simply skipped for that transition rather than guessed."""
    win = draw['win']
    return {
        'sum_band': _band(sum(win), sum_bounds),
        'parity': _parity_band(win),
        'span_band': _band(max(win) - min(win), span_bounds, labels=('tight', 'mid', 'wide')),
        'repeat_prev': (None if prev_draw is None else
                        ('repeat' if set(win) & set(prev_draw['win']) else 'no_repeat')),
    }


def conditional_number_rates(history):
    """For every condition dimension/value: each number's measured appearance rate in
    the draw immediately FOLLOWING a draw with that condition, plus the trial count
    (how many such transitions history actually contains). Returns
    ({dim: {value: ({number: hits}, trials)}}, (sum_bounds, span_bounds))."""
    n = len(history)
    if n < 3:
        return {}, ((0, 0), (0, 0))
    sums = [sum(d['win']) for d in history[:-1]]
    spans = [max(d['win']) - min(d['win']) for d in history[:-1]]
    sum_bounds = _tercile_bounds(sums)
    span_bounds = _tercile_bounds(spans)
    hits = defaultdict(lambda: defaultdict(Counter))  # dim -> value -> Counter(number)
    trials = defaultdict(Counter)                     # dim -> Counter(value)
    for i in range(n - 1):
        conds = draw_conditions(history[i], history[i - 1] if i > 0 else None,
                                sum_bounds, span_bounds)
        nxt = history[i + 1]['win']
        for dim, val in conds.items():
            if val is None:
                continue
            trials[dim][val] += 1
            for x in nxt:
                hits[dim][val][x] += 1
    out = {dim: {val: (dict(hits[dim][val]), trials[dim][val]) for val in trials[dim]}
           for dim in trials}
    return out, (sum_bounds, span_bounds)


def conditional_score(history, last, rates_and_bounds=None):
    """Score every number by its measured next-draw appearance rate under the CURRENT
    draw's condition, per dimension, Wilson-shrunk by that condition's trial count and
    averaged across the dimensions that apply. Returns (scores, detail) in the same
    shape as transform_score: detail[k] lists, per dimension, the condition matched,
    the raw measured rate, the trial count, and the shrunk weight actually used."""
    scores = {k: 0.0 for k in range(1, 91)}
    detail = defaultdict(list)
    if not last or len(history) < 3:
        return scores, detail
    rates, (sum_bounds, span_bounds) = (rates_and_bounds if rates_and_bounds is not None
                                        else conditional_number_rates(history))
    if not rates:
        return scores, detail
    prev = history[-2] if len(history) >= 2 else None
    conds = draw_conditions(last, prev, sum_bounds, span_bounds)
    dims_used = 0
    per_dim = []
    for dim, val in conds.items():
        if val is None or dim not in rates or val not in rates[dim]:
            continue
        num_hits, trials = rates[dim][val]
        if trials <= 0:
            continue
        dims_used += 1
        per_dim.append((dim, val, num_hits, trials))
    if not dims_used:
        return scores, detail
    for dim, val, num_hits, trials in per_dim:
        for k in range(1, 91):
            rate = num_hits.get(k, 0) / trials
            weight = _wilson_lower_bound(rate, trials)
            scores[k] += weight / dims_used
            if rate > 0:
                detail[k].append({'dim': dim, 'value': val, 'rate': rate,
                                  'trials': trials, 'weight': weight})
    return scores, detail


def joint_condition_key(draw, prev_draw, sum_bounds, span_bounds):
    """The FULL condition profile of a draw as a single hashable key -- the strict
    'all conditions match at once' identity used by joint_conditional_rates."""
    c = draw_conditions(draw, prev_draw, sum_bounds, span_bounds)
    return (c['sum_band'], c['parity'], c['span_band'], c['repeat_prev'])


def joint_conditional_rates(history):
    """STRICT (joint-cohort) version of conditional_number_rates, for the spatial
    engine: every historical draw is classified by its full 4-tuple of conditions
    (sum band AND parity AND span band AND repeat-from-previous), and each number's
    appearance rate in the following draw is measured strictly within each exact
    cohort. Where conditional_number_rates averages the dimensions independently
    (marginals, ~100-450 trials each), this matches all of them at once -- the
    precursor-matching idea of NCC template matching applied to declared features
    instead of the raw binary image. The price of strictness is cohort size (54
    possible cells over a few hundred draws), which is why every rate is
    Wilson-shrunk by its cohort's own trial count before it may score, and the cohort
    size is disclosed everywhere. Returns ({key: ({number: hits}, trials)},
    (sum_bounds, span_bounds))."""
    n = len(history)
    if n < 4:
        return {}, ((0, 0), (0, 0))
    sums = [sum(d['win']) for d in history[:-1]]
    spans = [max(d['win']) - min(d['win']) for d in history[:-1]]
    sum_bounds = _tercile_bounds(sums)
    span_bounds = _tercile_bounds(spans)
    hits = defaultdict(Counter)
    trials = Counter()
    for i in range(1, n - 1):  # start at 1: repeat_prev needs a predecessor
        key = joint_condition_key(history[i], history[i - 1], sum_bounds, span_bounds)
        trials[key] += 1
        for x in history[i + 1]['win']:
            hits[key][x] += 1
    return ({key: (dict(hits[key]), trials[key]) for key in trials},
            (sum_bounds, span_bounds))


def joint_conditional_score(history, last, rates_and_bounds=None):
    """Score every number by its measured next-draw rate strictly within the cohort of
    historical draws whose FULL condition profile matches the current draw's,
    Wilson-shrunk by the cohort size. Returns (scores, meta): meta carries the joint
    condition key, the cohort size, and per-number detail (appearances, raw rate,
    shrunk weight) for the evidence panel."""
    scores = {k: 0.0 for k in range(1, 91)}
    meta = {'key': None, 'cohort': 0, 'detail': defaultdict(list)}
    if not last or len(history) < 4:
        return scores, meta
    rates, (sum_bounds, span_bounds) = (rates_and_bounds if rates_and_bounds is not None
                                        else joint_conditional_rates(history))
    if not rates:
        return scores, meta
    key = joint_condition_key(last, history[-2], sum_bounds, span_bounds)
    num_hits, trials = rates.get(key, ({}, 0))
    meta['key'] = key
    meta['cohort'] = trials
    if not trials:
        return scores, meta
    for k in range(1, 91):
        h = num_hits.get(k, 0)
        rate = h / trials
        w = _wilson_lower_bound(rate, trials)
        scores[k] = w
        if h:
            meta['detail'][k].append({'condition': key, 'cohort': trials,
                                      'appearances': h, 'rate': rate, 'weight': w})
    return scores, meta


# ---------------------------------------------------------------- trend-window similarity
def window_profile(window_draws):
    """(feature vector, set of numbers) for a run of consecutive draws: mean sum, mean
    span, mean odd-count, internal carry-over rate, and the decade/terminal
    distributions of all its numbers -- the structural 'shape' of a trend, separate
    from the literal numbers (returned alongside for the Jaccard term)."""
    k = len(window_draws)
    nums = [x for d in window_draws for x in d['win']]
    tot = len(nums)
    decade = [0.0] * 9
    terminal = [0.0] * 10
    for x in nums:
        decade[(x - 1) // 10] += 1
        terminal[x % 10] += 1
    overlap = sum(len(set(a['win']) & set(b['win']))
                  for a, b in zip(window_draws, window_draws[1:]))
    feats = [
        sum(sum(d['win']) for d in window_draws) / k / 450.0,
        sum(max(d['win']) - min(d['win']) for d in window_draws) / k / 89.0,
        sum(1 for x in nums if x % 2 == 1) / tot,
        overlap / (5.0 * max(k - 1, 1)),
    ] + [c / tot for c in decade] + [c / tot for c in terminal]
    return feats, set(nums)


def trend_match_events(history, window=TREND_WINDOW, top_n=TREND_TOP_N):
    """The most similar non-overlapping historical windows to the current trend
    (the last `window` draws), with what won immediately after each. Returns
    (events, total_windows_considered). Each event discloses both similarity terms
    (structural profile + literal shared numbers) separately, the shared numbers
    themselves, and the follow-up draw -- the concrete evidence a trend-match score is
    built from.

    A candidate window must satisfy j + 1 <= n - 1 - window (its follow-up draw falls
    strictly BEFORE the current window): windows overlapping the present, or whose
    follow-up IS one of the current window's own draws, would make 'similarity' partly
    self-similarity and the follow-up evidence partly the current draws themselves --
    the echo failure mode documented in pattern_analysis's module docstring."""
    n = len(history)
    if n < 2 * window + 2:
        return [], 0
    cur_feats, cur_nums = window_profile(history[n - window:])
    candidates = []
    for j in range(window - 1, n - 1 - window):
        feats, nums = window_profile(history[j - window + 1:j + 1])
        candidates.append((j, feats, nums))
    if not candidates:
        return [], 0
    # z-normalize each feature across candidates + the current window, so no single
    # raw scale dominates the distance
    nfeat = len(cur_feats)
    all_rows = [f for _, f, _ in candidates] + [cur_feats]
    means = [sum(r[i] for r in all_rows) / len(all_rows) for i in range(nfeat)]
    stds = []
    for i in range(nfeat):
        var = sum((r[i] - means[i]) ** 2 for r in all_rows) / len(all_rows)
        stds.append(var ** 0.5 or 1.0)
    cur_z = [(cur_feats[i] - means[i]) / stds[i] for i in range(nfeat)]
    events = []
    for j, feats, nums in candidates:
        dist = (sum(((feats[i] - means[i]) / stds[i] - cur_z[i]) ** 2
                    for i in range(nfeat)) / nfeat) ** 0.5
        struct_sim = 1.0 / (1.0 + dist)
        union = nums | cur_nums
        jaccard = len(nums & cur_nums) / len(union) if union else 0.0
        sim = STRUCT_SIM_WEIGHT * struct_sim + JACCARD_WEIGHT * jaccard
        events.append({
            'window_start': history[j - window + 1]['date'],
            'window_end': history[j]['date'],
            'similarity': sim, 'struct_sim': struct_sim, 'jaccard': jaccard,
            'shared_numbers': sorted(nums & cur_nums),
            'followup_date': history[j + 1]['date'],
            'followup_win': history[j + 1]['win'],
        })
    events.sort(key=lambda e: -e['similarity'])
    return events[:top_n], len(candidates)


def trend_match_score(history, window=TREND_WINDOW, top_n=TREND_TOP_N, events=None):
    """Credit the numbers that won immediately after the historical windows most
    similar to the current trend, weighted by each window's similarity. Returns
    (scores, (events, total_windows)) -- the events are the full evidence trail."""
    if events is None:
        events, total = trend_match_events(history, window, top_n)
    else:
        events, total = events
    scores = {k: 0.0 for k in range(1, 91)}
    sim_total = sum(e['similarity'] for e in events)
    if sim_total > 0:
        for e in events:
            for x in e['followup_win']:
                scores[x] += e['similarity'] / sim_total
    return scores, (events, total)


# ---------------------------------------------------------------- yearly (anniversary) recurrence
# The "yearly behaviour" reading (studied from the Kaigee/Lottobrains forecasting
# app's yearly-pattern engine, re-implemented here with measured rates instead of its
# heuristic point-score confidences): does a number recur in the SAME +/-day_range-day
# calendar window across previous years? With a 64-year archive this is finally
# measurable at scale. Chance baseline: a window holding w draws gives every number a
# 1-(85/90)^w probability of appearing by luck alone (~15% for a typical 3-draw
# window) -- a "played 4 of the last 6 years around this date" claim must beat THAT,
# not zero, which is exactly what the disclosed eligible-year counts + Wilson shrink
# are for.
YEARLY_DAY_RANGE = 7


def _same_window_dates(target, year, day_range=YEARLY_DAY_RANGE):
    try:
        anchor = target.replace(year=year)
    except ValueError:  # Feb 29 in a non-leap year
        anchor = target.replace(year=year, day=target.day - 1)
    import datetime as _dt
    return anchor - _dt.timedelta(days=day_range), anchor + _dt.timedelta(days=day_range)


def yearly_rates(history, day_range=YEARLY_DAY_RANGE):
    """For each number: across every PRIOR year with any draw inside the same
    calendar window as the upcoming draw (last draw + 7 days), did the number appear
    (as a winning number) in that year's window? Returns
    ({number: (hit_years, eligible_years)}, {number: [(year, date), ...]},
    mean_window_draws) -- the literal per-year evidence included for explain()."""
    import datetime as _dt
    hits = {k: 0 for k in range(1, 91)}
    eligible = 0
    detail = defaultdict(list)
    if len(history) < 60:
        return {k: (0, 0) for k in range(1, 91)}, detail, 0.0
    target = history[-1]['date'] + _dt.timedelta(days=7)
    first_year = history[0]['date'].year
    window_draw_counts = []
    for year in range(first_year, target.year):
        lo, hi = _same_window_dates(target, year, day_range)
        window = [d for d in history if lo <= d['date'] <= hi]
        if not window:
            continue
        eligible += 1
        window_draw_counts.append(len(window))
        seen_this_year = set()
        for d in window:
            for x in d['win']:
                if x not in seen_this_year:
                    seen_this_year.add(x)
                    hits[x] += 1
                    detail[x].append((year, d['date']))
    mean_w = (sum(window_draw_counts) / len(window_draw_counts)) if window_draw_counts else 0.0
    return {k: (hits[k], eligible) for k in range(1, 91)}, detail, mean_w


def yearly_score(history, rates_pack=None, day_range=YEARLY_DAY_RANGE):
    """Score = Wilson lower bound of each number's same-window hit-year rate. Returns
    (scores, meta): meta has per-number year evidence, the eligible-year count, the
    mean window size, and the chance rate that a claim must beat."""
    rates, detail, mean_w = (rates_pack if rates_pack is not None
                             else yearly_rates(history, day_range))
    scores = {k: 0.0 for k in range(1, 91)}
    chance = 1 - (85 / 90) ** mean_w if mean_w else 0.0
    for k in range(1, 91):
        h, e = rates.get(k, (0, 0))
        scores[k] = _wilson_lower_bound(h / e if e else 0.0, e)
    return scores, {'detail': detail, 'rates': rates, 'mean_window_draws': mean_w,
                    'chance_per_window': chance}


# ---------------------------------------------------------------- counting-weeks progressions
# The "Counting Weeks" reading (also studied from the Kaigee app's worker): a number
# that appeared at draws t, t+g, t+2g, ... is expected again at the next term of the
# progression. Here every such chain's COMPLETION is measured from history first --
# of all comparable progression prefixes (bucketed by chain length: exactly 2 vs 3+),
# how often did the next term actually hit? Chance is 5.56% (the next term is one
# specific future draw), and that is precisely where the measured rates land -- the
# component exists to encode the strategy and show that measurement, not because
# periodicity is real in a memoryless game.
COUNTING_MAX_GAP = 150


def counting_week_stats(history, max_gap=COUNTING_MAX_GAP):
    """Completion rates of appearance-index progressions, bucketed by prefix length
    (2 = a single repeat pair, 3 = an established chain). For every pair of
    appearances of a number (gap g <= max_gap), every resolvable prefix of the
    maximal chain counts one trial; the prefix scores a hit iff the chain extended
    one more term. Returns {bucket: (rate, trials)}."""
    n = len(history)
    appearances = defaultdict(list)
    for i, d in enumerate(history):
        for x in d['win']:
            appearances[x].append(i)
    hits = {2: 0, 3: 0}
    trials = {2: 0, 3: 0}
    for idxs in appearances.values():
        iset = set(idxs)
        m = len(idxs)
        for a in range(m):
            for b in range(a + 1, m):
                g = idxs[b] - idxs[a]
                if g > max_gap:
                    break
                # maximal chain from (idxs[a], idxs[b]) with step g
                length = 2
                last = idxs[b]
                while last + g in iset:
                    length += 1
                    last += g
                for prefix in range(2, length + 1):
                    nxt = idxs[a] + prefix * g
                    if nxt >= n:
                        continue  # unresolved -- the next term hasn't happened yet
                    bucket = 2 if prefix == 2 else 3
                    trials[bucket] += 1
                    hits[bucket] += prefix < length or (prefix == length and last + g in iset)
    return {b: (hits[b] / trials[b] if trials[b] else 0.0, trials[b]) for b in (2, 3)}


def counting_week_score(history, stats=None, max_gap=COUNTING_MAX_GAP):
    """Credit every number whose appearance-index progression extrapolates EXACTLY to
    the next draw, at the measured, Wilson-shrunk completion rate for its chain-length
    bucket. Returns (scores, detail): detail[k] lists each firing chain (its draw
    dates, gap, length, and the measured rate/trials behind its weight)."""
    n = len(history)
    stats = stats if stats is not None else counting_week_stats(history, max_gap)
    scores = {k: 0.0 for k in range(1, 91)}
    detail = defaultdict(list)
    appearances = defaultdict(list)
    for i, d in enumerate(history):
        for x in d['win']:
            appearances[x].append(i)
    for k, idxs in appearances.items():
        iset = set(idxs)
        m = len(idxs)
        seen_chains = set()
        for a in range(m):
            for b in range(a + 1, m):
                g = idxs[b] - idxs[a]
                if g > max_gap:
                    break
                length = 2
                last = idxs[b]
                while last + g in iset:
                    length += 1
                    last += g
                if last + g != n:
                    continue  # progression doesn't land on the next draw
                chain_key = (g, last)
                if chain_key in seen_chains:
                    continue  # same maximal chain reached from an inner pair
                seen_chains.add(chain_key)
                bucket = 2 if length == 2 else 3
                rate, tr = stats.get(bucket, (0.0, 0))
                w = _wilson_lower_bound(rate, tr)
                scores[k] += w
                start = last - (length - 1) * g
                detail[k].append({'gap': g, 'chain_length': length,
                                  'chain_dates': [history[start + j * g]['date']
                                                  for j in range(length)],
                                  'rate': rate, 'trials': tr, 'weight': w})
    return scores, detail


# ---------------------------------------------------------------- cross-game transfer
def cross_game_rates(all_draws, target_game, window_days=CROSS_GAME_WINDOW_DAYS):
    """Per source game: the measured rate at which a number from the source's most
    recent draw (within `window_days` before a target-game draw) appeared in that
    target draw, with the trial count. The honest, chart-style measurement behind the
    'today's results point at tomorrow's game' reading -- chance is 5.56% per number."""
    by_game = defaultdict(list)
    for d in all_draws:
        by_game[d['code']].append(d)
    tdraws = by_game.get(target_game, [])
    rates = {}
    for src, sdraws in by_game.items():
        if src == target_game:
            continue
        hits = trials = 0
        idx = 0
        latest = None
        for t in tdraws:
            while idx < len(sdraws) and sdraws[idx]['date'] < t['date']:
                latest = sdraws[idx]
                idx += 1
            if latest is not None and (t['date'] - latest['date']).days <= window_days:
                tw = set(t['win'])
                for x in latest['win']:
                    trials += 1
                    hits += x in tw
        rates[src] = (hits / trials if trials else 0.0, trials)
    return rates


def cross_game_score(all_draws, target_game, last_target_date, rates=None):
    """Credit the numbers drawn in OTHER games since the target game's last draw
    ("this week's results so far"), each weighted by its source game's own measured,
    Wilson-shrunk transfer rate into the target game. Returns (scores, detail):
    detail[k] lists each source draw containing k with the rate/trials/weight used."""
    scores = {k: 0.0 for k in range(1, 91)}
    detail = defaultdict(list)
    if not all_draws:
        return scores, detail
    rates = rates if rates is not None else cross_game_rates(all_draws, target_game)
    latest_since = {}
    for d in all_draws:
        if d['code'] != target_game and (last_target_date is None or d['date'] > last_target_date):
            latest_since[d['code']] = d  # draws are date-sorted, so the last one wins
    for src, d in latest_since.items():
        rate, trials = rates.get(src, (0.0, 0))
        weight = _wilson_lower_bound(rate, trials)
        for x in d['win']:
            scores[x] += weight
            detail[x].append({'source_game': src, 'source_date': d['date'],
                              'rate': rate, 'trials': trials, 'weight': weight})
    return scores, detail


def slice_before(all_draws, cutoff_date, dates=None):
    """all_draws restricted to strictly before `cutoff_date` -- the walk-forward
    slice a backtest needs so cross-game scoring never sees the test draw's own week
    beyond what was actually drawn before it. Pass `dates` (the precomputed date list)
    to make repeated calls O(log n)."""
    if dates is None:
        dates = [d['date'] for d in all_draws]
    return all_draws[:bisect_left(dates, cutoff_date)]
