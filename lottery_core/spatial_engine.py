"""Spatial Pattern Matching Engine + statistical noise control.

Implements the project's spatial-engine blueprint: the lotto history is treated as a
two-channel binary image (weeks x numbers 1-90, winning + machine channels) and as a
dense positional matrix (the 10 as-drawn slots per week), and five strategy families
are run against it:

  1. NCC TEMPLATE MATCHING ("the Plan"): slide the current h-week window backwards
     through the chart, score each historical window by 2D normalized cross-correlation,
     and flag the numbers drawn in the row AFTER each top match (the historical "drop")
     as candidates. NOTE on the blueprint's 0.85 threshold: for binary 5/90 windows the
     NCC denominator makes Score = overlap/ones-per-window (e.g. /15 for h=3, win
     channel), so real charts top out around 0.3-0.4 -- a 0.85 threshold can never fire.
     The engine therefore selects top-K matches and DISCLOSES every score plus how many
     cleared any requested threshold, rather than silently returning nothing.
  2. LINEAR SPATIAL TRAJECTORIES (diagonals): measured completion rates for directional
     runs B[r,c]=1, B[r+1,c+d]=1 -> B[r+2,c+2d]=1 (per step d and run length), then
     extrapolation of every partial run that ends at the current row.
  3. POSITIONAL BOUNDARY ENCLOSURES (boxes & V-shapes): the box is scored in its
     predictive form -- a pair that co-occurred in a past row has one member in the
     current draw; the measured, Wilson-shrunk rate at which the OTHER member then
     appears within the lookahead credits that partner. V-shapes (symmetric descent/
     ascent) are detected and their completion measured honestly -- at 5/90 density
     full V's are near-nonexistent, and the component reports that rather than faking
     signal.
  4. ALGEBRAIC INVARIANTS AND KEYS: a registry of positional equations over the dense
     matrix (e.g. P[r,slot_i] + P[r,slot_j] = P[r+1,target_slot]) with the operator
     suite from the blueprint (turning/digit-inversion, mirror/counter, +/-K, double,
     pair sum/diff). Every candidate key is measured across history and screened with
     an exact binomial tail p-value against the 1/90 slot-match chance rate. The
     IDENTITY operator is deliberately excluded from scoring: an identity key is "slot
     i reappears in slot j next week" -- the lap/positional echo channel this project
     already measured and quarantined (see pattern_analysis's module docstring).
  5. MACHINE->WINNING CHANNEL CROSSOVERS: the lagged cross-correlation between the
     machine channel at t-tau and the winning channel at t, resolved per lag tau --
     the global transfer rate at each lag, applied to the numbers drawn as machine
     numbers tau draws ago.
  6. STRICT JOINT-CONDITIONAL COHORTS (trend_analysis.joint_conditional_score): every
     historical draw classified along the pre-declared dimensions (sum band, parity
     profile, span band, repeated-a-number-from-predecessor), and each number's
     appearance rate measured STRICTLY within the cohort whose full condition profile
     matches the current draw's -- all dimensions at once, not averaged marginals.
     This is the engine's declared-feature counterpart of NCC precursor matching
     (NCC matches the raw binary image; this matches the feature profile), and the
     'super' version of the pattern system's per-dimension 'conditional' component:
     strictness costs cohort size (54 possible cells over a few hundred draws), so
     every rate is Wilson-shrunk by its cohort's own trial count and the cohort size
     is disclosed everywhere.

STATISTICAL NOISE CONTROL (the blueprint's Component 2): searching ~1,500 candidate
keys guarantees spurious "winners" (at the 0.01 screen, ~15 false positives are
EXPECTED from pure noise). bootstrap_key_pvalue() re-measures a key on synthetic
charts (fresh uniform 5-of-90 rows -- density preserved, all structure destroyed) for
a per-key empirical p-value, and bootstrap_best_key_pvalue() runs the WHOLE key search
on each synthetic chart and compares the real best key against the null best-key
distribution (a max-statistic / Westfall-Young-style family-wise test) -- the honest
answer to "is my best key better than the best key pure noise would hand me?".
Bootstrapping is too expensive to run inside every walk-forward step; the backtest
uses the (walk-forward-honest) binomial screen alone, and the UI/CLI expose bootstrap
p-values for the survivors. Everything scored is Wilson-shrunk, chance-annotated, and
disclosed, exactly like the rest of lottery_core.
"""
import math
import random
from collections import Counter, defaultdict

from .transform_engine import _wilson_lower_bound, _wrap90, TRACE_LOOKAHEAD

NCC_WINDOW = 3        # h: weeks per template
NCC_TOP_N = 25        # top matches that vote
NCC_THRESHOLD = None  # optional hard threshold (see module docstring: 0.85 never fires)
DIAG_DELTAS = (1, -1, 2, -2)
KEY_SCREEN_ALPHA = 0.01   # binomial screen for keys (expect ~15 false positives/1500)
KEY_SLOT_CHANCE = 1 / 90  # P(a specific slot equals a specific value), approx.
CHANCE_PER_NUMBER = 5 / 90

# Fixed blend weights (ensemble.py's anti-auto-tuning stance -- this mode has no
# dynamic reweighting on purpose; it's the blueprint engine, evaluated as specified).
# 'periodic' is the Counting-Weeks reading (appearance-index arithmetic progressions
# extrapolated to the next draw; strategy studied from the Kaigee forecasting app's
# worker, re-implemented with MEASURED chain-completion rates instead of its heuristic
# point scores) -- see trend_analysis.counting_week_stats/counting_week_score.
SPATIAL_WEIGHTS = {'ncc': 0.22, 'diagonal': 0.13, 'box': 0.13, 'conditional': 0.14,
                   'keys': 0.13, 'mach_cross': 0.10, 'vshape': 0.05, 'periodic': 0.10}


# ---------------------------------------------------------------- representations
def binary_matrix(history, field='win'):
    """Representation A: each week as a 90-bit integer bitmask (bit k-1 = number k
    drawn) -- the sparse binary image, one channel per call ('win' or 'mach')."""
    return [sum(1 << (x - 1) for x in (d.get(field) or [])) for d in history]


def positional_matrix(history):
    """Representation B: the dense positional matrix -- each row the 10 as-drawn slots
    [W1..W5, M1..M5] (M slots None for pre-machine-era draws)."""
    rows = []
    for d in history:
        m = d.get('mach') or []
        rows.append(list(d['win']) + (list(m) if len(m) == 5 else [None] * 5))
    return rows


def _bits_to_numbers(bits):
    out = []
    k = 1
    while bits:
        if bits & 1:
            out.append(k)
        bits >>= 1
        k += 1
    return out


# ---------------------------------------------------------------- strategy 1: NCC template matching
def ncc_template_matches(history, h=NCC_WINDOW, top_n=NCC_TOP_N, threshold=NCC_THRESHOLD,
                          include_mach=True):
    """Slide the current h-week two-channel window backwards through the chart.
    Score(r) = sum of per-cell products / sqrt(ones(candidate) * ones(template)) --
    for binary data the numerator is just the bit-overlap. Returns (events,
    total_offsets, n_above_threshold): each event has the candidate window's rows,
    its NCC score, and the historical "drop" (the draw right after the window).

    Candidate windows must end before the current window begins AND their drop row
    must fall strictly before the current window (r <= n - 2h - 1) -- otherwise the
    best "matches" are the ones sharing rows with the present and the drop evidence is
    partly the current draws themselves (the echo failure mode this project measures
    for explicitly)."""
    n = len(history)
    if n < 2 * h + 2:
        return [], 0, 0
    W = binary_matrix(history, 'win')
    M = binary_matrix(history, 'mach') if include_mach else [0] * n
    tw = [W[n - h + i] for i in range(h)]
    tm = [M[n - h + i] for i in range(h)]
    t_ones = sum(r.bit_count() for r in tw) + sum(r.bit_count() for r in tm)
    if t_ones == 0:
        return [], 0, 0
    events = []
    total = 0
    for r in range(0, n - 2 * h):
        total += 1
        overlap = sum((W[r + i] & tw[i]).bit_count() for i in range(h))
        overlap += sum((M[r + i] & tm[i]).bit_count() for i in range(h))
        c_ones = sum(W[r + i].bit_count() for i in range(h)) + sum(M[r + i].bit_count() for i in range(h))
        if c_ones == 0:
            continue
        score = overlap / math.sqrt(c_ones * t_ones)
        events.append({
            'offset': r,
            'window_start': history[r]['date'], 'window_end': history[r + h - 1]['date'],
            'score': score, 'overlap_cells': overlap,
            'drop_date': history[r + h]['date'], 'drop_win': history[r + h]['win'],
        })
    events.sort(key=lambda e: -e['score'])
    n_above = sum(1 for e in events if threshold is not None and e['score'] >= threshold)
    return events[:top_n], total, n_above


def ncc_score(history, h=NCC_WINDOW, top_n=NCC_TOP_N, events=None):
    """Credit each number by the similarity-weighted frequency with which it appeared
    in the "drop" row after the top NCC matches. Returns (scores, evidence)."""
    if events is None:
        events, total, n_above = ncc_template_matches(history, h=h, top_n=top_n)
    else:
        events, total, n_above = events
    scores = {k: 0.0 for k in range(1, 91)}
    sim_total = sum(e['score'] for e in events)
    if sim_total > 0:
        for e in events:
            for x in e['drop_win']:
                scores[x] += e['score'] / sim_total
    return scores, {'events': events, 'total_offsets': total, 'n_above_threshold': n_above}


# ---------------------------------------------------------------- strategy 2: diagonals
def diagonal_completion_rates(history, deltas=DIAG_DELTAS, field='win'):
    """Per step-vector d: of every historical 2-cell partial run (B[r,c]=1 and
    B[r+1,c+d]=1), how often did the third cell B[r+2,c+2d]=1 complete the diagonal?
    (Coordinates wrap 1-90.) Chance is ~5.56% -- the completing cell is just 'a
    specific number in a specific future draw'. Returns {delta: (rate, trials)}."""
    n = len(history)
    B = binary_matrix(history, field)
    rates = {}
    for d in deltas:
        hits = trials = 0
        for r in range(n - 2):
            row0, row1, row2 = B[r], B[r + 1], B[r + 2]
            if not row0 or not row1:
                continue
            for c in _bits_to_numbers(row0):
                c1 = _wrap90(c + d)
                if not (row1 >> (c1 - 1)) & 1:
                    continue
                trials += 1
                c2 = _wrap90(c + 2 * d)
                hits += (row2 >> (c2 - 1)) & 1
        rates[d] = (hits / trials if trials else 0.0, trials)
    return rates


def diagonal_projections(history, deltas=DIAG_DELTAS, rates=None):
    """Every partial diagonal run that ends at the CURRENT row, extrapolated one step:
    cells (n-2, c) and (n-1, c+d) both set project candidate c+2d for the next draw,
    weighted by that step-vector's measured, Wilson-shrunk completion rate."""
    n = len(history)
    if n < 2:
        return []
    rates = rates if rates is not None else diagonal_completion_rates(history, deltas)
    prev_set = set(history[-2]['win'])
    cur_set = set(history[-1]['win'])
    out = []
    for d in deltas:
        rate, trials = rates.get(d, (0.0, 0))
        w = _wilson_lower_bound(rate, trials)
        for c in prev_set:
            c1 = _wrap90(c + d)
            if c1 in cur_set:
                out.append({'delta': d, 'run': [(history[-2]['date'], c), (history[-1]['date'], c1)],
                            'projected': _wrap90(c + 2 * d), 'rate': rate, 'trials': trials, 'weight': w})
    return out


def diagonal_score(history, deltas=DIAG_DELTAS, rates=None):
    scores = {k: 0.0 for k in range(1, 91)}
    detail = defaultdict(list)
    for p in diagonal_projections(history, deltas, rates):
        scores[p['projected']] += p['weight']
        detail[p['projected']].append(p)
    return scores, detail


# ---------------------------------------------------------------- strategy 3: boxes & V-shapes
def box_completion_rate(history, lookahead=TRACE_LOOKAHEAD):
    """The box pattern in predictive form. A 'box' is a pair {c1,c2} occupying two
    rows; predictively: a pair co-occurred in some PAST row, one member appears in the
    current row -- how often does the OTHER member appear within `lookahead` draws
    (completing the enclosure)? Measured globally (per-pair samples are too thin to
    rate individually -- disclosed as pair counts instead). Chance is
    1-(85/90)^lookahead (~24.6% at 5): 'partner appears somewhere in the next few
    draws' is a broad event, which is why the rate must be compared to THAT chance,
    not to 5.56%."""
    n = len(history)
    win_sets = [set(d['win']) for d in history]
    partner_map = defaultdict(set)  # number -> partners seen in rows strictly before t
    hits = trials = 0
    for t in range(n):
        cur = win_sets[t]
        if 0 < t < n - 1:  # needs at least one resolved future draw to count as a trial
            future = win_sets[t + 1:t + 1 + lookahead]
            counted = set()
            for x in cur:
                for y in partner_map.get(x, ()):
                    if y in counted or y in cur:  # self/echo guard: already-drawn partner isn't a prediction
                        continue
                    counted.add(y)
                    trials += 1
                    hits += any(y in f for f in future)
        for a, b in _pairs(cur):
            partner_map[a].add(b)
            partner_map[b].add(a)
    return (hits / trials if trials else 0.0), trials


def _pairs(nums):
    lst = sorted(nums)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            yield lst[i], lst[j]


def box_score(history, lookahead=TRACE_LOOKAHEAD, rate_trials=None):
    """Credit every historical box partner of the current draw's numbers by the
    global, Wilson-shrunk box completion rate, scaled by how many distinct past rows
    the pair occupied (log-damped -- more co-occurrences is more box evidence, but not
    linearly). Partners that are THEMSELVES in the current draw are disclosed but not
    scored (self_credit) -- crediting a number for co-occurring with its own draw-mates
    is the last-week-echo channel, not a prediction."""
    scores = {k: 0.0 for k in range(1, 91)}
    detail = defaultdict(list)
    if len(history) < 3:
        return scores, detail
    rate, trials = rate_trials if rate_trials is not None else box_completion_rate(history, lookahead)
    w_base = _wilson_lower_bound(rate, trials)
    pair_rows = defaultdict(int)
    for d in history[:-1]:
        for a, b in _pairs(d['win']):
            pair_rows[(a, b)] += 1
    last_set = set(history[-1]['win'])
    for x in last_set:
        for (a, b), cnt in pair_rows.items():
            if x not in (a, b):
                continue
            y = b if a == x else a
            self_credit = y in last_set
            weight = w_base * math.log1p(cnt)
            if not self_credit:
                scores[y] += weight
            detail[y].append({'partner_of': x, 'pair_rows': cnt, 'rate': rate,
                              'trials': trials, 'weight': weight, 'self_credit': self_credit})
    return scores, detail


def v_shape_events(history, deltas=(1, 2), field='win'):
    """Full V-shapes -- (r,c),(r+1,c-d),(r+2,c-2d),(r+3,c-d),(r+4,c) -- found anywhere
    in history, plus partial V's (first four cells, ending at the current row) with
    their projected completion c for the next draw. At 5/90 density the expected count
    of full V's is ~0 per game; the point of computing them is to REPORT that honestly.
    Returns (full_events, partial_projections, completion_rate, trials): trials = the
    number of historical 4-cell partial V's whose completing 5th row exists."""
    n = len(history)
    B = binary_matrix(history, field)
    full, partial = [], []
    hits = trials = 0
    for d in deltas:
        for r in range(n - 3):
            row0, row1, row2, row3 = B[r], B[r + 1], B[r + 2], B[r + 3]
            for c in _bits_to_numbers(row0):
                c1, c2, c3 = _wrap90(c - d), _wrap90(c - 2 * d), _wrap90(c - d)
                if not ((row1 >> (c1 - 1)) & 1 and (row2 >> (c2 - 1)) & 1 and (row3 >> (c3 - 1)) & 1):
                    continue
                if r + 4 < n:
                    trials += 1
                    if (B[r + 4] >> (c - 1)) & 1:
                        hits += 1
                        full.append({'delta': d, 'vertex_row': history[r + 2]['date'],
                                     'start': (history[r]['date'], c), 'completed_on': history[r + 4]['date']})
                elif r + 3 == n - 1:
                    partial.append({'delta': d, 'projected': c,
                                    'run': [(history[r]['date'], c), (history[r + 1]['date'], c1),
                                            (history[r + 2]['date'], c2), (history[r + 3]['date'], c3)]})
    rate = hits / trials if trials else 0.0
    return full, partial, rate, trials


def v_shape_score(history, deltas=(1, 2)):
    scores = {k: 0.0 for k in range(1, 91)}
    full, partial, rate, trials = v_shape_events(history, deltas)
    w = _wilson_lower_bound(rate, trials)
    detail = defaultdict(list)
    for p in partial:
        scores[p['projected']] += w
        detail[p['projected']].append({**p, 'rate': rate, 'trials': trials, 'weight': w})
    return scores, {'full': full, 'partial': partial, 'rate': rate, 'trials': trials, 'detail': detail}


# ---------------------------------------------------------------- strategy 4: keys
def _op_turning(x):
    """Digit inversion ('turning'): 23 -> 32; 5 -> 50; 90 -> 9.

    Numbers ending in 9 invert to 91-99, which are off the 1-90 board -- they return
    None (no derivation) rather than being wrapped modulo 90 into an unrelated number.
    See plan_engine._turning for the full reasoning; the traditional turning chart maps
    those numbers to themselves, and self-derivations are excluded from scoring anyway."""
    inv = (x % 10) * 10 + x // 10 if x >= 10 else x * 10
    return inv if 1 <= inv <= 90 else None


def _op_mirror(x): return 91 - x
def _op_double(x): return _wrap90(2 * x)


def _key_single_ops():
    ops = {'turning': _op_turning, 'mirror': _op_mirror, 'double': _op_double}
    for k in range(1, 10):
        ops[f'plus{k}'] = (lambda kk: (lambda x: _wrap90(x + kk)))(k)
        ops[f'minus{k}'] = (lambda kk: (lambda x: _wrap90(x - kk)))(k)
    return ops
    # NOTE: no 'identity' -- see module docstring (it's the lap/positional echo channel).


_KEY_PAIR_OPS = {'sum': lambda a, b: _wrap90(a + b),
                 'diff': lambda a, b: (_wrap90(abs(a - b)) if a != b else None)}
_SLOT_NAMES = ['W1', 'W2', 'W3', 'W4', 'W5', 'M1', 'M2', 'M3', 'M4', 'M5']


def _binom_tail_p(hits, trials, p0):
    """Exact binomial upper-tail P(X >= hits | trials, p0)."""
    if hits <= 0:
        return 1.0
    try:
        from scipy.stats import binom
        # binom.sf(k, n, p) calculates P(X > k). We want P(X >= hits), so we use hits - 1.
        return float(binom.sf(hits - 1, trials, p0))
    except ImportError:
        # Fallback if scipy is missing
        q = 0.0
        for i in range(hits, trials + 1):
            try:
                q += math.comb(trials, i) * (p0 ** i) * ((1 - p0) ** (trials - i))
            except OverflowError:
                return 0.0
            if q > 1.0:
                return 1.0
        return q


def key_search(history, alpha=KEY_SCREEN_ALPHA, min_trials=30):
    """The Key Identification Engine: measure every candidate positional equation
    (single-slot op or pair-slot sum/diff from week r's 10 slots, equaling one of week
    r+1's 5 WINNING slots) across the whole passed history, and return the keys whose
    hit count clears an exact binomial screen against the 1/90 slot-match chance.

    Honesty notes, in the strongest terms: ~1,500 keys are tested, so at alpha=0.01
    roughly 15 'significant' keys are EXPECTED from pure noise (they are listed with
    their p-values precisely so this is visible; `expected_false_positives` says it
    outright). Use bootstrap_best_key_pvalue() for the family-wise answer. Derivations
    that equal their own source value (turning fixed points etc.) are skipped, same
    self-credit rule as everywhere else in this project."""
    P = positional_matrix(history)
    n = len(P)
    singles = _key_single_ops()
    keys = []
    tested = 0
    for si in range(10):
        for op_name, fn in singles.items():
            for tj in range(5):
                tested += 1
                hits = trials = 0
                for r in range(n - 1):
                    src = P[r][si]
                    tgt = P[r + 1][tj]
                    if src is None or tgt is None:
                        continue
                    d = fn(src)
                    if d is None or d == src:
                        continue
                    trials += 1
                    hits += d == tgt
                if trials >= min_trials:
                    pv = _binom_tail_p(hits, trials, KEY_SLOT_CHANCE)
                    if pv < alpha:
                        keys.append({'kind': 'single', 'op': op_name, 'src': _SLOT_NAMES[si],
                                     'src_idx': (si,), 'target': _SLOT_NAMES[tj], 'target_idx': tj,
                                     'hits': hits, 'trials': trials, 'rate': hits / trials, 'p_value': pv})
    for si in range(10):
        for sj in range(si + 1, 10):
            for op_name, fn in _KEY_PAIR_OPS.items():
                for tj in range(5):
                    tested += 1
                    hits = trials = 0
                    for r in range(n - 1):
                        a, b, tgt = P[r][si], P[r][sj], P[r + 1][tj]
                        if a is None or b is None or tgt is None:
                            continue
                        d = fn(a, b)
                        if d is None:
                            continue
                        trials += 1
                        hits += d == tgt
                    if trials >= min_trials:
                        pv = _binom_tail_p(hits, trials, KEY_SLOT_CHANCE)
                        if pv < alpha:
                            keys.append({'kind': 'pair', 'op': op_name,
                                         'src': f'{_SLOT_NAMES[si]}+{_SLOT_NAMES[sj]}',
                                         'src_idx': (si, sj), 'target': _SLOT_NAMES[tj], 'target_idx': tj,
                                         'hits': hits, 'trials': trials, 'rate': hits / trials, 'p_value': pv})
    keys.sort(key=lambda k: k['p_value'])
    return {'keys': keys, 'tested': tested,
            'expected_false_positives': tested * alpha}


def key_score(history, key_report=None):
    """Apply every screened key to the current row: the derived value is a candidate
    for the key's target slot next week, weighted by the key's own Wilson-shrunk
    measured rate. (A key predicts a slot VALUE; as a number-level score we credit the
    derived number itself -- the slot is disclosed in the detail.)"""
    scores = {k: 0.0 for k in range(1, 91)}
    detail = defaultdict(list)
    if len(history) < 2:
        return scores, detail
    report = key_report if key_report is not None else key_search(history)
    P_last = positional_matrix(history[-1:])[0]
    singles = _key_single_ops()
    for key in report['keys']:
        if key['kind'] == 'single':
            src = P_last[key['src_idx'][0]]
            if src is None:
                continue
            d = singles[key['op']](src)
            if d is None or d == src:  # self-derivation guard (turning fixed points etc.)
                continue
        else:
            a, b = (P_last[i] for i in key['src_idx'])
            if a is None or b is None:
                continue
            d = _KEY_PAIR_OPS[key['op']](a, b)
            if d is None:
                continue
        w = _wilson_lower_bound(key['rate'], key['trials'])
        scores[d] += w
        detail[d].append({**key, 'derived': d, 'weight': w})
    return scores, detail


# ---------------------------------------------------------------- strategy 5: machine->win crossover
def mach_cross_rates(history, taus=(1, 2, 3, 4, 5)):
    """Lag-resolved machine->winning crossover: for each lag tau, the global measured
    rate at which a number drawn in the MACHINE channel at t appears in the WINNING
    channel at t+tau. Chance is 5.56% per number per draw. Returns
    {tau: (rate, trials)}."""
    n = len(history)
    win_sets = [set(d['win']) for d in history]
    rates = {}
    for tau in taus:
        hits = trials = 0
        for i in range(n - tau):
            mach = history[i].get('mach') or []
            for x in mach:
                trials += 1
                hits += x in win_sets[i + tau]
        rates[tau] = (hits / trials if trials else 0.0, trials)
    return rates


def mach_cross_score(history, taus=(1, 2, 3, 4, 5), rates=None):
    """Credit the numbers drawn as machine numbers tau draws ago by the measured,
    shrunk crossover rate for that specific lag (a number reachable at several lags
    accumulates each lag's weight -- disclosed per-lag in the detail)."""
    scores = {k: 0.0 for k in range(1, 91)}
    detail = defaultdict(list)
    n = len(history)
    rates = rates if rates is not None else mach_cross_rates(history, taus)
    for tau in taus:
        if tau > n:
            continue
        rate, trials = rates.get(tau, (0.0, 0))
        w = _wilson_lower_bound(rate, trials)
        d = history[n - tau]
        for x in (d.get('mach') or []):
            scores[x] += w
            detail[x].append({'tau': tau, 'mach_date': d['date'], 'rate': rate,
                              'trials': trials, 'weight': w})
    return scores, detail


# ---------------------------------------------------------------- the blended spatial mode
def spatial_state(history):
    """Everything expensive that the per-draw score reuses: measured diagonal rates,
    the box completion rate, the screened key list, the lagged crossover rates, and
    the strict joint-conditional cohort table. The walk-forward backtest recomputes
    this periodically (staleness in these slowly-drifting measurements is the same
    honest tradeoff as retrain_every / the pattern weights recompute)."""
    from .trend_analysis import joint_conditional_rates, counting_week_stats
    return {
        'diag_rates': diagonal_completion_rates(history),
        'box_rate': box_completion_rate(history),
        'key_report': key_search(history),
        'mach_rates': mach_cross_rates(history),
        'cond_rates': joint_conditional_rates(history),
        'cw_stats': counting_week_stats(history),
    }


def spatial_scores(history, state=None):
    """The blended Spatial Pattern Matching Engine score: NCC template matching,
    diagonal extrapolation, box completion, V-shapes, screened keys, lagged
    machine->win crossover, and strict joint-conditional cohort rates, combined with
    fixed disclosed weights. Returns (scores, components)."""
    from .ensemble import blend_scores
    from .trend_analysis import joint_conditional_score, counting_week_score
    if state is None:
        state = spatial_state(history)
    last = history[-1] if history else None
    ncc, _ = ncc_score(history)
    diag, _ = diagonal_score(history, rates=state['diag_rates'])
    box, _ = box_score(history, rate_trials=state['box_rate'])
    vsh, _ = v_shape_score(history)
    keys, _ = key_score(history, key_report=state['key_report'])
    mcross, _ = mach_cross_score(history, rates=state['mach_rates'])
    cond, _ = joint_conditional_score(history, last, rates_and_bounds=state['cond_rates'])
    periodic, _ = counting_week_score(history, stats=state['cw_stats'])
    comps = {'ncc': ncc, 'diagonal': diag, 'box': box, 'vshape': vsh,
             'keys': keys, 'mach_cross': mcross, 'conditional': cond,
             'periodic': periodic}
    # box structurally can't credit the current draw's own numbers (self-credit
    # exclusion) -- renormalize those numbers' remaining weights instead of letting
    # the min-max zero read as "worst possible" (same fix as terminal/group).
    exclude = {x: {'box'} for x in history[-1]['win']} if history else {}
    return blend_scores(comps, SPATIAL_WEIGHTS, exclude=exclude), comps


# ---------------------------------------------------------------- Component 2: bootstrap noise control
def _synthetic_history(history, rng):
    """A synthetic chart: same length, same per-row density (5 win numbers, machine
    numbers present exactly where the real data has them), numbers drawn fresh and
    uniformly -- data density preserved, every spatial/algebraic structure destroyed."""
    out = []
    nums = list(range(1, 91))
    for d in history:
        win = rng.sample(nums, 5)
        mach = rng.sample(nums, 5) if d.get('mach') else []
        out.append({'date': d['date'], 'code': d.get('code'), 'win': win, 'mach': mach})
    return out


def _key_hits_on(history, key):
    """Re-measure one key's (hits, trials) on an arbitrary (real or synthetic) chart."""
    P = positional_matrix(history)
    singles = _key_single_ops()
    hits = trials = 0
    for r in range(len(P) - 1):
        tgt = P[r + 1][key['target_idx']]
        if tgt is None:
            continue
        if key['kind'] == 'single':
            src = P[r][key['src_idx'][0]]
            if src is None:
                continue
            d = singles[key['op']](src)
            if d is None or d == src:
                continue
        else:
            a, b = (P[r][i] for i in key['src_idx'])
            if a is None or b is None:
                continue
            d = _KEY_PAIR_OPS[key['op']](a, b)
            if d is None:
                continue
        trials += 1
        hits += d == tgt
    return hits, trials


def bootstrap_key_pvalue(history, key, iterations=1000, seed=0):
    """Per-key empirical p-value: how often does THIS key hit at least as often on a
    structure-destroyed synthetic chart as it did on the real one? (Remember the
    look-elsewhere effect: a key that was FOUND by searching ~1,500 candidates will
    look significant per-key even under noise -- use bootstrap_best_key_pvalue for the
    family-wise answer.)"""
    rng = random.Random(seed)
    real_hits, _ = _key_hits_on(history, key)
    ge = 0
    for _ in range(iterations):
        fake_hits, _ = _key_hits_on(_synthetic_history(history, rng), key)
        ge += fake_hits >= real_hits
    return ge / iterations


def bootstrap_best_key_pvalue(history, iterations=200, seed=0, alpha=KEY_SCREEN_ALPHA):
    """Family-wise (max-statistic) test: run the ENTIRE key search on each synthetic
    chart and record the best (lowest) p-value noise produced; the reported p-value is
    the fraction of synthetic charts whose best key beat the real chart's best key.
    This is the honest answer to 'is my best key better than the best key pure noise
    would hand me?' -- if it's > 0.05, every screened key is indistinguishable from
    apophenia and the blueprint's decision rule says: discard."""
    rng = random.Random(seed)
    real = key_search(history, alpha=1.1)  # alpha>1: keep all, we want the true best p
    real_best = min((k['p_value'] for k in real['keys']), default=1.0)
    ge = 0
    for _ in range(iterations):
        fake = key_search(_synthetic_history(history, rng), alpha=1.1)
        fake_best = min((k['p_value'] for k in fake['keys']), default=1.0)
        ge += fake_best <= real_best
    return {'real_best_p': real_best, 'bootstrap_p': ge / iterations, 'iterations': iterations}
