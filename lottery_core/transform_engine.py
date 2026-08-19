"""Pattern transformation & replication engine.

Formalizes the pattern-analysis reading strategy as a rule-discovery and replication
system rather than a frequency counter:

1. PATTERN IDENTIFICATION -- groups (pairs, triplets) of numbers within a draw, plus
   their sorted-rank positions (positions_of) and, as a second lens, their as-drawn
   slot order (draw_order_positions) since sorted-rank is mechanically confounded with
   a number's own magnitude. spacing_profile() additionally discloses each draw's
   within-draw distribution characteristics (gaps, span, decade-clustering) -- purely
   descriptive, not scored (see its docstring for why).
2. TRANSITION ANALYSIS -- for every historical occurrence of a group, what actually won
   afterward: literal carry-over of the same number(s) (pattern_trace_events/
   pattern_trace_score, one event per historical repeat -- never the current draw
   citing itself), and which position it landed in relative to where the group's
   numbers sat (positional_carryover_rates and its as-drawn-order counterpart
   draw_order_carryover_rates, including the lag=1-only 'lap' special case,
   lap_carryover_rates). class_carryover_rates generalizes the same "did this group's
   membership carry into a later draw" question to two named lottery-paper groupings --
   terminal_of (last-digit terminal) and group_of (digital root) -- crediting ANY member
   of the class, not just the same number, since that's what those strategies actually
   claim.
3. A REGISTRY of candidate transform rules, not one hand-picked "the" rule -- arithmetic
   over individual numbers (doubling/mirror, tested once per number) and genuinely over
   pairs (sum/difference) and triplets (sum-of-three/mean-of-three). Each rule's
   reliability is MEASURED from the full draw history (mirrors
   classic.chart_transfer_rates) and shrunk by its trial count (Wilson lower bound)
   before it can drive a score -- never assumed, and never let a thin-evidence rate
   outweigh a well-sampled one.
4. REPLICATION -- every rule is applied to the *current* draw's numbers/groups, weighted
   by its own measured, shrunk reliability, to generate this week's candidates
   (transform_score). Multiple transformation paths are combined into one score, not a
   single deterministic guess.

Everything here is inspectable: pattern_trace_events() and transform_score()'s `detail`
return value expose the literal historical instances behind any candidate, so the
reasoning can be audited (see pattern_analysis.explain()).
"""
from collections import Counter, defaultdict
from itertools import combinations

TRACE_LOOKAHEAD = 5
TRACE_DECAY = 0.7


def _wrap90(x):
    return ((int(round(x)) - 1) % 90) + 1


# ---------------------------------------------------------------- rule registry
# Rules are split by how many numbers they actually depend on: SINGLE_RULES read one
# number, PAIR_RULES genuinely combine two, TRIPLET_RULES genuinely combine three. This
# split matters for measurement, not just style -- a single-argument rule tested once
# per PAIR (as double/mirror used to be, split into _a/_b variants keyed on which slot
# of the pair held the number) re-tests the same underlying fact once for every other
# number it happens to be paired with in a 5-number draw, inflating both the trial count
# and the score contribution ~4x relative to a genuinely pairwise rule. Testing them
# once per NUMBER instead removes that pseudo-replication.
def _sum(a, b): return _wrap90(a + b)
def _diff(a, b): return _wrap90(abs(a - b)) if a != b else None
def _double(a): return _wrap90(2 * a)
def _mirror(a): return 91 - a
def _plus1(a): return _wrap90(a + 1)   # 'one up' -- classic neighbor key
def _minus1(a): return _wrap90(a - 1)  # 'one down'
def _sum3(a, b, c): return _wrap90(a + b + c)
def _mean3(a, b, c): return _wrap90(round((a + b + c) / 3))


SINGLE_RULES = {'double': _double, 'mirror': _mirror, 'plus1': _plus1, 'minus1': _minus1}
PAIR_RULES = {'sum': _sum, 'diff': _diff}
TRIPLET_RULES = {'sum3': _sum3, 'mean3': _mean3}
# NOTE: the 9 CHARTS relationships (charts.py) are deliberately NOT folded in here as
# single-argument rules. They already have their own dedicated, separately-weighted
# component (pattern_analysis's 'charts', via classic.chart_scores) with their own
# win/machine-number weighting -- adding them again here would score the exact same
# chart lookups twice under two different weights. Measured on real history, the
# resulting 'transform' vs 'charts' components correlated at 0.60 when this was tried,
# i.e. mostly the same signal counted under two labels rather than two independent
# ones. If you want a chart-flavored rule in the arithmetic registry, measure it
# separately and be able to show it isn't just re-deriving the existing 'charts'
# component's own numbers.

Z95 = 1.959963984540054


def _wilson_lower_bound(rate, trials, z=Z95):
    """Conservative (lower-bound) read of a measured rate given its trial count -- the
    same Wilson-score formula the project already uses for honest backtest reporting
    (see backtest.py's _wilson_ci), applied here so a rule's SCORE weight reflects how
    much evidence backs it rather than just the raw point estimate: a rate measured from
    2 trials should not compete on equal footing with the same rate measured from 200."""
    if trials == 0:
        return 0.0
    denom = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denom
    margin = z * ((rate * (1 - rate) / trials + z * z / (4 * trials * trials)) ** 0.5) / denom
    return max(0.0, center - margin)


def measure_rule_rates(history):
    """Each rule's own measured transfer rate: how often applying it to a group from
    draw i produces a number that actually won in draw i+1. Generalizes
    classic.chart_transfer_rates to the arithmetic rule registry -- never assumed."""
    rates = {}
    for name, fn in SINGLE_RULES.items():
        hits = trials = 0
        for a_draw, b_draw in zip(history[:-1], history[1:]):
            bset = set(b_draw['win'])
            for a in a_draw['win']:
                d = fn(a)
                if d is None:
                    continue
                trials += 1
                hits += d in bset
        rates[name] = (hits / trials if trials else 0.0, trials)
    for name, fn in PAIR_RULES.items():
        hits = trials = 0
        for a_draw, b_draw in zip(history[:-1], history[1:]):
            bset = set(b_draw['win'])
            for a, b in combinations(a_draw['win'], 2):
                d = fn(a, b)
                if d is None:
                    continue
                trials += 1
                hits += d in bset
        rates[name] = (hits / trials if trials else 0.0, trials)
    for name, fn in TRIPLET_RULES.items():
        hits = trials = 0
        for a_draw, b_draw in zip(history[:-1], history[1:]):
            bset = set(b_draw['win'])
            for a, b, c in combinations(a_draw['win'], 3):
                d = fn(a, b, c)
                if d is None:
                    continue
                trials += 1
                hits += d in bset
        rates[name] = (hits / trials if trials else 0.0, trials)
    return rates


def transform_score(history, last, rule_rates=None):
    """Apply every rule in the registry to the current draw's numbers/pairs/triplets,
    weighted by each rule's own measured reliability (shrunk by trial count via a
    Wilson lower bound, so a rate backed by a handful of trials can't outweigh one
    backed by many). Returns (scores, detail) where detail[k] lists exactly which
    rule+group produced k, its raw measured rate/trials, and the shrunk weight actually
    used -- the full evidence trail."""
    scores = {k: 0.0 for k in range(1, 91)}
    detail = defaultdict(list)
    if not last or len(history) < 2:
        return scores, detail
    rates = rule_rates if rule_rates is not None else measure_rule_rates(history)
    for a in last['win']:
        for name, fn in SINGLE_RULES.items():
            d = fn(a)
            if d is None:
                continue
            rate, trials = rates[name]
            weight = _wilson_lower_bound(rate, trials)
            scores[d] += weight
            detail[d].append({'rule': name, 'group': (a,), 'derived': d, 'rate': rate, 'trials': trials, 'weight': weight})
    for a, b in combinations(last['win'], 2):
        for name, fn in PAIR_RULES.items():
            d = fn(a, b)
            if d is None:
                continue
            rate, trials = rates[name]
            weight = _wilson_lower_bound(rate, trials)
            scores[d] += weight
            detail[d].append({'rule': name, 'group': (a, b), 'derived': d, 'rate': rate, 'trials': trials, 'weight': weight})
    for a, b, c in combinations(last['win'], 3):
        for name, fn in TRIPLET_RULES.items():
            d = fn(a, b, c)
            if d is None:
                continue
            rate, trials = rates[name]
            weight = _wilson_lower_bound(rate, trials)
            scores[d] += weight
            detail[d].append({'rule': name, 'group': (a, b, c), 'derived': d, 'rate': rate, 'trials': trials, 'weight': weight})
    return scores, detail


# ---------------------------------------------------------------- literal-carryover (pattern) tracing
def positions_of(win):
    s = sorted(win)
    return {x: s.index(x) + 1 for x in win}


_positions = positions_of  # internal alias used throughout this module


def pattern_trace_events(history, group_sizes=(2, 3), lookahead=TRACE_LOOKAHEAD, field='win'):
    """Generalizes pair-tracing to both pairs and triplets, with position tracking: for
    every group among the current draw's winning numbers, every earlier draw (for this
    game) where that exact group also co-occurred, and what won in the draws that
    followed -- including which sorted-rank position the group held then vs. now, and
    which position the follow-up hit landed in.

    `field` selects which set of numbers is traced: 'win' (default) traces the winning
    numbers against historical winning numbers; 'mach' traces the current MACHINE
    numbers against historical MACHINE numbers -- the machine-pair reading strategy
    ("this machine pair has shown before; what won after it?"). Follow-ups are always
    winning numbers, since that's what a reader is trying to predict. Draws with no
    recorded machine numbers (pre-Aug 2018) simply never match.

    One event per historical repeat-INDEX, not per matching group: a historical draw
    that overlaps the current draw's numbers by 3+ can satisfy several of the seed
    pairs/triplets at once, but that is still a single coincidence, not several
    independent ones -- counting it once each would inflate both the score and the
    disclosed trial count for the exact same fact (all matched groups are listed under
    `groups` so the evidence isn't lost, just not double counted).

    The follow-up window is capped one draw short of the end of `history` so the
    current/seed draw (history[-1], which every seed group is drawn from and therefore
    trivially "matches" itself) can never be picked up as its own follow-up evidence."""
    if len(history) < 2:
        return []
    last = history[-1]
    seed_nums = last.get(field) or []
    if not seed_nums:
        return []
    current_pos = _positions(seed_nums)
    seed_groups = []
    for size in group_sizes:
        seed_groups.extend(combinations(sorted(seed_nums), size))
    match_sets = [set(d.get(field) or []) for d in history]
    n = len(history)
    events = []
    for i in range(n - 1):
        s = match_sets[i]
        if not s:
            continue
        matched = [g for g in seed_groups if all(x in s for x in g)]
        if not matched:
            continue
        s_pos = _positions(history[i].get(field) or [])
        followups = []
        for lag, j in enumerate(range(i + 1, min(i + 1 + lookahead, n - 1)), start=1):
            followups.append({'lag': lag, 'date': history[j]['date'], 'win': history[j]['win'],
                               'hit_positions': _positions(history[j]['win'])})
        events.append({
            'groups': matched, 'repeat_date': history[i]['date'],
            'seed_positions_then': {x: s_pos[x] for g in matched for x in g},
            'seed_positions_now': {x: current_pos[x] for g in matched for x in g},
            'followups': followups,
        })
    events.sort(key=lambda e: e['repeat_date'], reverse=True)
    return events


def pattern_trace_score(history, lookahead=TRACE_LOOKAHEAD, decay=TRACE_DECAY, events=None, field='win'):
    """Literal-carryover component: credits numbers that won after a historical repeat
    of one of the current draw's pairs/triplets, decayed by lag ('how many weeks
    down'). Returns (scores, trials). `field='mach'` gives the machine-pair variant
    (see pattern_trace_events)."""
    events = events if events is not None else pattern_trace_events(history, lookahead=lookahead, field=field)
    scores = {k: 0.0 for k in range(1, 91)}
    trials = len(events)
    for e in events:
        for f in e['followups']:
            w = decay ** (f['lag'] - 1)
            for k in f['win']:
                scores[k] += w
    if trials:
        scores = {k: v / trials for k, v in scores.items()}
    return scores, trials


# ---------------------------------------------------------------- positional carryover
def positional_carryover_rates(history, lookahead=TRACE_LOOKAHEAD):
    """For each sorted-rank source position (1-5): the empirical rate that the number
    occupying it carries over into ANY later draw within `lookahead`, and the
    distribution of which position it lands in when it does -- a measured 'does
    position 2 tend to become position 4' profile, not an assumption."""
    n = len(history)
    win_sets = [set(d['win']) for d in history]
    hits = Counter(); trials = Counter()
    landing = defaultdict(Counter)  # source_pos -> Counter(target_pos)
    for i in range(n - 1):
        pos_i = _positions(history[i]['win'])
        for x, p in pos_i.items():
            trials[p] += 1
            for lag, j in enumerate(range(i + 1, min(i + 1 + lookahead, n)), start=1):
                if x in win_sets[j]:
                    hits[p] += 1
                    landing[p][_positions(history[j]['win'])[x]] += 1
                    break
    rates = {p: (hits.get(p, 0) / trials[p] if trials.get(p) else 0.0, trials.get(p, 0)) for p in range(1, 6)}
    return rates, landing


def positional_carryover_score(history, rates=None):
    """Weights each of the current draw's 5 numbers by how 'sticky' the sorted-rank
    position it occupies has historically been. Necessarily only touches the 5 numbers
    just drawn (position is only defined relative to an actual draw) -- a narrow,
    explanatory signal rather than a broad candidate generator like the other
    components."""
    scores = {k: 0.0 for k in range(1, 91)}
    if not history:
        return scores
    last = history[-1]
    pos = _positions(last['win'])
    rates = rates if rates is not None else positional_carryover_rates(history)[0]
    for x, p in pos.items():
        scores[x] = rates.get(p, (0.0, 0))[0]
    return scores


# ---------------------------------------------------------------- as-drawn order (diagnostic only)
def draw_order_positions(win):
    """As-physically-drawn slot (1st..5th number recorded for this draw), NOT sorted-
    rank. `win` must be in the order it was recorded -- data.load() preserves the raw
    w1..w5 CSV column order, which is not ascending (see positions_of() above for the
    sorted-rank version used everywhere else in this module)."""
    return {x: i + 1 for i, x in enumerate(win)}


def draw_order_carryover_rates(history, lookahead=TRACE_LOOKAHEAD):
    """Same measurement as positional_carryover_rates(), but keyed on the as-drawn slot
    instead of sorted-rank -- a second, independent lens on 'position', since sorted-
    rank is mechanically confounded with a number's own magnitude (a low number is
    almost always rank 1, a high number almost always rank 5) while as-drawn slot is
    not. Diagnostic only: deliberately NOT wired into DEFAULT_WEIGHTS/component_scores,
    surfaced purely for inspection via pattern_analysis.explain(), exactly like
    positional_carryover_rates's own 'landing' distribution."""
    n = len(history)
    win_sets = [set(d['win']) for d in history]
    hits = Counter(); trials = Counter()
    landing = defaultdict(Counter)
    for i in range(n - 1):
        pos_i = draw_order_positions(history[i]['win'])
        for x, p in pos_i.items():
            trials[p] += 1
            for lag, j in enumerate(range(i + 1, min(i + 1 + lookahead, n)), start=1):
                if x in win_sets[j]:
                    hits[p] += 1
                    landing[p][draw_order_positions(history[j]['win'])[x]] += 1
                    break
    rates = {p: (hits.get(p, 0) / trials[p] if trials.get(p) else 0.0, trials.get(p, 0)) for p in range(1, 6)}
    return rates, landing


LAP_LOOKAHEAD = 1  # the classic lottery-paper 'lap': a physical draw-slot repeating in
# the VERY NEXT draw specifically, not 'anywhere within a few weeks' -- a special,
# narrowest-window case of the as-drawn-order lens above, kept as its own named
# function since 'lap' is a specific reading strategy a user would look for by name.


def lap_carryover_rates(history):
    """The 'lap' reading strategy: does the number occupying a given as-drawn slot (1st
    .. 5th number recorded) carry over into the IMMEDIATELY NEXT draw (lag=1), and which
    slot does it land in when it does? Literally draw_order_carryover_rates with
    lookahead pinned to 1 -- reusing that measurement rather than duplicating it."""
    return draw_order_carryover_rates(history, lookahead=LAP_LOOKAHEAD)


def lap_score(history, rates=None):
    """Weights each of the current draw's 5 numbers by the 'lap' (lag=1, as-drawn-slot)
    carryover rate for the physical draw-slot it occupied. Mirrors
    positional_carryover_score exactly, but keyed on as-drawn slot and the lag=1-only
    window."""
    scores = {k: 0.0 for k in range(1, 91)}
    if not history:
        return scores
    last = history[-1]
    pos = draw_order_positions(last['win'])
    rates = rates if rates is not None else lap_carryover_rates(history)[0]
    for x, p in pos.items():
        scores[x] = rates.get(p, (0.0, 0))[0]
    return scores


# ---------------------------------------------------------------- distribution characteristics (descriptive)
def spacing_profile(win):
    """Descriptive-only distribution characteristics of a single draw: the gaps between
    consecutive sorted numbers, the overall span (max-min), and how many of the 9
    decade-buckets (1-10, 11-20, ..., 81-90) the 5 numbers fall into (a crude clustering
    measure -- 5 buckets used means no two numbers share a decade, fewer means some
    decade holds 2+). Computed and disclosed (see pattern_analysis.explain()) so
    'distribution characteristics' are at least visible, but deliberately NOT turned
    into a scored prediction rule here: with no independently-motivated similarity
    metric and ~2,700 draws, a 'find draws with a similar spacing profile' matcher would
    be trivial to overfit to backtest noise -- exactly what this project's fixed-weight,
    never-auto-tuned design (see ensemble.py) exists to avoid. If real signal is ever
    found here, it belongs in the registry as its own honestly-measured, disclosed
    component, not folded in silently."""
    s = sorted(win)
    gaps = [b - a for a, b in zip(s, s[1:])]
    buckets = {(x - 1) // 10 for x in s}
    return {'sorted': s, 'gaps': gaps, 'span': s[-1] - s[0], 'decade_buckets_used': len(buckets)}


# ---------------------------------------------------------------- class/group carryover (terminal, digital-root)
def terminal_of(x):
    """Last-digit 'terminal' (0-9) -- the classic lottery-paper terminal grouping (e.g.
    the '7s terminal': 7, 17, 27, ..., 87). Exactly 9 numbers share each terminal."""
    return x % 10


def group_of(x):
    """Digital root (1-9): repeatedly sum the digits of x until a single digit remains
    (e.g. 67 -> 6+7=13 -> 1+3=4; 90 -> 9+0=9). The classic lottery-paper 'group' number.
    Exactly 10 numbers (1-90) share each digital root."""
    while x >= 10:
        x = sum(int(d) for d in str(x))
    return x


def class_members(classify):
    """{class_id: sorted list of numbers 1-90 in that class} for a classify() function
    such as terminal_of or group_of -- the fixed partition a class-carryover measurement
    is taken over."""
    members = defaultdict(list)
    for x in range(1, 91):
        members[classify(x)].append(x)
    return dict(members)


def class_carryover_rates(history, classify, lookahead=TRACE_LOOKAHEAD):
    """Generalizes positional_carryover_rates' question from sorted-rank position to an
    arbitrary partition of 1-90 (terminal digit or digital-root 'group'): given that SOME
    member of a class appeared in a draw, what fraction of the time did ANY member of
    that SAME class (the same number or a different one) appear in a later draw within
    `lookahead`? This is the group-based reading strategy as it's actually described --
    "the 7s terminal is due" means some member of the terminal, not necessarily the same
    number -- measured from history, never assumed."""
    n = len(history)
    members = class_members(classify)
    win_sets = [set(d['win']) for d in history]
    hits = Counter(); trials = Counter()
    for i in range(n - 1):
        classes_seen = {classify(x) for x in history[i]['win']}
        for c in classes_seen:
            trials[c] += 1
            cset = set(members[c])
            for j in range(i + 1, min(i + 1 + lookahead, n)):
                if win_sets[j] & cset:
                    hits[c] += 1
                    break
    rates = {c: (hits.get(c, 0) / trials[c] if trials.get(c) else 0.0, trials.get(c, 0)) for c in members}
    return rates, members


def class_carryover_score(history, last, classify, rates=None, lookahead=TRACE_LOOKAHEAD):
    """Applies class_carryover_rates' measured, Wilson-shrunk reliability to the current
    draw: every number sharing a class with one of the current draw's 5 numbers is
    credited by that class's own measured carryover rate. Returns (scores, detail) in
    the same shape as transform_score, so it's auditable the same way.

    SELF-credit is excluded from `scores`: a number is always trivially a member of its
    own class, so crediting it for "sharing a class with one of the current draw's
    numbers" when it IS that number is just re-crediting the number for having been
    drawn last time -- not a cross-number relationship. That fact is still recorded in
    `detail` (flagged `self_credit: True`) so it's inspectable via explain(), it just
    doesn't move the score. (Measured on real history: before this exclusion, the
    blended pattern-analysis score's top-5 picks overlapped last week's own 5 numbers at
    ~2.4/5 on average -- 8x the ~0.28/5 chance level -- while overlapping the actual next
    draw at just chance. This is one of several components responsible.)"""
    scores = {k: 0.0 for k in range(1, 91)}
    detail = defaultdict(list)
    if not last or len(history) < 2:
        return scores, detail
    if rates is None:
        rates, members = class_carryover_rates(history, classify, lookahead)
    else:
        members = class_members(classify)
    last_set = set(last['win'])
    for c in {classify(x) for x in last['win']}:
        rate, trials = rates.get(c, (0.0, 0))
        weight = _wilson_lower_bound(rate, trials)
        class_size = len(members.get(c, []))
        for k in members.get(c, []):
            self_credit = k in last_set
            if not self_credit:
                scores[k] += weight
            detail[k].append({'class': c, 'rate': rate, 'trials': trials, 'weight': weight,
                               'class_size': class_size, 'self_credit': self_credit})
    return scores, detail


def terminal_carryover_rates(history, lookahead=TRACE_LOOKAHEAD):
    return class_carryover_rates(history, terminal_of, lookahead)


def terminal_score(history, last, rates=None):
    return class_carryover_score(history, last, terminal_of, rates)


def group_carryover_rates(history, lookahead=TRACE_LOOKAHEAD):
    return class_carryover_rates(history, group_of, lookahead)


def group_score(history, last, rates=None):
    return class_carryover_score(history, last, group_of, rates)
