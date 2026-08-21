"""Plan Discovery Engine -- learning PLANS by backtracing, not voting by similarity.

This is the engine the project owner asked for, and it is deliberately one step deeper
than trend_analysis.trend_match_score / spatial_engine.ncc_score. Those find weeks that
look like the current one and then credit whatever numbers dropped right after them --
a similarity VOTE. A vote has no mechanism, so it can never answer "how did you get
this number?"; it can only say "numbers like this followed weeks like this".

A PLAN answers that question. The reading being formalized here is:

    look at the chart as an image, find situations that look like the one in front of
    you now, see what dropped the week after each of them, then BACKTRACE those dropped
    numbers -- where did they come from in the weeks before? (up to ~10 weeks back).
    Whatever way they kept arriving IS the plan. Apply that same way of arriving to the
    current window and it tells you this week's numbers. Learn several plans, keep them
    all, judge which one has actually earned trust -- and report exactly how each plan
    produced its numbers.

So a plan is a pair: (MATCHER, LINK).

  MATCHER -- what makes a past situation "like now". THE INITIAL STEP IS PAIR-ANCHORING:
    a lookalike week is first and foremost a week that shares a PAIR (or more) of the
    CURRENT draw's own numbers. That is how the reading actually starts -- you take a
    pair off this week's draw and go hunting for it in the old papers. Structure then
    REFINES that anchored set; it does not replace it. Every anchored match records
    WHICH numbers anchored it, so the report can name the pair the plan was built on.

    'pair'            the initial step: the seed week shares >= PAIR_ANCHOR_MIN (2) of
                      the current draw's numbers. The anchoring numbers are recorded.
    'pair_structure'  the same anchored set, then RANKED by structural equivalence --
                      pair-anchored first, shape-refined second. The strictest matcher.
    'triple'          a stronger anchor: >= 3 of the current numbers shared.
    'structure'       shape alone, identity ignored (kept as the comparison arm: it
                      answers "does the anchor matter, or is shape enough on its own?").
    'profile'         the conditional context: the full joint condition profile of the
                      seed draw (sum band AND parity AND span band AND
                      repeat-from-previous) matching the current draw's exactly.
    'image'           the chart-as-image reading: 2D normalized cross-correlation
                      between the last WINDOW weeks and every earlier WINDOW-week block.
    'any'             NO filter -- every historical week. This is not padding: it is the
                      control. A plan that only looks good under an anchored matcher but
                      identical under 'any' is claiming the anchor matters; if both read
                      the same, the anchor is doing nothing and the report says so.

  LINK -- the mechanism a dropped number arrived by, from a source week `lag` weeks
    before the drop (lag = 1..BACKTRACE_DEPTH, i.e. up to ten weeks back, as described).
    The catalogue (LINK_SPECS) covers every family the project has primitives for:
    literal carry-over, machine-number crossover, the arithmetic/"calculation" family
    (double, mirror, one-up, one-down, turning/digit-inversion, pair sum, pair
    difference, triplet sum), all nine traditional charts, and the last-digit /
    digital-root families.

MEASUREMENT, and why it is the honest one. Applying a link to a source week yields a
CANDIDATE SET -- the numbers that plan proposes. Measured over every matched situation:

    yield = (candidates that actually dropped) / (candidates proposed)

For a 5/90 draw, E[hits] = |candidates| x 5/90 exactly (linearity of expectation over
the hypergeometric draw), so the chance baseline for ANY plan, however broad or narrow
its candidate set, is exactly 5/90 = 5.56%. That makes a two-number plan and a
forty-number plan directly comparable on one scale, which is what lets the engine rank
them honestly. The yield is then Wilson lower-bounded by its proposal count before it
may score, so a plan seen a handful of times cannot outrank a well-sampled one.

The backtrace the user asked for and the measurement above are the SAME computation
seen from two sides: applying link L to a matched situation's window produces candidate
set C, and every x in C & (what actually dropped) is a backtraced explanation of x. So
the evidence trail comes out of the scoring loop for free -- see plan_report()'s
`situations`, which lists, per matched situation, which dropped numbers this plan
explained and from which source week.

ANTI-ECHO (the project's standing failure mode, see pattern_analysis' module docstring):
the 'carry' and 'mach_carry' links at lag=1 are, by construction, exactly last week's
own numbers -- a plan built on them cannot forecast, it can only replay. They are
computed and disclosed in the report but excluded from scoring (ECHO_EXCLUDED), the same
quarantine 'lap'/'positional' already live under. Deeper lags (2..10) are genuine
recurrence claims and are scored normally.

LOOK-ELSEWHERE (the project's other standing failure mode, see spatial_engine's key
search): this engine evaluates |matchers| x |links| x |lags| candidate plans -- 1,240 to
1,860 per game in practice, depending on how many matchers find enough lookalike weeks
to learn from -- so the best-looking plan is guaranteed to look impressive by selection
alone. Every plan carries its exact binomial p-value against the 5.56% baseline, and
bootstrap_best_plan_pvalue() runs the ENTIRE plan search against structure-destroyed
synthetic histories and reports how often noise hands back a better best-plan -- the
family-wise, max-statistic arbiter. Its verdict is displayed next to the chosen plan.
That test now runs the FULL search by default; on a reduced one it was judging a plan
the report never displayed, and reading ~5x more favourably for it (Monday Special:
p = 0.10 reduced vs p = 0.65 full). 0.65 is the honest number, and it is the expected
one -- see the module's own framing above, and REPORT.md.

ONE VOTE PER MECHANISM: a plan's numbers for the coming week are link_candidates(link,
lag) and nothing else -- the matcher decides what the plan was MEASURED against, never
what it proposes now. So plans sharing a (link, lag) propose identical numbers, and
letting each add its weight to the blend votes one reading several times. scoring_set()
keeps the best-evidenced plan per mechanism and reports the rest as collapsed rather
than dropping them quietly.
"""
import math
from collections import Counter, defaultdict

from .charts import CHARTS
from .transform_engine import _wilson_lower_bound, _wrap90, terminal_of, group_of
from .spatial_engine import binary_matrix, _synthetic_history, _binom_tail_p
from .trend_analysis import joint_condition_key, _tercile_bounds

BACKTRACE_DEPTH = 10      # "a plan can sometimes be back traced to about 10 weeks past"
WINDOW = 3                # weeks per situation-window (the image template height)
MATCH_TOP_N = 30          # most-similar situations an image match keeps
PAIR_ANCHOR_MIN = 2       # the initial step: a lookalike shares at least a PAIR with now
TRIPLE_ANCHOR_MIN = 3     # the stronger anchor
OVERLAP_MIN = PAIR_ANCHOR_MIN  # backwards-compatible alias
MIN_SITUATIONS = 10       # a plan needs this many matched situations to be scoreable
MIN_PROPOSALS = 60        # ...and this many total proposed candidates
MAX_SITUATIONS = 250      # cost cap; the most RECENT matches are kept (see matched_situations)
CANDIDATE_CAP = 46        # a link proposing more than half the board isn't a plan
CHANCE = 5 / 90           # 5.56%: exact per-candidate chance baseline (see docstring)
TOP_PLANS_SCORED = 12     # how many qualifying plans contribute to the blended score

# lag=1 carry links are last week's own numbers by construction -- disclosed, never scored.
# The five positional links at lag 1 are the same thing sliced by position, so they are
# quarantined too (pos3 at lag 1 IS "the middle number of last week's draw").
ECHO_EXCLUDED = {('carry', 1), ('mach_carry', 1)} | {(f'pos{p}', 1) for p in range(1, 6)}


# ---------------------------------------------------------------- the link catalogue
def _win(d):
    return d.get('win') or []


def _mach(d):
    return d.get('mach') or []


def _carry(d):
    return set(_win(d))


def _mach_carry(d):
    return set(_mach(d))


def _double(d):
    return {_wrap90(2 * x) for x in _win(d)}


def _mirror(d):
    return {91 - x for x in _win(d)}


def _plus1(d):
    return {_wrap90(x + 1) for x in _win(d)}


def _minus1(d):
    return {_wrap90(x - 1) for x in _win(d)}


def _turning(d):
    """Digit inversion ('turning'): 23 -> 32, 5 -> 50, 90 -> 9.

    The eight numbers ending in 9 (19, 29, ... 89) invert to 91-99, which do not exist
    on a 1-90 board. They produce NO candidate here. They used to be wrapped modulo 90,
    which silently turned 19 into 1 -- a number with no reading relationship to 19 at
    all, manufactured purely by the arithmetic. The traditional turning chart handles
    the same numbers by mapping them to themselves (see charts.TURNING: '19=19'), and a
    self-pointer is excluded from scoring everywhere in this project anyway, so dropping
    them is both faithful to the tradition and honest about the board's limits."""
    out = set()
    for x in _win(d):
        inv = (x % 10) * 10 + x // 10 if x >= 10 else x * 10
        if 1 <= inv <= 90 and inv != x:
            out.add(inv)
    return out


def _sum_pair(d):
    w = _win(d)
    return {_wrap90(w[i] + w[j]) for i in range(len(w)) for j in range(i + 1, len(w))}


def _diff_pair(d):
    w = _win(d)
    return {_wrap90(abs(w[i] - w[j])) for i in range(len(w)) for j in range(i + 1, len(w))
            if w[i] != w[j]}


def _sum_triple(d):
    w = _win(d)
    return {_wrap90(w[i] + w[j] + w[k])
            for i in range(len(w)) for j in range(i + 1, len(w)) for k in range(j + 1, len(w))}


def _terminal_family(d):
    """Every OTHER number sharing a last digit with a source number (self excluded --
    self-inclusion would smuggle 'carry' in under another name)."""
    src = set(_win(d))
    fams = {terminal_of(x) for x in src}
    return {k for k in range(1, 91) if terminal_of(k) in fams} - src


def _root_family(d):
    src = set(_win(d))
    fams = {group_of(x) for x in src}
    return {k for k in range(1, 91) if group_of(k) in fams} - src


def _chart_link(chart_name):
    mp = CHARTS[chart_name]

    def fn(d):
        return {mp[x] for x in _win(d) if x in mp and mp[x] != x}
    return fn


def _grid_offset(step):
    """Modular offset on the 1-90 board. Read as a 9-rows-of-10 grid (the shape a lotto
    chart is actually printed in), +/-1 is the horizontal neighbour, +/-10 the vertical
    one, and +/-9 / +/-11 the two diagonals -- i.e. these are the SPATIAL neighbours of a
    cell in the chart image, not just arithmetic."""
    def fn(d):
        return {_wrap90(x + step) for x in _win(d)}
    return fn


def _positional_link(pos):
    """POSITIONAL DYNAMICS: the number occupying sorted-rank position `pos` (1..5) of the
    source week. A one-number plan -- the sharpest kind -- asking 'does the number that
    sat in this slot come back?'. Where it LANDS next is reported as the plan's landing
    profile (see positional_landing), which is the 'position 1 moves to position 3'
    reading stated explicitly."""
    def fn(d):
        w = sorted(_win(d))
        return {w[pos - 1]} if len(w) >= pos else set()
    return fn


LINK_SPECS = {
    'carry': _carry,
    'mach_carry': _mach_carry,
    'double': _double,
    'mirror': _mirror,
    'plus1': _plus1,
    'minus1': _minus1,
    'turning': _turning,
    'sum_pair': _sum_pair,
    'diff_pair': _diff_pair,
    'sum_triple': _sum_triple,
    'terminal_family': _terminal_family,
    'root_family': _root_family,
}
# The traditional 'turning' chart is digit inversion, which _turning already implements
# (verified identical for all 90 source numbers -- tools/check_plan_links.py). Registering
# it twice made one mechanism compete as two plans: both qualified, both landed in the
# scored blend, and both credited the same numbers, so a single reading counted double in
# the final picks. One entry per mechanism.
DUPLICATE_CHARTS = {'turning': 'turning'}   # chart name -> the LINK_SPECS entry it duplicates

for _name in CHARTS:
    if _name in DUPLICATE_CHARTS:
        continue
    LINK_SPECS[f'chart_{_name}'] = _chart_link(_name)
for _step in (9, -9, 10, -10, 11, -11):
    LINK_SPECS[f'grid{_step:+d}'] = _grid_offset(_step)
for _p in range(1, 6):
    LINK_SPECS[f'pos{_p}'] = _positional_link(_p)

LINK_LABELS = {
    'carry': "the same number came back",
    'mach_carry': "a machine number became a winning number",
    'double': "double a number",
    'mirror': "the number's mirror across the board (23 becomes 68)",
    'plus1': "the number one up",
    'minus1': "the number one down",
    'turning': "turn the number's digits around (23 becomes 32)",
    'sum_pair': "add two of the numbers together",
    'diff_pair': "subtract one number from another",
    'sum_triple': "add three of the numbers together",
    'terminal_family': "another number ending in the same digit",
    'root_family': "another number from the same digital-root family",
}
for _name in CHARTS:
    if _name in DUPLICATE_CHARTS:
        continue
    LINK_LABELS[f'chart_{_name}'] = f"its partner on the {_name.replace('_', ' ')} chart"
_GRID_WORDS = {9: "diagonal neighbour on the chart grid (+9)", -9: "diagonal neighbour on the chart grid (−9)",
               10: "the cell directly below on the chart grid (+10)", -10: "the cell directly above on the chart grid (−10)",
               11: "the other diagonal neighbour (+11)", -11: "the other diagonal neighbour (−11)"}
for _step, _w in _GRID_WORDS.items():
    LINK_LABELS[f'grid{_step:+d}'] = _w
for _p in range(1, 6):
    LINK_LABELS[f'pos{_p}'] = f"the number that sat in position {_p} (lowest to highest) comes back"

MATCHER_LABELS = {
    'pair': "a pair of this week's own numbers turned up together in that week",
    'pair_structure': "a pair of this week's own numbers turned up together in that week, AND "
                      "the week was built like this one (spacing, spread, clustering, draw order)",
    'triple': f"at least {TRIPLE_ANCHOR_MIN} of this week's own numbers turned up together in that week",
    'structure': "the week had the same STRUCTURE as this one — same spacing between numbers, "
                 "same spread, same clustering, same draw-order shape, without needing to share "
                 "any numbers at all",
    'image': "the chart picture over the last few weeks looked like it does now",
    'profile': "the draw had the same shape as this one (total, odd/even, spread, repeat)",
    'overlap': f"the draw shared at least {OVERLAP_MIN} numbers with the current one",
    'any': "every week in history (no situation filter — the control)",
}
# 'structure' and 'profile' are IDENTITY-INDEPENDENT: they match how a week is built, never
# which numbers were in it. 'image' and 'overlap' are surface matchers (they score literal
# shared cells/numbers) and are kept deliberately as the comparison baseline -- if a plan
# scores no better under a structural matcher than under a surface one, the structure was
# not doing the work, and the report is expected to make that visible.
STRUCTURAL_MATCHERS = ('structure', 'profile')


def link_candidates(history, drop_index, link, lag, cache=None):
    """The numbers link `link` proposes for the draw at `drop_index`, reading from the
    source week `lag` draws earlier. `drop_index` may be len(history) -- that is the
    live case: the next, not-yet-drawn week.

    A plan's candidate set depends only on (source draw, link), never on the matcher or
    the lag that reached it, so results are memoized on (source_index, link): the same
    source week is read by every matcher and by several lags, and without this the
    engine recomputes the identical set thousands of times per prediction."""
    src = drop_index - lag
    if src < 0 or src >= len(history):
        return set()
    key = (src, link)
    if cache is not None and key in cache:
        return cache[key]
    fn = LINK_SPECS.get(link)
    if fn is None:
        return set()
    out = fn(history[src])
    if cache is not None:
        cache[key] = out
    return out


# ---------------------------------------------------------------- structural equivalence
def structure_signature(draw):
    """STRUCTURAL EQUIVALENCE, the strict rule: describe HOW a week is built, never WHICH
    numbers built it. Two weeks sharing no numbers at all can be structurally identical;
    two weeks sharing four numbers can be structurally different. Every term below is
    invariant to number identity:

      - the four gaps between consecutive sorted numbers (the spacing pattern)
      - the span, and how many of the nine decade-buckets the draw occupies (clustering)
      - the odd/even balance
      - the DRAW-ORDER permutation: where each sorted rank sat in the physical draw order
        (relative positioning -- 'the biggest number came out first' is a structural fact)
      - interaction behaviour: how many internal pairs share a last digit, share a
        digital root, or are literal neighbours (gap of 1)
    """
    w = sorted(_win(draw))
    if len(w) < 5:
        return None
    gaps = [b - a for a, b in zip(w, w[1:])]
    rank = {x: i for i, x in enumerate(w)}
    perm = [rank[x] for x in _win(draw)]
    term_pairs = sum(1 for i in range(5) for j in range(i + 1, 5) if w[i] % 10 == w[j] % 10)
    root_pairs = sum(1 for i in range(5) for j in range(i + 1, 5) if group_of(w[i]) == group_of(w[j]))
    consecutive = sum(1 for g in gaps if g == 1)
    return ([g / 89.0 for g in gaps]
            + [(w[-1] - w[0]) / 89.0,
               len({(x - 1) // 10 for x in w}) / 5.0,
               sum(1 for x in w if x % 2) / 5.0]
            + [p / 4.0 for p in perm]
            + [term_pairs / 10.0, root_pairs / 10.0, consecutive / 4.0])


def _structural_matches(history, idxs, top_n):
    """Rank candidate drop-rows by how structurally equivalent their seed week is to the
    current week. Features are z-normalized across the candidate pool so no single term
    (e.g. span, which has the widest raw range) dominates the distance."""
    n = len(history)
    cur = structure_signature(history[n - 1])
    if cur is None:
        return []
    rows = []
    for i in idxs:
        sig = structure_signature(history[i - 1])
        if sig is not None:
            rows.append((i, sig))
    if not rows:
        return []
    dim = len(cur)
    allv = [s for _, s in rows] + [cur]
    means = [sum(v[k] for v in allv) / len(allv) for k in range(dim)]
    stds = []
    for k in range(dim):
        var = sum((v[k] - means[k]) ** 2 for v in allv) / len(allv)
        stds.append(math.sqrt(var) or 1.0)
    curz = [(cur[k] - means[k]) / stds[k] for k in range(dim)]
    scored = []
    for i, sig in rows:
        dist = math.sqrt(sum(((sig[k] - means[k]) / stds[k] - curz[k]) ** 2 for k in range(dim)) / dim)
        scored.append((i, 1.0 / (1.0 + dist)))
    scored.sort(key=lambda t: -t[1])
    return scored[:top_n]


# ---------------------------------------------------------------- the matchers
def _profile_bounds(history):
    sums = [sum(d['win']) for d in history[:-1]]
    spans = [max(d['win']) - min(d['win']) for d in history[:-1]]
    return _tercile_bounds(sums), _tercile_bounds(spans)


def matched_situations(history, matcher, window=WINDOW, top_n=MATCH_TOP_N):
    """Drop-row indices of past situations resembling the present, per `matcher`.

    A situation's window is [i-window, i-1] and its drop is row i. Every returned i
    satisfies i < len(history) - window, so BOTH the matched window and its drop lie
    strictly before the current window -- without that guard the best "matches" are the
    ones sharing rows with the present and the drop evidence is partly the current draws
    themselves (the echo failure mode this project measures for explicitly).

    Returns [(drop_index, similarity, anchor)] where `anchor` is the tuple of CURRENT
    draw numbers that the matched week shares (the pair that anchored it), or () for the
    identity-independent matchers. Similarity is 1.0 for the non-graded matchers.
    At most MAX_SITUATIONS are kept: for graded matchers the best-scoring ones, for the
    rest the most RECENT ones (nearest to the situation being predicted).
    """
    n = len(history)
    limit = n - window                      # exclusive upper bound on the drop row
    lo = max(window, BACKTRACE_DEPTH)       # need a full backtrace runway behind the drop
    if limit <= lo:
        return []
    idxs = range(lo, limit)
    cur = set(history[n - 1]['win'])

    def _anchored(min_shared):
        """THE INITIAL STEP: weeks holding a pair (or more) of this week's own numbers,
        each tagged with exactly which numbers did the anchoring."""
        out = []
        for i in idxs:
            shared = tuple(sorted(set(history[i - 1]['win']) & cur))
            if len(shared) >= min_shared:
                out.append((i, 1.0, shared))
        return out

    if matcher == 'pair':
        return _anchored(PAIR_ANCHOR_MIN)[-MAX_SITUATIONS:]

    if matcher == 'triple':
        return _anchored(TRIPLE_ANCHOR_MIN)[-MAX_SITUATIONS:]

    if matcher == 'pair_structure':
        # Anchor on a shared pair FIRST, then refine that set by structural equivalence.
        anchored = _anchored(PAIR_ANCHOR_MIN)
        if not anchored:
            return []
        anchors = {i: a for i, _s, a in anchored}
        ranked = _structural_matches(history, [i for i, _s, _a in anchored], top_n)
        return [(i, s, anchors.get(i, ())) for i, s in ranked]

    if matcher == 'structure':
        return [(i, s, ()) for i, s in _structural_matches(history, idxs, top_n)]

    if matcher == 'any':
        return [(i, 1.0, ()) for i in idxs][-MAX_SITUATIONS:]

    if matcher == 'overlap':  # legacy alias for 'pair'
        return _anchored(PAIR_ANCHOR_MIN)[-MAX_SITUATIONS:]

    if matcher == 'profile':
        sum_b, span_b = _profile_bounds(history)
        cur_key = joint_condition_key(history[n - 1], history[n - 2], sum_b, span_b)
        out = []
        for i in idxs:
            if joint_condition_key(history[i - 1], history[i - 2], sum_b, span_b) == cur_key:
                out.append((i, 1.0, ()))
        return out[-MAX_SITUATIONS:]

    if matcher == 'image':
        # 2D normalized cross-correlation of the two-channel binary chart image.
        # Graded, and identity-dependent (overlap of literal cells) -- kept as a baseline.
        W = binary_matrix(history, 'win')
        M = binary_matrix(history, 'mach')
        tw = [W[n - window + k] for k in range(window)]
        tm = [M[n - window + k] for k in range(window)]
        t_ones = sum(r.bit_count() for r in tw) + sum(r.bit_count() for r in tm)
        if not t_ones:
            return []
        scored = []
        for i in idxs:
            s = i - window
            overlap = sum((W[s + k] & tw[k]).bit_count() for k in range(window))
            overlap += sum((M[s + k] & tm[k]).bit_count() for k in range(window))
            c_ones = (sum(W[s + k].bit_count() for k in range(window))
                      + sum(M[s + k].bit_count() for k in range(window)))
            if c_ones:
                scored.append((i, overlap / math.sqrt(c_ones * t_ones)))
        scored.sort(key=lambda t: -t[1])
        return [(i, s, tuple(sorted(set(history[i - 1]['win']) & cur))) for i, s in scored[:top_n]]

    raise ValueError(f"unknown matcher: {matcher}")


# Pair-anchored matchers come FIRST: anchoring on a pair of the current draw's own
# numbers is the initial step of the reading. The rest are refinements and controls.
MATCHERS = ('pair', 'pair_structure', 'triple', 'structure', 'profile', 'image', 'any')
ANCHORED_MATCHERS = ('pair', 'pair_structure', 'triple', 'overlap')


# ---------------------------------------------------------------- plan induction
def _evaluate_plan(history, situations, link, lag, cache=None):
    """Apply (link, lag) to every matched situation and measure what it proposed against
    what actually dropped. Returns (hits, proposals, evidence) -- evidence is the
    BACKTRACE: per situation, which dropped numbers this plan explained."""
    hits = proposals = 0
    evidence = []
    anchors = Counter()
    # Per-situation yields as well as the pooled one. A situation is the unit of
    # independent evidence here: within one, a link's candidates are all measured
    # against the SAME five drawn numbers, so they are one observation, not |C| of them.
    # See _summarize_yields for what the two estimators are each good for.
    per_situation = []
    for i, sim, anchor in situations:
        cand = link_candidates(history, i, link, lag, cache)
        if not cand or len(cand) > CANDIDATE_CAP:
            continue
        dropped = set(history[i]['win'])
        explained = sorted(cand & dropped)
        hits += len(explained)
        proposals += len(cand)
        per_situation.append(len(explained) / len(cand))
        if anchor:
            anchors[anchor] += 1
        if explained:
            evidence.append({
                'drop_date': history[i]['date'],
                'source_date': history[i - lag]['date'],
                'explained': explained,
                'dropped': list(history[i]['win']),
                'n_proposed': len(cand),
                'similarity': sim,
                # WHICH of the current draw's numbers anchored this lookalike week.
                'anchor': list(anchor),
            })
    return hits, proposals, evidence, anchors, per_situation


def _summarize_yields(hits, proposals, per_situation):
    """Two views of the same evidence, both reported, only one used for ranking.

    `rate`/`weight` pool every proposed candidate as a trial. That is what the module
    has always ranked on, and its denominator mixes two unrelated things: how many
    lookalike weeks the matcher found, and how many numbers the link proposes per week.
    A link proposing forty numbers banks forty "trials" a week against a two-number
    link's two, so it is shrunk far less at the same measured yield -- breadth buys rank
    without buying accuracy.

    `sit_rate`/`sit_weight` weight every matched situation equally and count the
    situations, not the candidates, as the sample. That denominator is the number of
    independent observations the plan actually has, so it compares a narrow plan and a
    broad one on the same footing. Reported alongside so the gap between the two is
    visible; changing which one RANKS plans changes every pick this mode makes, so that
    stays a deliberate decision rather than a silent one.
    """
    rate = hits / proposals if proposals else 0.0
    n_sit = len(per_situation)
    sit_rate = sum(per_situation) / n_sit if n_sit else 0.0
    return {
        'rate': rate,
        'weight': _wilson_lower_bound(rate, proposals),
        'sit_rate': sit_rate,
        'sit_weight': _wilson_lower_bound(sit_rate, n_sit),
        'n_contributing': n_sit,
    }


def discover_plans(history, matchers=MATCHERS, depth=BACKTRACE_DEPTH, window=WINDOW):
    """Learn every plan the history supports: for each matcher, for each link, at each
    lag 1..depth, measure the plan's yield and keep its backtrace evidence.

    Returns a list of plan dicts sorted best-first (by Wilson-shrunk yield). Every plan
    -- qualifying or not, echo-excluded or not -- is returned; filtering is the caller's
    (and the report's) business, so nothing is silently dropped."""
    n = len(history)
    plans = []
    cache = {}  # (source_index, link) -> candidate set; shared across matchers and lags
    for matcher in matchers:
        situations = matched_situations(history, matcher, window=window)
        if len(situations) < MIN_SITUATIONS:
            # Too little evidence to learn from. Not silently dropped: plan_report()
            # derives the missing matchers by comparing MATCHERS against what came back,
            # and says so in the report.
            continue
        for link in LINK_SPECS:
            for lag in range(1, depth + 1):
                hits, proposals, evidence, anchors, per_sit = _evaluate_plan(
                    history, situations, link, lag, cache)
                if proposals <= 0:
                    continue
                top_anchors = [{'numbers': list(a), 'weeks': c} for a, c in anchors.most_common(5)]
                y = _summarize_yields(hits, proposals, per_sit)
                rate, weight = y['rate'], y['weight']
                echo = (link, lag) in ECHO_EXCLUDED
                # Gate on the situations that actually CONTRIBUTED, not on how many the
                # matcher returned: a situation whose source week yields no candidates
                # (or more than CANDIDATE_CAP) is skipped above and is not evidence.
                qualifies = (not echo and proposals >= MIN_PROPOSALS
                             and y['n_contributing'] >= MIN_SITUATIONS)
                plans.append({
                    'id': f"{matcher}|{link}|lag{lag}",
                    'matcher': matcher, 'link': link, 'lag': lag,
                    'situations': len(situations),
                    'n_contributing': y['n_contributing'],
                    'hits': hits, 'proposals': proposals,
                    'rate': rate, 'chance': CHANCE, 'weight': weight,
                    'sit_rate': y['sit_rate'], 'sit_weight': y['sit_weight'],
                    'lift': (rate / CHANCE) if CHANCE else 0.0,
                    'p_value': _binom_tail_p(hits, proposals, CHANCE),
                    'echo_excluded': echo, 'qualifies': qualifies,
                    'anchored': matcher in ANCHORED_MATCHERS,
                    'top_anchors': top_anchors,
                    'evidence': evidence[:8],
                    'now': sorted(link_candidates(history, n, link, lag, cache)),
                })
    plans.sort(key=lambda p: (-p['weight'], p['p_value']))
    return plans


def plan_name(plan):
    """A human name for a plan, for headings and table rows. The machine id
    ('profile|turning|lag8') stays available for auditing, but nobody should have to
    read it to understand what the plan does."""
    how = LINK_LABELS.get(plan['link'], plan['link'])
    how = how[0].upper() + how[1:]
    wk = "last week's draw" if plan['lag'] == 1 else f"the draw {plan['lag']} weeks back"
    return f"{how} — using {wk}"


def matcher_short(plan):
    """Two or three words for HOW this plan finds its lookalike weeks."""
    return {
        'pair': "weeks holding a pair from this draw",
        'pair_structure': "weeks holding a pair from this draw, built the same way",
        'triple': "weeks holding three of this draw's numbers",
        'structure': "weeks built the same way",
        'profile': "weeks with the same shape",
        'image': "weeks with similar numbers on the chart",
        'overlap': "weeks sharing numbers with now",
        'any': "every week (no filter)",
    }.get(plan['matcher'], plan['matcher'])


def plan_sentence(plan):
    """The one plain-language line a non-technical reader sees for a plan."""
    where = MATCHER_LABELS.get(plan['matcher'], plan['matcher'])
    how = LINK_LABELS.get(plan['link'], plan['link'])
    wk = "the week before" if plan['lag'] == 1 else f"{plan['lag']} weeks before"
    return (f"When {where}, the numbers that dropped tended to arrive as: {how} "
            f"from {wk}. Across {plan['situations']} such situations this plan proposed "
            f"{plan['proposals']:,} numbers and {plan['hits']:,} of them actually dropped "
            f"({plan['rate']:.1%} vs {plan['chance']:.1%} by luck).")


def plan_steps(plan, history):
    """The step-by-step statement of a plan: how it was learned and how it was applied,
    in the order the reasoning actually runs. Every predicted number is traceable through
    these steps back to a specific earlier draw."""
    n = len(history)
    src_i = n - plan['lag']
    src = history[src_i] if 0 <= src_i < n else None
    wk = "the week before" if plan['lag'] == 1 else f"{plan['lag']} weeks before"
    anchors = plan.get('top_anchors') or []
    if plan.get('anchored') and anchors:
        a0 = anchors[0]
        anchor_bits = ", ".join(f"{' & '.join(map(str, a['numbers']))} ({a['weeks']} week"
                                + ("" if a['weeks'] == 1 else "s") + ")"
                                for a in anchors[:3])
        step1 = (f"**Start from a pair in this week's draw.** Take a pair of the numbers just drawn and "
                 f"hunt for the weeks where that same pair came out together. The pairs that anchored "
                 f"the search: {anchor_bits}. That gave {plan['situations']} lookalike weeks in total"
                 + (" — then those weeks were narrowed further to the ones built like this one "
                    "(spacing, spread, clustering, draw order)."
                    if plan['matcher'] == 'pair_structure' else "."))
    else:
        step1 = (f"**Find the lookalike weeks.** Go through the whole history and pick out the weeks where "
                 f"{MATCHER_LABELS.get(plan['matcher'], plan['matcher'])}. There were {plan['situations']} of them"
                 + (" — note this one compares how the week was BUILT, without needing a shared number, "
                    "so it is the comparison arm against the pair-anchored plans."
                    if plan['matcher'] in STRUCTURAL_MATCHERS else "."))
    steps = [
        step1,
        f"**See what came next.** For each of those weeks, look at the very next draw and write down "
        f"the numbers that dropped.",
        f"**Work out where those numbers came from.** Trace each one back through the earlier weeks. "
        f"The same answer kept coming up: {LINK_LABELS.get(plan['link'], plan['link'])}, reading from "
        f"{wk}. That is the plan.",
        f"**Check whether it actually works.** Following that method across those weeks suggested "
        f"{plan['proposals']:,} numbers, and {plan['hits']:,} of them really dropped — "
        f"{plan['rate']:.1%} right, where pure luck gets {plan['chance']:.1%}.",
    ]
    if src:
        steps.append(
            f"**Now do the same thing for this week.** Take the draw of {src['date']} "
            f"({' · '.join(map(str, sorted(src['win'])))}) and apply that same method to it.")
        steps.append(
            f"**The numbers it gives:** {' · '.join(map(str, plan['now']))}.")
    return steps


def number_derivations(history, numbers, plans):
    """CRITICAL CONSTRAINT: every predicted number must be explainable. For each number,
    the concrete derivation -- which plan produced it, from which earlier draw, by which
    mechanism. A number with no derivation is not returned as a pick."""
    out = {}
    n = len(history)
    for num in numbers:
        rows = []
        for p in plans:
            if p['qualifies'] and num in p['now']:
                src_i = n - p['lag']
                src = history[src_i] if 0 <= src_i < n else None
                rows.append({
                    'plan': p['id'],
                    'mechanism': LINK_LABELS.get(p['link'], p['link']),
                    'from_draw': src['date'] if src else None,
                    'from_numbers': sorted(src['win']) if src else [],
                    'lag': p['lag'], 'rate': p['rate'], 'weight': p['weight'],
                })
        rows.sort(key=lambda r: -r['weight'])
        out[num] = rows
    return out


def plan_comparison(primary, secondaries):
    """Why the primary plan was selected, and how it differs from the runners-up."""
    if not primary:
        return {'why_primary': 'No plan held up well enough on this history.', 'differences': []}
    why = (f"It has the best record once we take into account how much evidence is behind it: "
           f"{primary['rate']:.1%} of the {primary['proposals']:,} numbers it suggested actually "
           f"dropped, where luck alone gets {primary['chance']:.1%}, and it was learned from "
           f"{primary['situations']} past weeks. We rank plans by that evidence-adjusted record, not "
           f"by the raw percentage — otherwise a plan that got lucky in three weeks would beat one "
           f"that has been right across hundreds.")
    diffs = []
    for s in secondaries[:6]:
        bits = []
        if s['matcher'] != primary['matcher']:
            bits.append(f"picks its lookalike weeks a different way ({matcher_short(s)})")
        if s['link'] != primary['link']:
            bits.append(f"gets its numbers a different way ({LINK_LABELS.get(s['link'], s['link'])})")
        if s['lag'] != primary['lag']:
            bits.append(f"looks {s['lag']} weeks back instead of {primary['lag']}")
        shared = sorted(set(s['now']) & set(primary['now']))
        bits.append(f"agrees with the top plan on {len(shared)} number"
                    + ("" if len(shared) == 1 else "s")
                    + (f": {', '.join(map(str, shared))}" if shared else ""))
        diffs.append({'plan': s['id'], 'name': plan_name(s), 'rate': s['rate'],
                      'numbers': s['now'], 'difference': "; ".join(bits)})
    return {'why_primary': why, 'differences': diffs}


# ---------------------------------------------------------------- scoring & report
def scoring_set(plans, limit=TOP_PLANS_SCORED):
    """The plans that get a vote in the blend: the best-evidenced `limit` of them, ONE
    PER MECHANISM. Returns (chosen, collapsed).

    A plan's numbers for the coming week come from link_candidates(history, n, link,
    lag) -- (link, lag) and nothing else. The matcher decides which past weeks the plan
    was MEASURED on, never what it proposes now. So every plan sharing a (link, lag)
    proposes exactly the same numbers, and letting each of them add its weight votes one
    mechanism several times: on Friday Bonanza the twelve scoring plans were only eight
    mechanisms, with `plus1` at lag 8 voting three times for one set of numbers because
    it happened to qualify under three matchers.

    The extra rows are not extra evidence, they are the same reading measured against
    three different reference classes. The best-evidenced one keeps the vote; the rest
    are returned in `collapsed` so the report can show them rather than lose them.
    """
    chosen, kept_by_mech, collapsed = [], {}, []
    for p in plans:
        if not p['qualifies']:
            continue
        mech = (p['link'], p['lag'])
        if mech in kept_by_mech:
            collapsed.append({'dropped': p['id'], 'kept': kept_by_mech[mech],
                              'mechanism': f"{p['link']}|lag{p['lag']}",
                              'matcher': p['matcher'], 'rate': p['rate'],
                              'weight': p['weight']})
            continue
        kept_by_mech[mech] = p['id']
        chosen.append(p)
        if len(chosen) >= limit:
            break
    return chosen, collapsed


def plan_scores(history, plans=None):
    """Blended score from the learned plans: each qualifying plan credits the numbers it
    proposes for the coming week, weighted by its own Wilson-shrunk measured yield --
    one vote per mechanism, see scoring_set().
    Returns (scores, report)."""
    plans = plans if plans is not None else discover_plans(history)
    scores = {k: 0.0 for k in range(1, 91)}
    scored, collapsed = scoring_set(plans)
    for p in scored:
        for x in p['now']:
            scores[x] += p['weight']
    report = plan_report(history, plans, scored)
    report['collapsed_duplicates'] = collapsed
    # No untraceable picks: attach the derivation of every number the blend puts on top.
    top = [k for k, _ in sorted(scores.items(), key=lambda t: (-t[1], t[0]))[:10] if scores[k] > 0]
    report['derivations'] = number_derivations(history, top, plans)
    return scores, report


def plan_report(history, plans, scored=None):
    """The deliverable the project owner asked for: how EACH plan produced its numbers.

    `best` is the chosen plan (highest Wilson-shrunk yield among qualifying plans);
    `others` keeps every runner-up with its own numbers and track record; `echoes` keeps
    the quarantined lag-1 carry plans so their exclusion is visible rather than silent."""
    scored = scored if scored is not None else scoring_set(plans)[0]
    qualifying = [p for p in plans if p['qualifies']]
    echoes = [p for p in plans if p['echo_excluded']]
    best = qualifying[0] if qualifying else None
    secondaries = qualifying[1:20]
    # Structural vs surface: the honest side-by-side the strict matching rule demands.
    best_structural = next((p for p in qualifying if p['matcher'] in STRUCTURAL_MATCHERS), None)
    best_surface = next((p for p in qualifying if p['matcher'] not in STRUCTURAL_MATCHERS), None)
    # Anchored vs unanchored: does starting from a pair of this week's own numbers actually
    # beat searching without an anchor? 'any' is the control that answers it.
    best_anchored = next((p for p in qualifying if p.get('anchored')), None)
    best_unanchored = next((p for p in qualifying if not p.get('anchored')), None)
    return {
        'best': best,
        'best_sentence': plan_sentence(best) if best else None,
        'best_numbers': best['now'] if best else [],
        'best_steps': plan_steps(best, history) if best else [],
        'scored': scored,
        'others': secondaries,
        'comparison': plan_comparison(best, secondaries),
        'echoes': echoes[:6],
        'n_plans_evaluated': len(plans),
        'collapsed_duplicates': [],
        'n_qualifying': len(qualifying),
        'matchers_used': sorted({p['matcher'] for p in qualifying}),
        'best_structural': best_structural,
        'best_surface': best_surface,
        'best_anchored': best_anchored,
        'best_unanchored': best_unanchored,
        # Matchers that found too few lookalike weeks to learn anything from (e.g. 'triple'
        # on most games: sharing 3+ numbers with the current draw is genuinely rare).
        # Disclosed rather than silently dropped.
        'matchers_skipped': [m for m in MATCHERS if m not in {p['matcher'] for p in plans}],
        # How many lookalike weeks each matcher found. This is not trivia: `weight` is a
        # Wilson LOWER bound, so a matcher that kept 250 situations is shrunk far less
        # than one that kept 30 at the same measured yield, and the ranking partly
        # reflects that rather than performance. Reported so the comparison rows above
        # ('does the anchor matter?', 'does structure matter?') can be read with the
        # sample sizes visible instead of as a like-for-like contest.
        'matcher_situations': {m: n for m, n in sorted(
            {p['matcher']: p['situations'] for p in plans}.items())},
    }


def explain_number(history, number, plans=None):
    """Which plans put `number` on this week's board, and how each of them derived it --
    the per-number view of the same evidence."""
    plans = plans if plans is not None else discover_plans(history)
    out = []
    for p in plans:
        if p['qualifies'] and number in p['now']:
            src = len(history) - p['lag']
            out.append({
                'plan': p['id'], 'matcher': p['matcher'], 'link': p['link'], 'lag': p['lag'],
                'sentence': plan_sentence(p),
                'from_draw': history[src]['date'] if 0 <= src < len(history) else None,
                'rate': p['rate'], 'chance': p['chance'], 'weight': p['weight'],
                'situations': p['situations'], 'proposals': p['proposals'],
            })
    out.sort(key=lambda e: -e['weight'])
    return out


# ---------------------------------------------------------------- noise control
def bootstrap_best_plan_pvalue(history, iterations=40, seed=0, depth=BACKTRACE_DEPTH,
                                matchers=MATCHERS):
    """Family-wise (max-statistic) test, the same arbiter spatial_engine applies to its
    key search: re-run the WHOLE plan search on structure-destroyed synthetic histories
    (same shape, same density, numbers redrawn uniformly) and report how often noise
    produces a better best-plan than the real chart does.

    A p-value above 0.05 means the winning plan is indistinguishable from what pure
    chance hands a search of this size -- i.e. the plan is apophenia, however convincing
    its story reads.

    The search tested defaults to the FULL one, so the plan being judged is the plan the
    report displays. It used to default to matchers=('image','any'), depth=4 for cost:
    internally consistent (the real best was recomputed under the same reduction) but it
    left the verdict attached to a plan the reduced search had never considered -- on
    Monday Special the report's best plan was structure|carry|lag6 while the test was
    judging image|mach_carry|lag3, a different matcher family at a lag outside the tested
    depth. The full search costs ~0.7s per synthetic history, so 40 iterations is ~30s;
    a caller that needs it cheaper can still pass a reduction, and `search` in the result
    records what was actually tested along with whether it matched the full search."""
    import random
    rng = random.Random(seed)
    real = discover_plans(history, matchers=matchers, depth=depth)
    real_q = [p for p in real if p['qualifies']]
    real_best = real_q[0]['weight'] if real_q else 0.0
    ge = 0
    for _ in range(iterations):
        fake = discover_plans(_synthetic_history(history, rng), matchers=matchers, depth=depth)
        fake_q = [p for p in fake if p['qualifies']]
        fake_best = fake_q[0]['weight'] if fake_q else 0.0
        ge += fake_best >= real_best
    p = ge / iterations
    resolution = 1.0 / iterations
    if p > 0.05:
        verdict = ('indistinguishable from noise — treat every plan below as a reading, '
                   'not a finding')
    elif p > 0.005:
        verdict = (f'borderline: only {ge} of {iterations} noise charts beat it, but at this '
                   f'iteration count the p-value resolves no finer than {resolution:.3f}, and '
                   f'testing 7 games makes one borderline result per sweep the EXPECTED outcome '
                   f'— not evidence of a real edge')
    else:
        verdict = ('the best plan beats what this size of search finds in pure noise, across '
                   'every synthetic chart tried — worth re-testing at higher iterations before '
                   'believing it')
    full_search = set(matchers) == set(MATCHERS) and depth == BACKTRACE_DEPTH
    return {
        'real_best_weight': real_best,
        'real_best_plan': real_q[0]['id'] if real_q else None,
        'bootstrap_p': p, 'iterations': iterations, 'n_noise_wins': ge,
        'resolution': resolution,
        'search': {'matchers': list(matchers), 'depth': depth, 'is_full_search': full_search},
        # True when the plan judged here is the same one plan_report() calls `best`.
        # A reduced search can pick a different winner, and then the verdict below is
        # about that other plan, not the one on screen.
        'judged_the_displayed_plan': full_search,
        'verdict': verdict,
    }
