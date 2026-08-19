"""Feature engineering.

Two feature sets live here:

1. The LEGACY 10-feature set (`FEAT_NAMES`, `ml_features`) plus the hand-rolled NumPy
   `LogReg` / `train_ml` / `ml_scores` — moved verbatim (unchanged behavior) from the
   original predictor.py so the `ml` CLI/backtest mode stays byte-identical.

2. The EXTENDED ~36-feature set (`EXT_FEAT_NAMES`, `build_extended_features`) used by
   the new scikit-learn ensemble (sklearn_models.py) and, via its own per-draw
   representation, the deep model (deep_model.py). It generalizes several legacy
   features (full-history pairwise co-occurrence instead of last-draw-only, chart
   pointers weighted by measured transfer rate instead of raw counts, a true
   percentile freq_rank instead of a duplicate of freq_all) and adds positional,
   parity, decile, entropy, seasonal, streak, game-code, a full parallel set of
   machine-number features, and a pair-tracing/addition block that operationalizes
   common lottery-paper reading strategies as measured quantities (see the comment
   above EXT_FEAT_NAMES for what each one encodes).
"""
import math
from collections import Counter, defaultdict

from .charts import CHARTS
from .config import EXP_GAP, GAME_INDEX
from . import transform_engine

# ---------------------------------------------------------------- legacy (unchanged)
FEAT_NAMES = ['freq_all', 'freq_30', 'freq_10', 'wfreq_60', 'gap_ratio', 'cooc_last',
              'chart_in', 'chart_in_mach', 'last_draw_sum', 'freq_rank']


def ml_features(history, last):
    """90 x F feature matrix from a game's history (last = most recent draw)."""
    import numpy as np
    n = len(history)
    F = np.zeros((90, len(FEAT_NAMES)))
    freq = Counter(); f30 = Counter(); f10 = Counter(); wf = defaultdict(float); last_seen = {}
    for i, d in enumerate(history):
        for x in d['win']:
            freq[x] += 1; wf[x] += 0.5 ** ((n - 1 - i) / 60.0); last_seen[x] = i
            if i >= n - 30: f30[x] += 1
            if i >= n - 10: f10[x] += 1
    cin = Counter(); cinm = Counter()
    if last:
        for mp in CHARTS.values():
            for x in last['win']:
                if x in mp: cin[mp[x]] += 1
            for x in last['mach']:
                if x in mp: cinm[mp[x]] += 1
    co = Counter()
    if last:
        lastset = set(last['win'])
        for d in history[:-1]:
            s = set(d['win'])
            if s & lastset:
                for x in s: co[x] += 1
    wtot = sum(wf.values()) or 1e-9
    for k in range(1, 91):
        gap = n - last_seen.get(k, -1) - 1
        F[k - 1] = [freq.get(k, 0) / max(n, 1) * EXP_GAP, f30.get(k, 0) / 30 * EXP_GAP, f10.get(k, 0) / 10 * EXP_GAP,
                    wf.get(k, 0.0) / wtot * 90, min(gap / EXP_GAP, 4.0),
                    co.get(k, 0) / max(n, 1) * EXP_GAP, cin.get(k, 0), cinm.get(k, 0),
                    (sum(last['win']) / 225.0 if last else 1.0), freq.get(k, 0) / max(n, 1) * EXP_GAP]
    return F


class LogReg:
    def __init__(self, nfeat):
        import numpy as np
        self.w = np.zeros(nfeat); self.b = 0.0

    def fit(self, X, y, epochs=200, lr=0.1, l2=1e-4):
        import numpy as np
        w, b = self.w, self.b
        for _ in range(epochs):
            z = X @ w + b
            p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            g = X.T @ (p - y) / len(y) + l2 * w
            gb = float((p - y).mean())
            w -= lr * g; b -= lr * gb
        self.w, self.b = w, b

    def scores(self, X):
        return X @ self.w + self.b


def train_ml(history, step=1, min_hist=60):
    import numpy as np
    Xs, ys = [], []
    for i in range(min_hist, len(history), step):
        F = ml_features(history[:i], history[i - 1])
        y = np.zeros(90); y[[n - 1 for n in history[i]['win']]] = 1
        Xs.append(F); ys.append(y)
    X = np.vstack(Xs); y = np.concatenate(ys)
    m = LogReg(X.shape[1]); m.fit(X, y)
    return m


def ml_scores(history, model=None):
    if model is None:
        model = train_ml(history, step=2)
    F = ml_features(history, history[-1] if history else None)
    return {k + 1: float(s) for k, s in enumerate(model.scores(F))}


# ---------------------------------------------------------------- extended (new)
# Chart relationships (chart_in_w/chart_in_mach_w), machine numbers (freq_mach_*/
# mach_to_win_affinity) and pair-tracing/addition (pair_trace_*/sum_derived_score) are
# all deliberately treated as FEATURE INPUTS to the learned models here, not as
# standalone prediction rules -- the `charts` CLI mode still scores numbers directly
# from the charts alone (kept only as an honest benchmark: "does the folk heuristic by
# itself beat chance?"), but a real test of "do these traditions encode information
# about the next draw" is whether a model that can *see* them (RF/GBM/MLP/deep) learns
# to weight them above zero. mach_to_win_affinity, pair_trace_score and
# sum_derived_score each operationalize a specific lottery-paper reading strategy as a
# measurable quantity computed from actual history, rather than an assumed rule:
#   - mach_to_win_affinity: "a number drawn as a machine number tends to reappear as a
#     winner soon after" (measured per-number transfer rate).
#   - pattern_trace_score/pattern_trace_trials: readers take a pair OR triplet of
#     numbers from the current draw, search the past for every time that exact group
#     repeated for this game, and see what won in the draws that followed each repeat
#     ("how many weeks down"), weighting nearer follow-ups more heavily. See
#     transform_engine.pattern_trace_events for the full position-tracking detail.
#   - transform_score: generalizes the "addition" technique to a small REGISTRY of
#     arithmetic rules (sum/difference/doubling/mirror over pairs; sum/mean over
#     triplets), each weighted by its OWN measured historical transfer rate (see
#     transform_engine.measure_rule_rates) -- not one hand-picked formula assumed to
#     work, but several candidate transformation paths combined by how reliable each
#     has actually been.
#   - positional_carryover_score: weights each of the current draw's numbers by how
#     'sticky' the sorted-rank position (1st-5th) it occupies has historically been
#     (see transform_engine.positional_carryover_rates) -- operationalizes "position
#     matters, not just identity."
# Machine numbers and chart relationships are the traditional "guide" for which
# patterns to trust -- here that's left for the learned model to discover via their own
# separate features (chart_in_w/chart_in_mach_w, freq_mach_*) rather than hand-coded
# into the trace itself, so the model can find real interactions instead of an assumed
# one. The full rule/position/group detail behind these three is inspectable via
# lottery_core.transform_engine directly (used by pattern_analysis.explain()).
CHART_NAMES = ['bonanza', 'counterpart', 'malta', 'string_key', 'shadow', 'partner', 'equivalent', 'code', 'turning']

EXT_FEAT_NAMES = [
    'freq_all', 'freq_30', 'freq_10', 'wfreq_60', 'gap_ratio', 'cooc_last',
] + [f'chart_{c}_w' for c in CHART_NAMES] + [f'chart_{c}_mach_w' for c in CHART_NAMES] + [
    'last_draw_sum', 'freq_rank',
    'pair_affinity', 'pos_mean', 'pos_std', 'is_odd', 'odd_balance',
    'decile_bucket', 'decile_dev', 'entropy_all', 'entropy_30', 'entropy_10',
    'month_sin', 'month_cos', 'day_of_month', 'hit_last_draw', 'hit_2ago', 'game_code',
    'freq_mach_all', 'freq_mach_30', 'wfreq_mach_60', 'gap_ratio_mach',
    'hit_last_draw_mach', 'hit_2ago_mach', 'mach_to_win_affinity',
    'pattern_trace_score', 'pattern_trace_trials', 'transform_score', 'positional_carryover_score',
]
MACH_LOOKAHEAD = 3  # "soon after" window for mach_to_win_affinity


def _entropy(counter, total):
    if total <= 0: return 0.0
    ent = 0.0
    for v in counter.values():
        if v <= 0: continue
        p = v / total
        ent -= p * math.log2(p)
    return ent


def mach_to_win_affinity(history, lookahead=MACH_LOOKAHEAD):
    """For each number 1-90: of the times it appeared as a MACHINE number, what fraction
    of the time did it appear as a WINNING number within the next `lookahead` draws?
    An empirical, per-number test of the common belief that machine numbers foreshadow
    upcoming winners -- not an assumption baked into the scoring.

    An occurrence in the most recent draw (i == n-1) has no follow-up window yet -- it
    hasn't had a chance to resolve one way or the other -- so it is excluded from the
    trial count rather than counted as a guaranteed miss, which would otherwise slightly
    deflate the rate for whichever numbers happen to be machine numbers in the latest
    draw."""
    win_sets = [set(d['win']) for d in history]
    occurrences = defaultdict(list)
    for i, d in enumerate(history):
        for x in d['mach']:
            occurrences[x].append(i)
    n = len(history)
    rates, trials = {}, {}
    for k in range(1, 91):
        occ = [i for i in occurrences.get(k, []) if i + 1 < n]
        hits = sum(1 for i in occ if any(k in win_sets[j] for j in range(i + 1, min(i + 1 + lookahead, n))))
        rates[k] = hits / len(occ) if occ else 0.0
        trials[k] = len(occ)
    return rates, trials


def _chart_transfer_rates(history):
    """Same transfer-rate computation as classic.chart_scores, exposed here so the
    extended feature set can weight chart pointers by measured rate instead of raw
    count without importing classic (avoids a circular import at module load time)."""
    rates = {}
    for name, mp in CHARTS.items():
        hits = trials = 0
        for a, b in zip(history[:-1], history[1:]):
            nxt = set(b['win'])
            for x in a['win']:
                if x in mp:
                    trials += 1
                    hits += mp[x] in nxt
        rates[name] = hits / trials if trials else 0.0
    return rates


def build_extended_features(history, last, game_code, target_date=None):
    """90 x len(EXT_FEAT_NAMES) feature matrix. `last` = most recent draw (or None),
    `game_code` = the 2-letter code of the game being predicted (for pooled training
    across all 7 games), `target_date` = date of the draw being predicted (defaults to
    the last draw's date if not given -- month/day are diagnostic-only features)."""
    import numpy as np
    n = len(history)
    F = np.zeros((90, len(EXT_FEAT_NAMES)))
    if target_date is None:
        target_date = history[-1]['date'] if history else None

    freq = Counter(); f30 = Counter(); f10 = Counter(); wf = defaultdict(float); last_seen = {}
    pos_sum = defaultdict(float); pos_sq = defaultdict(float); pos_cnt = defaultdict(int)
    bucket_hits_60 = Counter()
    odd_count_30 = 0; odd_total_30 = 0
    W_bucket = min(n, 60)
    W_odd = min(n, 30)

    mfreq = Counter(); mf30 = Counter(); mwf = defaultdict(float); mach_last_seen = {}

    for i, d in enumerate(history):
        w = 0.5 ** ((n - 1 - i) / 60.0)
        sorted_win = sorted(d['win'])
        for pos, x in enumerate(sorted_win):
            freq[x] += 1; wf[x] += w; last_seen[x] = i
            if i >= n - 30: f30[x] += 1
            if i >= n - 10: f10[x] += 1
            pos_sum[x] += (pos + 1); pos_sq[x] += (pos + 1) ** 2; pos_cnt[x] += 1
            if i >= n - W_bucket: bucket_hits_60[(x - 1) // 10] += 1
            if i >= n - W_odd:
                odd_total_30 += 1
                if x % 2 == 1: odd_count_30 += 1
        for x in d['mach']:
            mfreq[x] += 1; mwf[x] += w; mach_last_seen[x] = i
            if i >= n - 30: mf30[x] += 1

    mwtot = sum(mwf.values()) or 1e-9
    mach_to_win_rate, _ = mach_to_win_affinity(history)
    trace_scores, trace_trials = transform_engine.pattern_trace_score(history)
    tform_scores, _tform_detail = transform_engine.transform_score(history, last)
    pos_scores = transform_engine.positional_carryover_score(history)

    pair_aff = Counter()
    co_last = Counter()
    if last:
        lastset = set(last['win'])
        for d in history[:-1]:
            s = set(d['win'])
            overlap = len(s & lastset)
            if overlap:
                for x in s:
                    pair_aff[x] += overlap
                    co_last[x] += 1

    chart_rates = _chart_transfer_rates(history) if last else {}
    cin_w = {name: Counter() for name in CHART_NAMES}
    cinm_w = {name: Counter() for name in CHART_NAMES}
    if last:
        for name, mp in CHARTS.items():
            rate = chart_rates.get(name, 0.0)
            for x in last['win']:
                if x in mp: cin_w[name][mp[x]] += rate
            for x in last['mach']:
                if x in mp: cinm_w[name][mp[x]] += rate

    tot_all = sum(freq.values()) or 1
    tot_30 = sum(f30.values()) or 1
    tot_10 = sum(f10.values()) or 1
    ent_all = _entropy(freq, tot_all)
    ent_30 = _entropy(f30, tot_30)
    ent_10 = _entropy(f10, tot_10)

    wtot = sum(wf.values()) or 1e-9
    odd_balance = (odd_count_30 / odd_total_30 - 0.5) if odd_total_30 else 0.0

    expected_bucket = W_bucket * 5 * 10 / 90.0
    hit_last = set(last['win']) if last else set()
    hit_2ago = set(history[-2]['win']) if len(history) >= 2 else set()
    hit_last_mach = set(last['mach']) if last else set()
    hit_2ago_mach = set(history[-2]['mach']) if len(history) >= 2 else set()

    month_sin = math.sin(2 * math.pi * target_date.month / 12.0) if target_date else 0.0
    month_cos = math.cos(2 * math.pi * target_date.month / 12.0) if target_date else 0.0
    day_of_month = (target_date.day / 31.0) if target_date else 0.5

    order = sorted(range(1, 91), key=lambda k: freq.get(k, 0))
    freq_rank_of = {k: idx / 89.0 for idx, k in enumerate(order)}

    gcode = GAME_INDEX.get(game_code, 0)

    for k in range(1, 91):
        gap = n - last_seen.get(k, -1) - 1
        cnt = pos_cnt.get(k, 0)
        if cnt > 0:
            pmean = pos_sum[k] / cnt
            pvar = max(pos_sq[k] / cnt - pmean ** 2, 0.0)
            pstd = math.sqrt(pvar)
        else:
            pmean, pstd = 3.0, 1.5
        bucket = (k - 1) // 10
        bucket_dev = ((bucket_hits_60.get(bucket, 0) - expected_bucket) / expected_bucket) if expected_bucket > 0 else 0.0
        mgap = n - mach_last_seen.get(k, -1) - 1
        F[k - 1] = [
            freq.get(k, 0) / max(n, 1) * EXP_GAP,
            f30.get(k, 0) / 30 * EXP_GAP,
            f10.get(k, 0) / 10 * EXP_GAP,
            wf.get(k, 0.0) / wtot * 90,
            min(gap / EXP_GAP, 4.0),
            co_last.get(k, 0) / max(n, 1) * EXP_GAP,
        ] + [cin_w[name].get(k, 0.0) for name in CHART_NAMES] \
          + [cinm_w[name].get(k, 0.0) for name in CHART_NAMES] \
          + [
            (sum(last['win']) / 225.0 if last else 1.0),
            freq_rank_of.get(k, 0.5),
            pair_aff.get(k, 0) / max(n, 1),
            pmean, pstd,
            1.0 if k % 2 == 1 else 0.0,
            odd_balance,
            float(bucket),
            bucket_dev,
            ent_all, ent_30, ent_10,
            month_sin, month_cos, day_of_month,
            1.0 if k in hit_last else 0.0,
            1.0 if k in hit_2ago else 0.0,
            float(gcode),
            mfreq.get(k, 0) / max(n, 1) * EXP_GAP,
            mf30.get(k, 0) / 30 * EXP_GAP,
            mwf.get(k, 0.0) / mwtot * 90,
            min(mgap / EXP_GAP, 4.0),
            1.0 if k in hit_last_mach else 0.0,
            1.0 if k in hit_2ago_mach else 0.0,
            mach_to_win_rate.get(k, 0.0),
            trace_scores.get(k, 0.0),
            min(trace_trials / 5.0, 4.0),
            tform_scores.get(k, 0.0),
            pos_scores.get(k, 0.0),
        ]
    return F
