"""Extended walk-forward backtest: a per-game walk-forward loop and comb-based
random-chance baseline for every mode -- the untrained modes (legacy stat/chart
heuristics plus the live 'pattern' system, pure functions of history) and the trained
sklearn/deep/ensemble modes (periodic retraining, independently per game -- see
sklearn_models.py's docstring for why models are no longer pooled across games). Every
mode is scored on hit-rate (>=1/>=2 picks, with Wilson confidence intervals), ROC-AUC,
and Brier score -- both pooled (aggregated across games for a single headline number)
and per-game."""
import math
from collections import Counter, defaultdict

from .config import UNTRAINED_MODES, GAMES
from . import classic
from . import ensemble as ens_mod


def _score_untrained(history, mode, cached_weights=None, all_draws=None, spatial_state=None):
    if mode == 'pattern':
        from . import pattern_analysis
        scores, _ = pattern_analysis.pattern_scores(history, weights=cached_weights,
                                                     all_draws=all_draws)
        return scores
    if mode == 'spatial':
        from . import spatial_engine
        scores, _ = spatial_engine.spatial_scores(history, state=spatial_state)
        return scores
    return classic.get_scores(history, mode)


# pattern_analysis.dynamic_weights() (the default reweighting for 'pattern') costs
# ~30x a single component_scores() call and is memoized per-history there, but every
# walk-forward test point here has a genuinely different history (one draw longer than
# the last), so that cache never hits in this loop. Measured drift in the weights from
# adding a single draw is tiny (~0.001-0.004 per weight), so refreshing this often
# rather than every single test point trades negligible weight-staleness for a large,
# honest speedup -- the same tradeoff this project already makes for trained models via
# retrain_every (see _backtest_per_game_trained below).
PATTERN_WEIGHTS_RECOMPUTE_EVERY = 20

Z95 = 1.959963984540054


def p_at_least(k, j):
    tot = math.comb(90, 5)
    return sum(math.comb(k, i) * math.comb(90 - k, 5 - i) for i in range(j, min(k, 5) + 1)) / tot


def _wilson_ci(successes, n, z=Z95):
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _summarize(hits_by_k, n_tests, score_label_pairs):
    hitrate = {}
    for k in (2, 3, 4, 5):
        c = hits_by_k[k]
        ge1 = sum(v for h, v in c.items() if h >= 1)
        ge2 = sum(v for h, v in c.items() if h >= 2)
        ge1_rate = ge1 / n_tests if n_tests else 0.0
        ge2_rate = ge2 / n_tests if n_tests else 0.0
        hitrate[str(k)] = {
            'ge1_rate': ge1_rate, 'ge1_ci': list(_wilson_ci(ge1, n_tests)),
            'ge2_rate': ge2_rate, 'ge2_ci': list(_wilson_ci(ge2, n_tests)),
            'random_ge1': p_at_least(k, 1), 'random_ge2': p_at_least(k, 2),
        }
    auc = brier = None
    if score_label_pairs:
        from sklearn.metrics import roc_auc_score, brier_score_loss
        scores_arr = [s for s, _ in score_label_pairs]
        labels_arr = [l for _, l in score_label_pairs]
        if len(set(labels_arr)) > 1:
            auc = float(roc_auc_score(labels_arr, scores_arr))
        brier = float(brier_score_loss(labels_arr, scores_arr))
    return {'hitrate': hitrate, 'auc': auc, 'brier': brier, 'n_tests': n_tests}


# ---------------------------------------------------------------- untrained modes
def _backtest_untrained(by_game, mode, min_hist, all_draws=None):
    """`all_draws` (every game's draws, date-sorted) is only used by the 'pattern'
    mode, for its cross-game component -- each test point sees the walk-forward slice
    strictly before its own date, exactly like the per-game history."""
    from . import trend_analysis as ta
    all_dates = [d['date'] for d in all_draws] if all_draws else None
    pooled_hits = {k: Counter() for k in (2, 3, 4, 5)}
    pooled_pairs = []
    pooled_n = 0
    per_game = {}
    for g, seq in by_game.items():
        hits = {k: Counter() for k in (2, 3, 4, 5)}
        pairs = []
        n_tests = 0
        cached_weights = None
        spatial_state = None
        charts2_tables = None
        since_recompute = PATTERN_WEIGHTS_RECOMPUTE_EVERY  # force a compute on the first test point
        for i in range(min_hist, len(seq)):
            hist = seq[:i]
            if mode == 'pattern':
                sub_all = (ta.slice_before(all_draws, seq[i]['date'], all_dates)
                           if all_draws else None)
                if since_recompute >= PATTERN_WEIGHTS_RECOMPUTE_EVERY:
                    from . import pattern_analysis
                    cached_weights = pattern_analysis.dynamic_weights(hist, all_draws=sub_all)
                    since_recompute = 0
                sc = _score_untrained(hist, mode, cached_weights, all_draws=sub_all)
                since_recompute += 1
            elif mode == 'spatial':
                # spatial_state (diagonal/box/key/crossover measurements) drifts slowly
                # with one extra draw -- same periodic-recompute tradeoff as the pattern
                # weights above. NCC matching itself is recomputed fresh every point.
                if since_recompute >= PATTERN_WEIGHTS_RECOMPUTE_EVERY:
                    from . import spatial_engine
                    spatial_state = spatial_engine.spatial_state(hist)
                    since_recompute = 0
                sc = _score_untrained(hist, mode, spatial_state=spatial_state)
                since_recompute += 1
            elif mode == 'plans':
                # Plans are re-learned from scratch at every test point: unlike a rate
                # table, a plan's matchers are defined RELATIVE to the current window,
                # so a cached plan set would be answering a different question. The
                # engine is fast enough (~0.4s/call) to afford it honestly.
                from . import plan_engine
                sc, _ = plan_engine.plan_scores(hist)
            elif mode == 'charts2':
                # pooled entry tables drift slowly with one extra draw -- same periodic
                # recompute as above; the tables are measured on the walk-forward slice
                # of ALL games' draws (pooling, see chart_analysis.py).
                if since_recompute >= PATTERN_WEIGHTS_RECOMPUTE_EVERY:
                    from . import chart_analysis
                    sub_all = (ta.slice_before(all_draws, seq[i]['date'], all_dates)
                               if all_draws else hist)
                    charts2_tables = chart_analysis.chart_tables(sub_all)
                    since_recompute = 0
                from . import chart_analysis
                sc = chart_analysis.chart2_scores(hist, tables=charts2_tables)
                since_recompute += 1
            else:
                sc = _score_untrained(hist, mode)
            actual = set(seq[i]['win'])
            for k in (2, 3, 4, 5):
                h = len(set(classic.picks(sc, k)) & actual)
                hits[k][h] += 1
                pooled_hits[k][h] += 1
            norm = ens_mod.normalize(sc)
            for num in range(1, 91):
                pair = (norm[num], 1 if num in actual else 0)
                pairs.append(pair); pooled_pairs.append(pair)
            n_tests += 1
        per_game[g] = _summarize(hits, n_tests, pairs)
        pooled_n += n_tests
    return {'pooled': _summarize(pooled_hits, pooled_n, pooled_pairs), 'per_game': per_game}


# ---------------------------------------------------------------- trained modes (per game)
def _train_game_model(mode, game_code, hist_g, min_hist, quick):
    from . import sklearn_models, deep_model
    if mode in ('rf', 'gbm', 'mlp'):
        return sklearn_models.train_sklearn(mode, game_code, hist_g, step=2, min_hist=min_hist, quick=quick)
    if mode == 'deep':
        model, meta = deep_model.train_deep(hist_g, min_hist=min_hist, quick=quick)
        return (model, meta)
    if mode == 'ensemble':
        sub = {}
        for name in ('rf', 'gbm', 'mlp'):
            sub[name] = sklearn_models.train_sklearn(name, game_code, hist_g, step=2, min_hist=min_hist, quick=quick)
        # No torch installed (the default for a served deployment) -- backtest the
        # ensemble without its deep component rather than failing the whole run.
        if deep_model.torch_available():
            sub['deep'] = deep_model.train_deep(hist_g, min_hist=min_hist, quick=quick)
        else:
            sub['deep'] = (None, None)
        return sub
    raise ValueError(f'unknown trained mode: {mode}')


def _score_game_model(mode, model_state, hist_g, g):
    from . import sklearn_models, deep_model
    if mode in ('rf', 'gbm', 'mlp'):
        return sklearn_models.sklearn_scores(mode, model_state, hist_g, g)
    if mode == 'deep':
        model, meta = model_state
        if model is None:
            return {k: 0.0 for k in range(1, 91)}
        return deep_model.deep_scores(model.state_dict(), meta, hist_g, g)
    if mode == 'ensemble':
        comp = {'blend': classic.get_scores(hist_g, 'blend'), 'charts': classic.get_scores(hist_g, 'charts')}
        for name in ('rf', 'gbm', 'mlp'):
            comp[name] = sklearn_models.sklearn_scores(name, model_state[name], hist_g, g)
        dmodel, dmeta = model_state['deep']
        comp['deep'] = deep_model.deep_scores(dmodel.state_dict(), dmeta, hist_g, g) if dmodel else {k: 0.0 for k in range(1, 91)}
        return ens_mod.blend_scores(comp)
    raise ValueError(f'unknown trained mode: {mode}')


def _backtest_per_game_trained(by_game, mode, min_hist, retrain_every, quick):
    """Walk-forward, with the model retrained periodically -- independently for each
    game, using only that game's own history up to the test point. No game's model is
    ever fit on another game's data."""
    pooled_hits = {k: Counter() for k in (2, 3, 4, 5)}
    pooled_pairs = []
    pooled_n = 0
    per_game = {}
    for g, seq in by_game.items():
        hits = {k: Counter() for k in (2, 3, 4, 5)}
        pairs = []
        n_tests = 0
        model_state = None
        since_retrain = retrain_every  # force a retrain on the first eligible draw
        for i in range(min_hist, len(seq)):
            if model_state is None or since_retrain >= retrain_every:
                model_state = _train_game_model(mode, g, seq[:i], min_hist, quick)
                since_retrain = 0
            hist = seq[:i]
            sc = _score_game_model(mode, model_state, hist, g)
            actual = set(seq[i]['win'])
            for k in (2, 3, 4, 5):
                h = len(set(classic.picks(sc, k)) & actual)
                hits[k][h] += 1
                pooled_hits[k][h] += 1
            norm = ens_mod.normalize(sc)
            for num in range(1, 91):
                pair = (norm[num], 1 if num in actual else 0)
                pairs.append(pair); pooled_pairs.append(pair)
            n_tests += 1; since_retrain += 1
        per_game[g] = _summarize(hits, n_tests, pairs)
        pooled_n += n_tests
    return {'pooled': _summarize(pooled_hits, pooled_n, pooled_pairs), 'per_game': per_game}


# ---------------------------------------------------------------- entry point
def backtest(draws, modes=None, min_hist=100, retrain_every=75, quick=False, progress=None):
    modes = tuple(modes) if modes else UNTRAINED_MODES + ('rf', 'gbm', 'mlp', 'deep', 'ensemble')
    by_game = defaultdict(list)
    for d in draws:
        by_game[d['code']].append(d)
    results = {}
    for m in modes:
        if progress:
            progress(m)
        if m in UNTRAINED_MODES:
            results[m] = _backtest_untrained(by_game, m, min_hist,
                                             all_draws=draws if m in ('pattern', 'charts2') else None)
        else:
            results[m] = _backtest_per_game_trained(by_game, m, min_hist, retrain_every, quick)
    return {'meta': {'n_draws': len(draws), 'min_hist': min_hist, 'modes': list(modes),
                      'games': GAMES, 'retrain_every': retrain_every, 'quick': quick},
            'results': results}


def print_summary(report):
    print(f"Walk-forward backtest: {report['meta']['n_draws']} draws, "
          f"predictions start after {report['meta']['min_hist']} draws/game")
    print(f"{'strategy':<10}{'picks':<6}{'>=1 hit':<11}{'95% CI':<19}{'>=2 hits':<10}"
          f"{'95% CI':<19}{'random>=1':<11}{'random>=2':<11}{'AUC':<7}{'Brier':<8}{'n tests'}")
    for mode, res in report['results'].items():
        pooled = res['pooled']
        for k in (2, 3, 4, 5):
            hr = pooled['hitrate'][str(k)]
            ci1 = f"[{hr['ge1_ci'][0]:.1%},{hr['ge1_ci'][1]:.1%}]"
            ci2 = f"[{hr['ge2_ci'][0]:.1%},{hr['ge2_ci'][1]:.1%}]"
            auc_s = f"{pooled['auc']:.3f}" if pooled['auc'] is not None else 'n/a'
            brier_s = f"{pooled['brier']:.4f}" if pooled['brier'] is not None else 'n/a'
            print(f"{mode:<10}{k:<6}{hr['ge1_rate']:<11.3%}{ci1:<19}{hr['ge2_rate']:<10.3%}{ci2:<19}"
                  f"{hr['random_ge1']:<11.3%}{hr['random_ge2']:<11.3%}{auc_s:<7}{brier_s:<8}{pooled['n_tests']}")
        print()
