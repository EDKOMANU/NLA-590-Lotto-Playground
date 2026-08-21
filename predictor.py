"""
Ghana NLA 5/90 Prediction System v3
Data: ghana_lotto_history.csv (Dec 2017 -> present, 7 classic games)

Commands
--------
  python predictor.py update                # fetch latest results from ghanayello.com and append
  python predictor.py predict                # picks for the next draw of every game
  python predictor.py predict MS             # one game (MS LT MW FT FB NW SA)
  python predictor.py predict --date 2026-07-20        # picks for the draw on that date
  python predictor.py predict --week 2026-07-20        # picks for every game that week
  python predictor.py backtest               # honest walk-forward evaluation of all strategies

Strategies: hot | recent | overdue | blend | charts | ml (legacy numpy logistic regression)
            pattern (live pattern-analysis system, no training needed)
            rf | gbm | mlp (scikit-learn, one model per game) | deep (pytorch LSTM, per game) | ensemble

Honest note: 5/90 draws are random; the backtest measures every strategy
against pure chance so you can see exactly what an edge would look like.
No strategy here -- including the ML/deep-learning ones -- is expected to
or has been shown to beat random chance. Run `train.py` first to produce
the trained artifacts the rf/gbm/mlp/deep/ensemble modes load (one set per
game -- a data update for one game only ever affects that game's own
artifacts, never another game's).
"""
import sys
import datetime as dt

from lottery_core import config, data, classic, ensemble as ens_mod, backtest as bt, artifacts

GAMES = config.GAMES
NAMES = config.NAMES
DOW = config.DOW


def get_scores_any(history, mode, game_code, all_draws=None):
    """Dispatch across two systems: the untrained modes (legacy stat/chart heuristics
    plus the live 'pattern' system -- pure functions of history, always fresh, no
    train.py needed) and the trained modes that load an artifact from disk (produced by
    train.py, and stale after `update` until it's re-run). `all_draws` (every game's
    draws, same date cutoff as `history`) feeds the pattern system's cross-game
    component; without it that component is simply absent."""
    if mode in config.LEGACY_MODES:
        return classic.get_scores(history, mode)
    if mode == 'pattern':
        from lottery_core import pattern_analysis
        scores, _ = pattern_analysis.pattern_scores(history, all_draws=all_draws)
        return scores
    if mode == 'spatial':
        from lottery_core import spatial_engine
        scores, _ = spatial_engine.spatial_scores(history)
        return scores
    if mode == 'charts2':
        from lottery_core import chart_analysis
        return chart_analysis.chart2_scores(history, all_draws=all_draws)
    if mode == 'plans':
        from lottery_core import plan_engine
        scores, _ = plan_engine.plan_scores(history)
        return scores
    from lottery_core import sklearn_models, deep_model
    if mode in ('rf', 'gbm', 'mlp'):
        model = artifacts.load_sklearn_model(f'{mode}_{game_code}')
        if model is None:
            print(f"  [{mode}_{game_code}] no trained artifact found -- run `python train.py` first.")
            return {k: 0.0 for k in range(1, 91)}
        return sklearn_models.sklearn_scores(mode, model, history, game_code)
    if mode == 'deep':
        state_dict, meta = artifacts.load_deep_weights(f'deep_{game_code}')
        if state_dict is None:
            print(f"  [deep_{game_code}] no trained artifact found -- run `python train.py` first.")
            return {k: 0.0 for k in range(1, 91)}
        return deep_model.deep_scores(state_dict, meta, history, game_code)
    if mode == 'ensemble':
        comp = {'blend': classic.get_scores(history, 'blend'), 'charts': classic.get_scores(history, 'charts')}
        for name in ('rf', 'gbm', 'mlp'):
            model = artifacts.load_sklearn_model(f'{name}_{game_code}')
            if model is not None:
                comp[name] = sklearn_models.sklearn_scores(name, model, history, game_code)
        state_dict, meta = artifacts.load_deep_weights(f'deep_{game_code}')
        if state_dict is not None:
            comp['deep'] = deep_model.deep_scores(state_dict, meta, history, game_code)
        return ens_mod.blend_scores(comp)
    raise ValueError(f"unknown mode: {mode}")


def predict_game(draws, game, upto=None, mode='blend', quiet=False):
    seq = [d for d in draws if d['code'] == game and (upto is None or d['date'] < upto)]
    if len(seq) < 30:
        print(f"{NAMES[game]}: not enough history before {upto}"); return None
    all_d = [d for d in draws if upto is None or d['date'] < upto]
    sc = get_scores_any(seq, mode, game, all_draws=all_d)
    ranked = sorted(sc.items(), key=lambda t: (-t[1], t[0]))
    p = [n for n, _ in ranked[:10]]
    if not quiet:
        tag = f" (as of {upto}, data through {seq[-1]['date']})" if upto else f" (data through {seq[-1]['date']})"
        print(f"{NAMES[game]}{tag}  [strategy: {mode}]")
        print(f"  2-sure: {p[:2]}   3-direct: {p[:3]}   5 picks: {p[:5]}   alternates: {p[5:10]}")
    return p[:5]


def next_date_for(game, start):
    target = config.GAME_DOW[game]
    d = start
    while d.weekday() != target: d += dt.timedelta(days=1)
    return d


def cli():
    draws = data.load()
    args = sys.argv[1:]
    cmd = args[0] if args else 'predict'
    if cmd == 'update':
        data.update(); return
    if cmd == 'backtest':
        modes = tuple(args[1:]) or None
        report = bt.backtest(draws, modes=modes, progress=lambda m: print(f"-- running {m} --"))
        bt.print_summary(report)
        return
    if cmd == 'keys':
        # Key Identification Engine + Monte Carlo bootstrap noise control (spatial
        # engine, Component 2): screened positional keys with per-key AND family-wise
        # empirical p-values, per game.
        from lottery_core import spatial_engine as se
        games = [a.upper() for a in args[1:] if a.upper() in GAMES] or GAMES
        for g in games:
            seq = [d for d in draws if d['code'] == g]
            report = se.key_search(seq)
            print(f"\n{NAMES[g]}: {len(report['keys'])} screened key(s) out of {report['tested']} tested "
                  f"(~{report['expected_false_positives']:.0f} false positives EXPECTED from noise at this screen)")
            for k in report['keys'][:10]:
                bp = se.bootstrap_key_pvalue(seq, k, iterations=500)
                print(f"  {k['src']} --{k['op']}--> {k['target']}: {k['hits']}/{k['trials']} "
                      f"({k['rate']:.2%}, chance 1.11%), screen p={k['p_value']:.4f}, bootstrap p={bp:.3f}")
            fw = se.bootstrap_best_key_pvalue(seq, iterations=100)
            print(f"  FAMILY-WISE bootstrap: best real key p={fw['real_best_p']:.5f}, "
                  f"P(noise produces a better best-key)={fw['bootstrap_p']:.2f} "
                  f"-> {'DISCARD ALL (indistinguishable from noise)' if fw['bootstrap_p'] > 0.05 else 'best key beats the noise baseline'}")
        return
    mode = 'blend'; date = None; week = False; game = None
    rest = args[1:] if cmd == 'predict' else args
    i = 0
    while i < len(rest):
        a = rest[i]
        if a == '--date': date = dt.date.fromisoformat(rest[i + 1]); i += 2
        elif a == '--week': date = dt.date.fromisoformat(rest[i + 1]); week = True; i += 2
        elif a == '--mode': mode = rest[i + 1]; i += 2
        elif a.upper() in GAMES: game = a.upper(); i += 1
        else: i += 1
    if week:
        monday = date - dt.timedelta(days=date.weekday())
        print(f"Predictions for the week of {monday} ({mode} strategy):\n")
        for g in GAMES:
            gd = next_date_for(g, monday)
            print(f"-- {gd} --")
            predict_game(draws, g, upto=gd, mode=mode)
        return
    if date:
        g = game or DOW[date.weekday()]
        print(f"Prediction for {date}:")
        predict_game(draws, g, upto=date, mode=mode)
        return
    today = dt.date.today()
    for g in ([game] if game else GAMES):
        gd = next_date_for(g, today)
        print(f"-- next {NAMES[g]}: {gd} --")
        predict_game(draws, g, mode=mode)
        print()


if __name__ == '__main__':
    cli()
