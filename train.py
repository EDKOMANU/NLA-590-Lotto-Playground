"""Offline training/backtest pipeline. Not imported by predictor.py or app.py --
those only *load* the artifacts this script writes. Run this whenever the data
changes (after `predictor.py update`) to refresh predictions and the Model
Performance dashboard.

Models are trained independently PER GAME (see sklearn_models.py's docstring): a
data update for one game only ever changes that game's own artifacts, so
`--games MS LT` retrains just those two without touching the other five.

Usage
-----
  python train.py                 # full run: sklearn + deep models (all 7 games) + full backtest
  python train.py --quick         # fast dev run: fewer estimators/epochs, coarser backtest
  python train.py --games MS LT   # only retrain these games' sklearn/deep models
  python train.py --skip-sklearn  # skip fitting rf/gbm/mlp
  python train.py --skip-deep     # skip fitting the LSTM
  python train.py --skip-backtest # skip the (slowest) walk-forward evaluation
"""
import argparse
import datetime as dt
import time
from collections import defaultdict

from lottery_core import config, data, sklearn_models, deep_model, backtest as bt, artifacts


def build_by_game(draws):
    by_game = defaultdict(list)
    for d in draws:
        by_game[d['code']].append(d)
    return dict(by_game)


def train_and_save_sklearn(by_game, games, fp, quick):
    for g in games:
        history = by_game.get(g, [])
        game_fp = artifacts.game_data_fingerprint(g)
        for name in ('rf', 'gbm', 'mlp'):
            t0 = time.time()
            model = sklearn_models.train_sklearn(name, g, history, step=2, min_hist=60, quick=quick)
            artifact_name = f'{name}_{g}'
            if model is None:
                print(f"  [{artifact_name}] skipped -- not enough history yet")
                continue
            artifacts.save_sklearn_model(artifact_name, model, meta={
                'data_fingerprint': fp, 'game_fingerprint': game_fp, 'trained_at': dt.datetime.now().isoformat(),
                'quick': quick, 'game': g,
            })
            print(f"  [{artifact_name}] trained in {time.time() - t0:.1f}s -> artifacts/{artifact_name}.pkl")


def train_and_save_deep(by_game, games, fp, quick):
    for g in games:
        history = by_game.get(g, [])
        game_fp = artifacts.game_data_fingerprint(g)
        t0 = time.time()
        model, meta = deep_model.train_deep(history, min_hist=60, quick=quick)
        artifact_name = f'deep_{g}'
        if model is None:
            print(f"  [{artifact_name}] skipped -- not enough history yet")
            continue
        meta.update({'data_fingerprint': fp, 'game_fingerprint': game_fp, 'trained_at': dt.datetime.now().isoformat(),
                      'quick': quick, 'game': g})
        artifacts.save_torch_model(artifact_name, model.state_dict(), meta)
        print(f"  [{artifact_name}] trained in {time.time() - t0:.1f}s (val BCE {meta['best_val_bce']:.4f}, "
              f"{meta['epochs_run']} epochs) -> artifacts/{artifact_name}.pt")


def run_backtest_and_save(draws, fp, quick, skip_deep=False):
    t0 = time.time()
    retrain_every = 150 if quick else 75
    trained = ('rf', 'gbm', 'mlp', 'ensemble') if skip_deep else ('rf', 'gbm', 'mlp', 'deep', 'ensemble')
    modes = config.UNTRAINED_MODES + trained
    report = bt.backtest(draws, modes=modes, retrain_every=retrain_every, quick=quick,
                          progress=lambda m: print(f"  -- backtesting {m} --"))
    artifacts.save_backtest_cache(report, fp)
    print(f"  backtest finished in {time.time() - t0:.1f}s -> backtest_cache/")
    bt.print_summary(report)


def _status(stage, **extra):
    s = {'status': 'running', 'stage': stage, 'updated_at': dt.datetime.now().isoformat()}
    s.update(extra)
    prev = artifacts.load_training_status() or {}
    if 'started_at' in prev and prev.get('status') == 'running':
        s['started_at'] = prev['started_at']
    artifacts.save_training_status(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--games', nargs='+', choices=config.GAMES, default=config.GAMES,
                     help='only (re)train sklearn/deep models for these games (backtest still covers all games)')
    ap.add_argument('--skip-sklearn', action='store_true')
    ap.add_argument('--skip-deep', action='store_true')
    ap.add_argument('--skip-backtest', action='store_true')
    args = ap.parse_args()

    # Scoring the deep model is pure NumPy, but *fitting* it needs torch, which
    # requirements.txt leaves out so deployments stay small. Degrade to the sklearn
    # models rather than failing the run.
    skip_deep = args.skip_deep or not deep_model.torch_available()
    if skip_deep and not args.skip_deep:
        print("PyTorch not installed -- skipping the deep LSTM (rf/gbm/mlp still train). "
              "Install it with `pip install -r requirements-train.txt` to include it.")

    started_at = dt.datetime.now().isoformat()
    artifacts.save_training_status({'status': 'running', 'stage': 'starting',
                                     'started_at': started_at, 'quick': args.quick, 'games': args.games})
    try:
        draws = data.load()
        by_game = build_by_game(draws)
        fp = artifacts.data_fingerprint(modes=config.ALL_MODES)
        print(f"Loaded {len(draws)} draws across {len(by_game)} games (fingerprint {fp}, quick={args.quick}, "
              f"games={args.games})")

        if not args.skip_sklearn:
            print("Training scikit-learn ensemble (rf/gbm/mlp) per game...")
            _status('sklearn', quick=args.quick)
            train_and_save_sklearn(by_game, args.games, fp, args.quick)

        if not skip_deep:
            print("Training deep LSTM model per game...")
            _status('deep', quick=args.quick)
            train_and_save_deep(by_game, args.games, fp, args.quick)

        if not args.skip_backtest:
            print("Running full walk-forward backtest (this is the slow part)...")
            _status('backtest', quick=args.quick)
            run_backtest_and_save(draws, fp, args.quick, skip_deep=skip_deep)

        print("train.py done.")
        artifacts.save_training_status({'status': 'done', 'stage': 'done', 'started_at': started_at,
                                         'finished_at': dt.datetime.now().isoformat(), 'quick': args.quick,
                                         'games': args.games})
    except Exception as e:
        artifacts.save_training_status({'status': 'error', 'stage': 'error', 'started_at': started_at,
                                         'finished_at': dt.datetime.now().isoformat(), 'quick': args.quick,
                                         'games': args.games, 'error': repr(e)})
        raise


if __name__ == '__main__':
    main()
