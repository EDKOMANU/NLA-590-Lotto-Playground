"""scikit-learn ensemble: Random Forest, HistGradientBoosting, MLP.

One independent model PER GAME (not pooled): each game's model is trained only on that
game's own ~400-draw history, so a data update for one game (e.g. after
`predictor.py update`) can only ever change predictions for that game -- a game's own
picks never shift because a *different* game got new draws. This was previously pooled
across all 7 games for a larger effective training set, but that meant retraining after
any single game's update silently changed every other game's predictions too, which is
the wrong tradeoff: correctness/predictability per game matters more here than the
modest statistical benefit of a larger pooled fit (and the backtest already shows that
benefit doesn't translate into a real edge either way)."""
import warnings

from .features import build_extended_features, EXT_FEAT_NAMES

# RandomForestClassifier(n_jobs=-1) emits this benign joblib/sklearn-config-propagation
# UserWarning once per parallel batch; across many walk-forward retrains during a full
# backtest that adds up to tens of thousands of near-duplicate lines in any redirected
# log, so it's silenced here at the source rather than left for every caller to filter.
warnings.filterwarnings('ignore', message=r'.*sklearn\.utils\.parallel\.delayed.*', category=UserWarning)


def _build_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=30, n_jobs=-1, random_state=0)


def _build_gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=150, max_depth=4, learning_rate=0.05, l2_regularization=1.0, random_state=0)


def _build_mlp():
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, early_stopping=True, max_iter=300, random_state=0)


ESTIMATOR_BUILDERS = {'rf': _build_rf, 'gbm': _build_gbm, 'mlp': _build_mlp}


def build_game_training_set(game_code, history, step=2, min_hist=60, quick=False):
    """history: a single game's draws, already truncated to whatever cutoff date is
    desired (full history for a production fit, or 'before test point' for backtest).
    Returns (X, y) for that one game only."""
    import numpy as np
    if quick:
        step = max(step, 4)
    n = len(history)
    if n <= min_hist:
        return None, None
    Xs, ys = [], []
    for i in range(min_hist, n, step):
        F = build_extended_features(history[:i], history[i - 1], game_code, target_date=history[i]['date'])
        y = np.zeros(90); y[[num - 1 for num in history[i]['win']]] = 1
        Xs.append(F); ys.append(y)
    if not Xs:
        return None, None
    X = np.vstack(Xs); y = np.concatenate(ys)
    return X, y


def train_sklearn(name, game_code, history, step=2, min_hist=60, quick=False):
    X, y = build_game_training_set(game_code, history, step=step, min_hist=min_hist, quick=quick)
    if X is None:
        return None
    model = ESTIMATOR_BUILDERS[name]()
    if quick and name == 'rf':
        model.set_params(n_estimators=50)
    if quick and name == 'gbm':
        model.set_params(max_iter=50)
    if quick and name == 'mlp':
        model.set_params(max_iter=80)
    model.fit(X, y)
    return model


def sklearn_scores(name, model, history, game_code):
    """history = the target game's own history up to (not including) the draw being
    predicted. Returns {1..90: probability}."""
    if model is None or not history:
        return {k: 0.0 for k in range(1, 91)}
    F = build_extended_features(history, history[-1], game_code, target_date=None)
    probs = model.predict_proba(F)[:, 1]
    return {k + 1: float(p) for k, p in enumerate(probs)}
