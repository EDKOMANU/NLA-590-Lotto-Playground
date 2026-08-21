"""Persistence: trained sklearn/torch models and cached backtest results, so
predictor.py and app.py can *load* artifacts instead of retraining interactively.
train.py is the only place these are written for "production" use."""
import hashlib
import json
import os

from .config import ARTIFACT_DIR, CACHE_DIR, CSVF


def data_fingerprint(modes=()):
    """Hash of CSV mtime + size + row count + mode list, used to detect whether
    cached artifacts/backtest results are stale relative to the current data."""
    st = os.stat(CSVF)
    with open(CSVF, 'rb') as f:
        nrows = sum(1 for _ in f) - 1
    key = f"{st.st_mtime_ns}:{st.st_size}:{nrows}:{','.join(sorted(modes))}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def game_data_fingerprint(game_code):
    """Fingerprint of just one game's own draws (row count + last row's content) --
    unlike data_fingerprint (whole-CSV), this doesn't change when a data update only
    affects OTHER games, so it doesn't spuriously flag this game's artifacts as stale."""
    import csv
    with open(CSVF) as f:
        rows = [r for r in csv.DictReader(f) if r['code'] == game_code]
    if not rows:
        key = f"{game_code}:empty"
    else:
        last = rows[-1]
        key = f"{game_code}:{len(rows)}:{last['date']}:{','.join(last[f'w{i}'] for i in range(1, 6))}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def data_content_fingerprint():
    """Hash of the CSV's *contents*. Unlike data_fingerprint (which mixes in mtime), a
    fresh clone or a redeploy of unchanged data produces the same value -- which is what
    makes it safe to use as a cache key for results that survive a restart."""
    h = hashlib.sha256()
    with open(CSVF, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def data_latest_date():
    import csv
    with open(CSVF) as f:
        rows = list(csv.DictReader(f))
    return max(r['date'] for r in rows) if rows else None


# ---------------------------------------------------------------- sklearn models
def save_sklearn_model(name, model, meta=None):
    import joblib
    path = os.path.join(ARTIFACT_DIR, f'{name}.pkl')
    joblib.dump(model, path)
    if meta is not None:
        with open(os.path.join(ARTIFACT_DIR, f'{name}_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
    return path


def load_sklearn_model(name):
    import joblib
    path = os.path.join(ARTIFACT_DIR, f'{name}.pkl')
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def load_sklearn_meta(name):
    path = os.path.join(ARTIFACT_DIR, f'{name}_meta.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- deep model
# Written twice: the .pt checkpoint (torch's own format, for anyone resuming training)
# and a plain .npz of the same weights, which is what the *app* loads. The app runs
# the LSTM forward pass in NumPy (lottery_core/deep_runtime.py), so a deployment only
# needs torch if it is going to train -- see requirements-train.txt. That keeps the
# served image off the ~2.5 GB of CUDA wheels the Linux torch build pulls in.
def _to_numpy(state_dict):
    """Accepts a torch state_dict or an already-NumPy mapping and returns NumPy arrays."""
    import numpy as np
    out = {}
    for k, v in state_dict.items():
        if hasattr(v, 'detach'):
            v = v.detach().cpu().numpy()
        out[k] = np.asarray(v, dtype=np.float32)
    return out


def save_torch_model(name, state_dict, meta):
    import numpy as np
    import torch
    torch.save(state_dict, os.path.join(ARTIFACT_DIR, f'{name}.pt'))
    np.savez(os.path.join(ARTIFACT_DIR, f'{name}.npz'), **_to_numpy(state_dict))
    with open(os.path.join(ARTIFACT_DIR, f'{name}_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def load_deep_weights(name):
    """(weights_as_numpy, meta) for a trained per-game LSTM, or (None, None).

    Prefers the .npz so no torch import is needed. Falls back to the .pt checkpoint
    only when the .npz is missing (an artifact directory trained before the NumPy
    runtime landed) -- that path does need torch; run tools/convert_deep_artifacts.py
    once to regenerate the .npz files and the fallback stops being used.
    """
    import numpy as np
    meta_path = os.path.join(ARTIFACT_DIR, f'{name}_meta.json')
    if not os.path.exists(meta_path):
        return None, None
    with open(meta_path) as f:
        meta = json.load(f)
    npz_path = os.path.join(ARTIFACT_DIR, f'{name}.npz')
    if os.path.exists(npz_path):
        with np.load(npz_path) as z:
            return {k: z[k] for k in z.files}, meta
    pt_path = os.path.join(ARTIFACT_DIR, f'{name}.pt')
    if not os.path.exists(pt_path):
        return None, None
    import torch
    return _to_numpy(torch.load(pt_path, map_location='cpu', weights_only=True)), meta


def load_torch_model(name):
    """Back-compat alias -- returns NumPy weights, which deep_model.deep_scores takes."""
    return load_deep_weights(name)


# ---------------------------------------------------------------- backtest cache
def save_backtest_cache(report, fingerprint):
    """`report` is the full {'meta': ..., 'results': {mode: {...}}} dict returned by
    backtest.backtest() -- saved as-is so load_latest_backtest_cache() round-trips it
    without re-wrapping."""
    path = os.path.join(CACHE_DIR, f'backtest_{fingerprint}.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    latest_path = os.path.join(CACHE_DIR, 'latest.json')
    with open(latest_path, 'w') as f:
        json.dump({'fingerprint': fingerprint, 'file': os.path.basename(path)}, f)
    return path


def load_latest_backtest_cache():
    """Returns the {'meta': ..., 'results': {mode: {...}}} report dict, or None."""
    latest_path = os.path.join(CACHE_DIR, 'latest.json')
    if not os.path.exists(latest_path):
        return None
    with open(latest_path) as f:
        pointer = json.load(f)
    path = os.path.join(CACHE_DIR, pointer['file'])
    if not os.path.exists(path):
        return None
    with open(path) as f:
        report = json.load(f)
    report.setdefault('meta', {})['fingerprint'] = pointer['fingerprint']
    return report


# ---------------------------------------------------------------- training status
# Lets app.py launch train.py as a background subprocess and poll its progress instead
# of blocking the Streamlit script thread for the ~5-15 minutes a full run takes.
TRAINING_STATUS_PATH = os.path.join(ARTIFACT_DIR, 'training_status.json')
TRAINING_LOG_PATH = os.path.join(ARTIFACT_DIR, 'training_log.txt')


def save_training_status(status):
    with open(TRAINING_STATUS_PATH, 'w') as f:
        json.dump(status, f, indent=2)


def load_training_status():
    if not os.path.exists(TRAINING_STATUS_PATH):
        return None
    with open(TRAINING_STATUS_PATH) as f:
        return json.load(f)


def tail_training_log(n=40):
    if not os.path.exists(TRAINING_LOG_PATH):
        return ""
    with open(TRAINING_LOG_PATH, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    return "".join(lines[-n:])


def game_artifact_meta(game_code, name='rf'):
    """Training metadata (data_fingerprint, trained_at, quick) for one game's artifact
    -- models are trained independently per game, so freshness is checked per game too."""
    return load_sklearn_meta(f'{name}_{game_code}')


def artifacts_trained_fingerprint(game_code=None, name='rf'):
    """The game_data_fingerprint recorded at training time for a given game's artifact
    (or the first available across all games if none given), used to warn when that
    game's own data has moved on since its models were last trained -- unaffected by
    updates to other games."""
    from .config import GAMES
    if game_code:
        meta = game_artifact_meta(game_code, name)
        return meta.get('game_fingerprint') if meta else None
    for g in GAMES:
        meta = game_artifact_meta(g, name)
        if meta:
            return meta.get('game_fingerprint')
    return None
