"""A small per-game LSTM sequence model over draw history.

One independent model PER GAME (not pooled): trained only on that game's own ~400-draw
history, so a data update for one game can only ever change predictions for that game.
Previously pooled across all 7 games with a learned game embedding for a larger
effective training set, but that meant retraining after any single game's update
silently changed every other game's predictions too -- the wrong tradeoff here (see
sklearn_models.py's docstring for the full reasoning). Dropping the game embedding also
simplifies the model appropriately for the smaller per-game sequence count (~350-400
windows instead of ~2,700 pooled).

Deliberately small: a single-layer LSTM (64 hidden units) reading a 20-draw sliding
window. Regularized with dropout + weight decay + early stopping on a time-based
holdout. Expected outcome (consistent with REPORT.md's existing findings) is
~chance-level AUC (~0.5) -- that is the correct result for independent uniform-random
draws, not a shortcoming to fix by adding capacity.
"""
import math

import numpy as np

from .charts import CHARTS

# 90 win multi-hot + 90 machine-number multi-hot + 90 chart-pointer channel + 3 scalars.
# The machine-number channel is a plain 0/1 indicator (not hand-weighted) so the
# network's own input projection learns how much to trust it, rather than baking in an
# assumed 0.5 prior. The chart-pointer channel gives the LSTM the same relationship
# signal the sklearn models get via chart_in_w/chart_in_mach_w (section 2 of
# features.py) -- raw pointer counts per candidate number (win pointers weighted 1.0,
# machine pointers 0.5, matching classic.chart_scores' own win/machine weighting).
# Whether either channel matters is left for training to discover, not assumed.
INPUT_DIM = 273
WINDOW = 20
HIDDEN = 64


def _chart_pointer_vec(d):
    v = np.zeros(90, dtype=np.float32)
    for mp in CHARTS.values():
        for x in d['win']:
            if x in mp:
                v[mp[x] - 1] += 1.0
        for x in d['mach']:
            if x in mp:
                v[mp[x] - 1] += 0.5
    return v


def _draw_repr(d):
    v = np.zeros(INPUT_DIM, dtype=np.float32)
    for x in d['win']:
        v[x - 1] = 1.0
    for x in d['mach']:
        v[90 + x - 1] = 1.0
    v[180:270] = _chart_pointer_vec(d)
    v[270] = sum(d['win']) / 225.0
    month = d['date'].month
    v[271] = math.sin(2 * math.pi * month / 12.0)
    v[272] = math.cos(2 * math.pi * month / 12.0)
    return v


class LotterySeqModel:
    """Thin wrapper so the module works without importing torch at module load time
    (mirrors the lazy-import style already used for LogReg in features.py)."""

    def __init__(self, window=WINDOW, hidden=HIDDEN):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(INPUT_DIM, 32)
                self.lstm = nn.LSTM(input_size=32, hidden_size=hidden, num_layers=1, batch_first=True)
                self.dropout = nn.Dropout(0.3)
                self.head = nn.Linear(hidden, 90)

            def forward(self, x):
                proj = self.input_proj(x)
                out, _ = self.lstm(proj)
                return self.head(self.dropout(out[:, -1, :]))

        self.net = _Net()
        self.window = window

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)

    def eval(self):
        self.net.eval()
        return self


def build_sequences(history, window=WINDOW, min_hist=60):
    """Fixed-length (window, INPUT_DIM) sequences for a single game, each labeled with
    the next draw's multi-hot 90-vector, sorted chronologically (used for a time-based
    train/validation split, not random shuffling)."""
    n = len(history)
    start = max(window, min_hist)
    if n <= start:
        return None
    reps = np.stack([_draw_repr(d) for d in history])
    Xs, ys = [], []
    for i in range(start, n):
        Xs.append(reps[i - window:i])
        y = np.zeros(90, dtype=np.float32)
        y[[num - 1 for num in history[i]['win']]] = 1.0
        ys.append(y)
    if not Xs:
        return None
    return np.stack(Xs), np.stack(ys)


def torch_available():
    """Training needs torch; scoring does not (see deep_runtime). A deployment built
    from requirements.txt alone will not have it, and that is the intended setup --
    callers use this to skip the deep model instead of crashing."""
    import importlib.util
    return importlib.util.find_spec('torch') is not None


def train_deep(history, window=WINDOW, min_hist=60, max_epochs=40, quick=False, patience=5):
    if not torch_available():
        raise ImportError(
            "Training the deep model needs PyTorch, which the runtime requirements "
            "deliberately omit. Install it with: pip install -r requirements-train.txt"
        )
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    built = build_sequences(history, window=window, min_hist=min_hist)
    if built is None:
        return None, None
    X, y = built
    n = len(X)
    if n < 30:
        return None, None
    val_frac = 0.1
    split = max(1, int(n * (1 - val_frac)))
    # Consecutive sliding windows share input frames at the boundary (window i and i+1
    # overlap by construction); drop `window` samples between train and val so no
    # validation sample's input frames overlap any training sample's, even though the
    # prediction *targets* never overlapped either way (val always predicts strictly
    # later draws than any training example).
    gap = window
    Xtr, Xval = X[:split], X[split + gap:]
    ytr, yval = y[:split], y[split + gap:]
    if len(Xval) == 0:
        Xval, yval = Xtr[-1:], ytr[-1:]

    # Labels are ~5/90 positive per row; pos_weight rebalances the loss so the network
    # isn't dominated by the majority-negative gradient.
    pos_weight = torch.full((90,), (90 - 5) / 5)
    model = LotterySeqModel(window=window)
    opt = torch.optim.AdamW(model.net.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    Xval_t = torch.tensor(Xval); yval_t = torch.tensor(yval)

    epochs = min(max_epochs, 10) if quick else max_epochs
    best_val = float('inf'); best_state = None; bad_epochs = 0
    for epoch in range(epochs):
        model.net.train()
        for xb, yb in loader:
            opt.zero_grad()
            logits = model.net(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        model.net.eval()
        with torch.no_grad():
            val_logits = model.net(Xval_t)
            val_loss = loss_fn(val_logits, yval_t).item()
        if val_loss < best_val - 1e-4:
            best_val = val_loss; best_state = {k: v.clone() for k, v in model.net.state_dict().items()}; bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.net.load_state_dict(best_state)
    meta = {'window': window, 'hidden': HIDDEN, 'best_val_bce': best_val, 'epochs_run': epoch + 1}
    return model, meta


def deep_scores(state_dict, meta, history, game_code=None):
    """Score the next draw from a trained checkpoint, in NumPy.

    Runs deep_runtime.forward rather than rebuilding the torch module, so the
    deployed app never imports torch (see deep_runtime's docstring). Accepts either a
    live torch state_dict (what backtest.py passes, straight off a just-trained model)
    or the NumPy weights artifacts.load_deep_weights returns.
    """
    from . import deep_runtime
    if state_dict is None or not history:
        return {k: 0.0 for k in range(1, 91)}
    weights = {k: (v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v))
               for k, v in state_dict.items()}
    window = meta.get('window', WINDOW)
    recent = history[-window:]
    reps = np.stack([_draw_repr(d) for d in recent])
    probs = deep_runtime.forward(weights, reps)
    return {k + 1: float(p) for k, p in enumerate(probs)}
