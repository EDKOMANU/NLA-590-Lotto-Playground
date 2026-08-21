"""Check that the NumPy LSTM runtime reproduces torch's forward pass.

Run after touching lottery_core/deep_runtime.py or the model architecture. Needs torch
(pip install -r requirements-train.txt); the app itself does not.

    python tools/verify_deep_parity.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lottery_core import artifacts, config, data, deep_model, deep_runtime  # noqa: E402

TOL = 1e-5


def main():
    draws = data.load()
    worst = 0.0
    checked = 0
    for game in config.GAMES:
        weights, meta = artifacts.load_deep_weights(f'deep_{game}')
        if weights is None:
            print(f"{game}: no artifact, skipped")
            continue
        history = [d for d in draws if d['code'] == game]
        window = meta.get('window', deep_model.WINDOW)
        reps = np.stack([deep_model._draw_repr(d) for d in history[-window:]])

        model = deep_model.LotterySeqModel(window=window, hidden=meta.get('hidden', deep_model.HIDDEN))
        model.load_state_dict({k: torch.tensor(v) for k, v in weights.items()})
        model.eval()
        with torch.no_grad():
            ref = torch.sigmoid(model.net(torch.tensor(reps[None, :, :])))[0].numpy()

        got = deep_runtime.forward(weights, reps)
        diff = float(np.max(np.abs(ref - got)))
        worst = max(worst, diff)
        checked += 1
        rank_match = np.array_equal(np.argsort(-ref, kind='stable'), np.argsort(-got, kind='stable'))
        print(f"{game}: max|torch - numpy| = {diff:.3e}  identical ranking: {rank_match}")
        if not rank_match:
            print(f"  FAIL: {game} ranks numbers differently")
            return 1

    if not checked:
        print("no artifacts to check")
        return 1
    print(f"\nworst deviation across {checked} games: {worst:.3e} (tolerance {TOL:.0e})")
    if worst > TOL:
        print("FAIL: NumPy runtime diverges from torch")
        return 1
    print("PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
