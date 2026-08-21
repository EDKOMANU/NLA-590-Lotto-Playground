"""One-off: write a .npz next to every artifacts/deep_*.pt checkpoint.

The app scores the LSTM in NumPy (lottery_core/deep_runtime.py) and reads weights from
.npz, so an artifacts directory produced before that change needs converting once.
train.py writes both formats from then on. Needs torch (a training-side dependency):

    pip install -r requirements-train.txt
    python tools/convert_deep_artifacts.py
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lottery_core.config import ARTIFACT_DIR  # noqa: E402


def main():
    pts = sorted(glob.glob(os.path.join(ARTIFACT_DIR, 'deep_*.pt')))
    if not pts:
        print(f"no deep_*.pt checkpoints in {ARTIFACT_DIR}")
        return
    for pt in pts:
        sd = torch.load(pt, map_location='cpu', weights_only=True)
        out = pt[:-3] + '.npz'
        np.savez(out, **{k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()})
        print(f"{os.path.basename(pt)} -> {os.path.basename(out)} ({os.path.getsize(out):,} bytes)")


if __name__ == '__main__':
    main()
