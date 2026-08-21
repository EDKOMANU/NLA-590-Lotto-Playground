"""NumPy-only inference for the per-game LSTM in deep_model.py.

The trained network is tiny and fixed-shape (Linear 273->32, one 64-unit LSTM layer,
Linear 64->90), so scoring it needs nothing beyond the weights and a dozen lines of
matrix algebra. Doing that here instead of through torch is what lets the *deployed*
app run without PyTorch installed at all: torch (and, on Linux wheels, the ~2.5 GB of
bundled CUDA libraries it drags in) is a training-only dependency, see
requirements-train.txt. train.py still uses torch; the app never imports it.

Weights are read from artifacts/deep_<GAME>.npz, written alongside the .pt checkpoint
by artifacts.save_torch_model(). Outputs match torch's forward pass to float32
precision -- tools/verify_deep_parity.py is the check.
"""
import numpy as np


def _sigmoid(x):
    # expit without scipy; the two-branch form avoids overflow warnings on large |x|.
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def forward(weights, seq):
    """seq: (T, INPUT_DIM) float32 for one game. Returns 90 sigmoid probabilities.

    Mirrors _Net.forward in deep_model.py: input projection, LSTM over the window,
    head on the final hidden state. Dropout is identity at eval time.
    """
    w_in, b_in = weights['input_proj.weight'], weights['input_proj.bias']
    w_ih, w_hh = weights['lstm.weight_ih_l0'], weights['lstm.weight_hh_l0']
    b_ih, b_hh = weights['lstm.bias_ih_l0'], weights['lstm.bias_hh_l0']
    w_head, b_head = weights['head.weight'], weights['head.bias']

    proj = seq.astype(np.float32) @ w_in.T + b_in           # (T, 32)
    hidden = w_hh.shape[1]
    h = np.zeros(hidden, dtype=np.float32)
    c = np.zeros(hidden, dtype=np.float32)
    # torch stacks the four gates in the order input, forget, cell, output.
    gates_x = proj @ w_ih.T + b_ih                          # (T, 4H), precomputed
    for t in range(proj.shape[0]):
        g = gates_x[t] + w_hh @ h + b_hh
        i = _sigmoid(g[0:hidden])
        f = _sigmoid(g[hidden:2 * hidden])
        cand = np.tanh(g[2 * hidden:3 * hidden])
        o = _sigmoid(g[3 * hidden:4 * hidden])
        c = f * c + i * cand
        h = o * np.tanh(c)
    logits = w_head @ h + b_head
    return _sigmoid(logits)
