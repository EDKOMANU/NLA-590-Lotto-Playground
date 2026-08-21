# Deployment Guide — Ghana NLA 5/90 Predictor

A Streamlit app over Pandas, Plotly and scikit-learn, plus a small per-game LSTM that
is **scored in NumPy** and only **trained** with PyTorch.

That split is the whole reason this app deploys the way it does, so it is worth
stating once up front. `requirements.txt` — what a deployment installs — is ~160 MB of
wheels. Adding `torch` to it takes that to ~2.8 GB, because the Linux PyPI build of
PyTorch ships a full CUDA stack (cuBLAS, cuDNN, NCCL, Triton and a dozen more) that a
CPU-only 160 KB LSTM never touches. Every environment rebuild paid for all of it.

If you change one thing about how this app is hosted, keep torch out of the serving
environment.

---

## 1. Why the hosted app was slow to wake, and what changed

Streamlit Community Cloud sleeps an app after a period of inactivity and rebuilds its
Python environment on wake. The wake was therefore dominated by installing
dependencies, not by starting the app:

| | wheels to download | packages |
|---|---|---|
| before | **2,764 MB** | 61 |
| now | **161 MB** | 44 |

The removed bulk was `torch` (527 MB) plus nineteen `nvidia-*` / `triton` CUDA
packages (~2.1 GB). On top of the download, the unpacked install ran to roughly 6 GB —
comfortably over Community Cloud's resource limits, which is the other half of why the
app kept getting evicted and had to cold-start again.

Three changes got it there:

1. **The deep model no longer needs PyTorch to make predictions.**
   `lottery_core/deep_runtime.py` runs the LSTM forward pass in NumPy, reading weights
   from `artifacts/deep_*.npz`. `tools/verify_deep_parity.py` checks it against
   PyTorch's own output: the largest disagreement across all seven games is 9e-8, and
   the ranking of all 90 numbers is identical. **Predictions are unchanged.**
   Training still uses PyTorch — see §5.
2. **Requirements are pinned and split** into `requirements.txt` (serving),
   `requirements-train.txt` (adds torch) and `requirements-dev.txt`. Unpinned ranges
   let a resolver re-derive the world on every rebuild and let a major release land in
   production unreviewed.
3. **The expensive analyses are memoized, keyed on a hash of the draw archive.**
   Streamlit re-runs the entire script on every widget interaction and `st.tabs`
   renders *all* tabs, so each click used to re-pay for the pattern scoring, the
   per-pick explanations and the chart-relationship analyses whether or not you were
   looking at them. They now come from cache, and because the key is content-based the
   cache persists to disk — on a host whose disk survives a restart, a woken container
   answers its first request warm.

---

## 2. Choosing a platform

The wake cost is now a container start rather than a dependency install, which makes
the platform question mostly a question of whether the container is kept around.

| Platform | Sleeps? | Cold wake | Cost | Notes |
|---|---|---|---|---|
| **Fly.io** (§3) | only if you let it | ~1s suspended, a few seconds stopped | free allowance covers one small machine; ~$2–5/mo beyond it | Recommended. The image stays on the host, so nothing reinstalls. Volume keeps refreshed data and retrained models. |
| **Google Cloud Run** (§4) | yes, scale-to-zero | seconds (image cached) | pay-per-request, generous free tier | Good if you are already on GCP. `--min-instances=1` removes the cold start entirely. |
| **Streamlit Community Cloud** (§6) | yes, after inactivity | now ~10–20s instead of minutes | free | Still the least effort. The slimmed requirements are most of the fix; the rest is the sleep itself. |
| **Render / Railway** | free tier sleeps; paid does not | ~30–50s free, none paid | ~$5–7/mo paid | Deploys the same Dockerfile. Fine, just pricier than Fly for this size. |
| **Hugging Face Spaces** | after ~48h idle | seconds | free | Docker Spaces work; the persistent-storage story is weaker than Fly's volume. |

**Recommendation:** Fly.io with `min_machines_running = 1` if you want it always
instant, `0` if you would rather pay nothing and accept a ~1-second wake. Both are a
different category from where this started.

---

## 3. Fly.io (recommended)

`fly.toml` in the repo root is ready to go. It builds the `runtime` stage of the
Dockerfile, mounts a volume at `/data` for state that must outlive a restart, and
suspends rather than destroys an idle machine.

```bash
# once
curl -L https://fly.io/install.sh | sh
fly auth signup            # or: fly auth login

# claim a name and a region (jnb = Johannesburg, closest to Ghana)
fly launch --no-deploy --copy-config --name <your-app-name>
fly volumes create lotto_data --size 1 --region jnb

fly deploy

# seed the volume from the copies baked into the image (first deploy only)
fly ssh console -C "python /app/scripts/seed_volume.py"

fly open
```

Notes:

- **One machine, deliberately.** Streamlit keeps each session's state in the process
  that holds its websocket, so a second machine would strand half your users' state.
  `max_machines_running = 1`; scale the VM up rather than out.
- **Never wait for a wake:** set `min_machines_running = 1` in `fly.toml`. The machine
  then stays up and every visit is instant.
- **State on the volume.** `LOTTO_DATA_CSV`, `LOTTO_ARTIFACT_DIR` and
  `LOTTO_CACHE_DIR` point at `/data`, so "Fetch latest draws" and any retrain persist.
  `HOME=/data/.cache` puts Streamlit's persisted cache there too, which is what makes a
  woken machine's first render fast.
- Logs: `fly logs`. Shell: `fly ssh console`.

---

## 4. Google Cloud Run

```bash
gcloud run deploy ghana-lotto-predictor \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 8501 \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \        # 1 removes cold starts entirely
  --max-instances 1 \        # see the one-machine note in §3
  --session-affinity
```

`--session-affinity` matters: without it, Streamlit's websocket can land on a
different instance than the one holding the session.

Cloud Run's filesystem is ephemeral, so data refreshes and retrains do **not** persist
across a cold start unless you mount a GCS bucket with `--add-volume` and point
`LOTTO_DATA_CSV` / `LOTTO_ARTIFACT_DIR` / `LOTTO_CACHE_DIR` at it. Without that, the
app still works — it just always serves the archive and models baked into the image.

---

## 5. Docker, locally or on any VPS

```bash
docker compose up -d --build      # http://localhost:8501
docker compose logs -f
docker compose down
```

Or without compose:

```bash
docker build --target runtime -t ghana-lotto-predictor:latest .
docker run -d --name ghana-lotto-app -p 8501:8501 \
  -v lotto_data:/data \
  -e LOTTO_DATA_CSV=/data/ghana_lotto_history.csv \
  -e LOTTO_ARTIFACT_DIR=/data/artifacts \
  -e LOTTO_CACHE_DIR=/data/backtest_cache \
  -e HOME=/data/.cache \
  ghana-lotto-predictor:latest
docker exec ghana-lotto-app python /app/scripts/seed_volume.py
```

**Training** is a separate image, because it is the only thing that needs PyTorch:

```bash
docker build --target training -t ghana-lotto-trainer .
docker run --rm -v "$PWD/artifacts:/app/artifacts" \
  -v "$PWD/backtest_cache:/app/backtest_cache" ghana-lotto-trainer python train.py
```

Retraining in the *serving* container is possible but not recommended: a full run is
~5–15 minutes at 2 CPU / 2 GB while the app is trying to serve requests. If the
serving image has no torch, the sidebar's retrain button still works — it refits
`rf`/`gbm`/`mlp` and says so — and the deep model keeps scoring from its saved weights.

---

## 6. Streamlit Community Cloud

Unchanged from before, and still the least-effort option:

1. [share.streamlit.io](https://share.streamlit.io/) → **New app**.
2. Pick the repo and branch, main file `app.py`, Python 3.11.
3. **Deploy**.

It will still sleep when idle — that is the platform, not the app. What changed is
that waking it now installs 161 MB instead of 2.8 GB, and no longer risks blowing the
memory ceiling mid-install. If sleeping at all is the problem, move to §3 or §4.

---

## 7. Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # serving only, no torch
streamlit run app.py

# to retrain models as well:
pip install -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cpu
python train.py --quick
```

The `--extra-index-url` gets the CPU build of PyTorch (~200 MB) instead of the default
CUDA one (~2.5 GB). Skip it only if you actually have a GPU to train on.

After changing `lottery_core/deep_runtime.py` or the model architecture, re-run the
parity check:

```bash
python tools/verify_deep_parity.py
```

If you have `.pt` checkpoints without matching `.npz` files (an artifacts directory
from before this change), convert them once:

```bash
python tools/convert_deep_artifacts.py
```

`train.py` writes both formats from then on.
