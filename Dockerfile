# syntax=docker/dockerfile:1
#
# Serving image for the Ghana NLA 5/90 Predictor. Deliberately CPU/NumPy only: the
# deep model is scored from artifacts/deep_*.npz by lottery_core/deep_runtime.py, so
# PyTorch (a ~2.5 GB CUDA stack in its Linux PyPI build) never enters the image.
# The whole dependency set is ~160 MB of wheels, which is what keeps a cold start
# measured in seconds. To *train*, use the training image below, not this one.
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501 \
    # No source files change at runtime, so the file watcher is pure overhead, and
    # usage stats add a network call to every start.
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# curl is only here for the healthcheck. No build-essential: every runtime dependency
# ships a manylinux wheel, and pulling a compiler added minutes to the image build.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies in their own layer so a code change doesn't reinstall them.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit writes its disk cache under HOME; give the non-root user somewhere to put
# it. Mount a volume at /app/.cache to carry that cache across restarts -- a woken
# container then serves its first request without recomputing the chart analyses.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/.cache \
    && chown -R appuser:appuser /app
USER appuser
ENV HOME=/app/.cache

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]


# ---------------------------------------------------------------------------
# Optional training image: the runtime plus PyTorch, for `python train.py`.
# Build with:  docker build --target training -t ghana-lotto-trainer .
FROM runtime AS training
USER root
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
USER appuser
CMD ["python", "train.py"]
