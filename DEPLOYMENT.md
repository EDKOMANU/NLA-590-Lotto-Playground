# Deployment Guide for Ghana NLA 5/90 Predictor

This repository contains the full Ghana NLA 5/90 Predictor application built with Streamlit, Plotly, PyTorch, and Scikit-learn.

This guide provides step-by-step instructions for deploying the application using **Streamlit Community Cloud**, **Docker**, or **Docker Compose**.

---

## 1. Deploying to Streamlit Community Cloud

Streamlit Community Cloud is a free and easy platform for hosting Streamlit apps directly from GitHub.

### Prerequisites
- A GitHub account.
- This repository pushed to your GitHub account (public or private).

### Steps
1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New app** in the upper right corner.
3. Select your repository, branch (e.g., `main`), and specify `app.py` as the **Main file path**.
4. (Optional) Advanced settings:
   - Python Version: `3.10` or higher.
5. Click **Deploy!**

 Streamlit Cloud will automatically install dependencies from `requirements.txt` and load settings from `.streamlit/config.toml`. The application, including data (`ghana_lotto_history.csv`) and model artifacts (`artifacts/`, `backtest_cache/`), will be ready immediately.

---

## 2. Deploying with Docker

Docker containerization packages the application, dependencies, historical dataset, and pre-trained ML model artifacts into a portable container.

### Prerequisites
- Docker Engine installed (version 20.10+).

### Building and Running the Docker Image

1. **Build the Docker Image**:
   ```bash
   docker build -t ghana-lotto-predictor:latest .
   ```

2. **Run the Container**:
   ```bash
   docker run -d \
     --name ghana-lotto-app \
     -p 8501:8501 \
     -v $(pwd)/ghana_lotto_history.csv:/app/ghana_lotto_history.csv \
     -v $(pwd)/artifacts:/app/artifacts \
     -v $(pwd)/backtest_cache:/app/backtest_cache \
     ghana-lotto-predictor:latest
   ```

3. **Access the Application**:
   Open your browser and go to `http://localhost:8501` (or `http://<your-server-ip>:8501`).

---

## 3. Deploying with Docker Compose

Docker Compose simplifies container management with pre-configured settings for environment variables, ports, healthchecks, and persistent volume mounts.

### Prerequisites
- Docker and Docker Compose (or `docker compose` CLI plugin).

### Steps

1. **Start the Application**:
   ```bash
   docker compose up -d
   ```

2. **View Logs**:
   ```bash
   docker compose logs -f
   ```

3. **Stop the Application**:
   ```bash
   docker compose down
   ```

---

## 4. Persistent Storage & Retraining Notes

- **Data Updates**: The application allows users to fetch the latest draw results from the sidebar ("Fetch latest draws"). Mounting `ghana_lotto_history.csv` as a volume ensures new draw data persists across container restarts.
- **Model Retraining**: Triggering model retraining ("Retrain selected games") writes updated model files into `artifacts/` and backtest evaluations into `backtest_cache/`. Volume mounts keep these artifacts persistent across container updates.
- **Resource Recommendations**: If performing full background retraining inside a container, allocate at least 2 CPU cores and 2 GB RAM.
