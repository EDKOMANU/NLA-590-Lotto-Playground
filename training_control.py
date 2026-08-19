"""Launches train.py as a background subprocess from the Streamlit app and tracks its
progress via lottery_core.artifacts' status/log files, so the app never blocks the UI
thread for the ~5-15 minutes a run takes. Not part of lottery_core: this is process
orchestration for the app, not prediction logic."""
import datetime as dt
import subprocess
import sys

from lottery_core import artifacts
from lottery_core.config import BASE

STALE_AFTER_MIN = 20  # if 'running' but no status update in this long, treat as dead

_popen = None  # module-level: persists across Streamlit reruns within one server process


def is_running():
    global _popen
    if _popen is not None and _popen.poll() is None:
        return True
    s = artifacts.load_training_status()
    if not s or s.get('status') != 'running':
        return False
    try:
        updated = dt.datetime.fromisoformat(s.get('updated_at', s.get('started_at')))
        return (dt.datetime.now() - updated).total_seconds() < STALE_AFTER_MIN * 60
    except (TypeError, ValueError):
        return True


def is_stale(s):
    if not s or s.get('status') != 'running':
        return False
    try:
        updated = dt.datetime.fromisoformat(s.get('updated_at', s.get('started_at')))
        return (dt.datetime.now() - updated).total_seconds() >= STALE_AFTER_MIN * 60
    except (TypeError, ValueError):
        return False


def start(quick=False, games=None, skip_backtest=False):
    global _popen
    if is_running():
        return False
    log_f = open(artifacts.TRAINING_LOG_PATH, 'w', encoding='utf-8')
    cmd = [sys.executable, 'train.py'] + (['--quick'] if quick else [])
    if games:
        cmd += ['--games'] + list(games)
    if skip_backtest:
        cmd += ['--skip-backtest']
    _popen = subprocess.Popen(cmd, cwd=BASE, stdout=log_f, stderr=subprocess.STDOUT)
    return True


def status():
    return artifacts.load_training_status()


def log_tail(n=40):
    return artifacts.tail_training_log(n)


def clear_stuck_status():
    artifacts.save_training_status({'status': 'idle'})
