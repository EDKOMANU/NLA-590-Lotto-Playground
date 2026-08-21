import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Overridable so a container can point the mutable state (the draw archive that
# "Fetch latest draws" appends to, and the model/backtest artifacts a retrain writes)
# at a mounted volume instead of the image's own filesystem, which is thrown away on
# every restart. Unset, everything stays where it has always been, next to the code.
CSVF = os.environ.get('LOTTO_DATA_CSV') or os.path.join(BASE, 'ghana_lotto_history.csv')
ARTIFACT_DIR = os.environ.get('LOTTO_ARTIFACT_DIR') or os.path.join(BASE, 'artifacts')
CACHE_DIR = os.environ.get('LOTTO_CACHE_DIR') or os.path.join(BASE, 'backtest_cache')

GAMES = ['MS', 'LT', 'MW', 'FT', 'FB', 'NW', 'SA']
NAMES = {'MS': 'Monday Special', 'LT': 'Lucky Tuesday', 'MW': 'MidWeek', 'FT': 'Fortune Thursday',
         'FB': 'Friday Bonanza', 'NW': 'National Weekly', 'SA': 'Sunday Aseda'}
DOW = {0: 'MS', 1: 'LT', 2: 'MW', 3: 'FT', 4: 'FB', 5: 'NW', 6: 'SA'}
GAME_DOW = {'MS': 0, 'LT': 1, 'MW': 2, 'FT': 3, 'FB': 4, 'NW': 5, 'SA': 6}
GAME_INDEX = {g: i for i, g in enumerate(GAMES)}
EXP_GAP = 18.0  # expected draws between hits of a number: 90/5

LEGACY_MODES = ('hot', 'recent', 'overdue', 'blend', 'charts', 'ml')
# LIVE_MODES: pure functions of history, recomputed fresh on every call -- no trained
# artifact, no weekly retrain step. 'pattern' is the composite reading-strategy system
# (chart relationships + machine-number affinity + pair-tracing + addition-derivation
# + trend/conditional/cross-game/machine-trace). 'spatial' is the blueprint Spatial
# Pattern Matching Engine (NCC template matching over the binary chart image,
# diagonal trajectories, box/V-shape enclosures, screened positional keys, lagged
# machine->win crossover) -- see lottery_core/spatial_engine.py.
# 'charts2' is the upgraded chart strategy (per-entry measured rates, measured
# machine-source weighting, cross-game pooled trials -- see chart_analysis.py);
# the legacy 'charts' mode is kept unchanged as its backtest baseline.
# 'plans' is the Plan Discovery Engine (plan_engine.py): it finds situations resembling
# the current one, backtraces what dropped after them to where those numbers came from
# (up to ~10 weeks back), induces the recurring origin->drop mechanisms as PLANS,
# validates each plan on its own measured yield, applies them to the current window and
# reports how every plan produced its numbers.
LIVE_MODES = ('pattern', 'spatial', 'charts2', 'plans')
# NEW_MODES: need a trained artifact from train.py; stale after `predictor.py update`
# pulls in new draws until train.py is re-run.
NEW_MODES = ('rf', 'gbm', 'mlp', 'deep', 'ensemble')
UNTRAINED_MODES = LEGACY_MODES + LIVE_MODES  # scoreable via a pure history -> scores function
ALL_MODES = LEGACY_MODES + LIVE_MODES + NEW_MODES

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
