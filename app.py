"""Ghana NLA 5/90 Predictor -- Streamlit app.

All prediction/scoring/backtest logic lives in lottery_core (and predictor.py's thin
dispatch on top of it); this file is presentation only. Run with:
    streamlit run app.py
"""
import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lottery_core import config, data, classic, ensemble as ens_mod, artifacts, pattern_analysis, features
from lottery_core import transform_engine as te_mod
from predictor import get_scores_any, next_date_for
import training_control

st.set_page_config(page_title="Ghana NLA 5/90 Predictor", layout="wide")

# Custom CSS for enhanced mobile responsiveness
st.markdown("""
<style>
@media (max-width: 768px) {
    /* Allow metric values and labels to wrap cleanly on mobile screens */
    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        word-break: break-word !important;
        white-space: normal !important;
        line-height: 1.3 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
    }
    /* Enable column wrapping on mobile narrow viewports */
    [data-testid="column"] {
        min-width: 130px !important;
        flex: 1 1 130px !important;
        margin-bottom: 0.5rem !important;
    }
    /* Ensure dataframes scroll horizontally on mobile without overflowing page */
    [data-testid="stDataFrame"] {
        width: 100% !important;
        overflow-x: auto !important;
    }
    /* Compact padding for small mobile displays */
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

HONESTY_LINE = (
    "5/90 lottery draws are independent, uniform-random events. No strategy in this "
    "app -- including the pattern analysis and the machine-learning/deep-learning "
    "models -- is expected to, or has been shown to, beat random chance. This app is "
    "for structured, transparent play, not prediction. See the Methodology tab for the "
    "numbers behind that claim."
)

PATTERN_MODES = ('hot', 'recent', 'overdue', 'blend', 'charts', 'charts2', 'pattern', 'spatial', 'plans')
ML_MODES = ('ml', 'rf', 'gbm', 'mlp', 'deep', 'ensemble')

# Plain-language names for the pattern components, used everywhere the 'Show your
# work' panel talks to the user -- the internal keys stay as-is in the code/score.
PLAIN_LABELS = {
    'recent': 'Recent form',
    'charts': 'Chart pointers',
    'mach_to_win': 'Machine-number record',
    'pattern_trace': 'Pair & triplet repeats',
    'transform': 'Number arithmetic',
    'terminal': 'Last-digit family',
    'group': 'Digital-root family',
    'trend': 'Similar past stretches',
    'conditional': 'Draws like this one',
    'cross_game': 'Other games this week',
    'mach_trace': 'Machine-pair repeats',
    'yearly': 'Same date, past years',
}


# ---------------------------------------------------------------- cached loaders
# Short TTLs so the app self-heals if a data update or a background retrain finishes
# without the explicit .clear() calls below being hit (e.g. a retrain started in an
# earlier session). load_draws changes rarely (only on `update`) so a longer TTL is
# fine; load_backtest_cache should reflect a finished retrain fairly promptly.
@st.cache_data(ttl=300)
def load_draws():
    return data.load()


@st.cache_data(ttl=30)
def load_backtest_cache():
    return artifacts.load_latest_backtest_cache()


@st.cache_data
def data_latest_date():
    return artifacts.data_latest_date()


def by_game_dict(draws):
    out = {}
    for d in draws:
        out.setdefault(d['code'], []).append(d)
    return out


# ---------------------------------------------------------------- helpers
def fig_bar_scores(scores, top_n=15, title=""):
    ranked = sorted(scores.items(), key=lambda t: (-t[1], t[0]))[:top_n]
    nums = [str(n) for n, _ in ranked]
    vals = [v for _, v in ranked]
    fig = go.Figure(go.Bar(x=nums, y=vals, marker_color="#4C78A8"))
    fig.update_layout(title=title, xaxis_title="number", yaxis_title="score",
                       margin=dict(l=10, r=10, t=40, b=10), height=350, autosize=True)
    return fig


def fig_hitrate_comparison(results, k, per_game=None):
    modes = list(results.keys())
    ge1, ge1_lo, ge1_hi, random_ge1 = [], [], [], []
    for m in modes:
        section = results[m]['per_game'].get(per_game, results[m]['pooled']) if per_game else results[m]['pooled']
        hr = section['hitrate'][str(k)]
        ge1.append(hr['ge1_rate'] * 100)
        ge1_lo.append((hr['ge1_rate'] - hr['ge1_ci'][0]) * 100)
        ge1_hi.append((hr['ge1_ci'][1] - hr['ge1_rate']) * 100)
        random_ge1.append(hr['random_ge1'] * 100)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='strategy', x=modes, y=ge1,
                          error_y=dict(type='data', symmetric=False, array=ge1_hi, arrayminus=ge1_lo),
                          marker_color="#4C78A8"))
    fig.add_trace(go.Scatter(name='random chance', x=modes, y=random_ge1, mode='lines+markers',
                              line=dict(color="#E45756", dash='dash')))
    fig.update_layout(title=f"P(>=1 hit) for {k} picks, with 95% CI", yaxis_title="%",
                       margin=dict(l=10, r=10, t=40, b=10), height=400, autosize=True)
    return fig


def fig_auc_brier(results, per_game=None):
    modes = list(results.keys())
    aucs, briers = [], []
    for m in modes:
        section = results[m]['per_game'].get(per_game, results[m]['pooled']) if per_game else results[m]['pooled']
        aucs.append(section['auc'] if section['auc'] is not None else None)
        briers.append(section['brier'] if section['brier'] is not None else None)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='ROC-AUC (chance = 0.5)', x=modes, y=aucs, marker_color="#72B7B2"))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#E45756")
    fig.update_layout(title="ROC-AUC by strategy", yaxis_title="AUC",
                       margin=dict(l=10, r=10, t=40, b=10), height=350, autosize=True)
    return fig


def methodology_text(cache):
    if not cache:
        return HONESTY_LINE + "\n\nNo backtest cache found yet -- run `python train.py` to generate one."
    results = cache['results']
    lines = [HONESTY_LINE, ""]
    lines.append(f"Backtest: {cache['meta']['n_draws']} draws, predictions start after "
                 f"{cache['meta']['min_hist']} draws/game, {list(results.values())[0]['pooled']['n_tests']} walk-forward test draws.")
    lines.append("")
    lines.append("5-pick P(>=1 hit) by strategy vs. random chance:")
    for m, res in results.items():
        hr = res['pooled']['hitrate']['5']
        lines.append(f"  - {m}: {hr['ge1_rate']:.1%} (95% CI {hr['ge1_ci'][0]:.1%}-{hr['ge1_ci'][1]:.1%}) "
                      f"vs random {hr['random_ge1']:.1%}")
    lines.append("")
    lines.append("Every strategy's confidence interval overlaps the random-chance rate -- "
                  "none has demonstrated a real edge.")
    return "\n".join(lines)


# ---------------------------------------------------------------- sidebar
draws = load_draws()
by_game = by_game_dict(draws)
bt_cache = load_backtest_cache()

st.sidebar.title("Ghana NLA 5/90")
st.sidebar.caption(HONESTY_LINE)
st.sidebar.divider()
st.sidebar.write(f"**Data through:** {draws[-1]['date']}")
st.sidebar.write(f"**Total draws:** {len(draws)}")
stale_games = [g for g in config.GAMES
               if artifacts.artifacts_trained_fingerprint(g) != artifacts.game_data_fingerprint(g)]
if not bt_cache and stale_games == config.GAMES:
    st.sidebar.warning("No trained artifacts found. Retrain below.")
elif stale_games:
    st.sidebar.warning(f"Stale artifacts for: {', '.join(stale_games)}. Retrain below "
                        f"(each game's model only depends on that game's own data).")
else:
    st.sidebar.success("Artifacts up to date for all games.")

st.sidebar.divider()
st.sidebar.subheader("1. Update data")
st.sidebar.caption("Pulls the latest results from the live results sites. The Pattern Analysis "
                    "tab reflects new draws immediately; the ML Models tab needs a retrain (step 2).")
if st.sidebar.button("Fetch latest draws"):
    try:
        with st.spinner("Fetching the latest results..."):
            result = data.update()
        load_draws.clear()
        n = result.get('added', 0) if isinstance(result, dict) else 0
        st.sidebar.success(f"Update complete — {n} new draw(s) added. Reloading." if n
                           else "Already up to date. Reloading.")
        st.rerun()
    except data.UpdateError as e:
        st.sidebar.error(str(e))
    except Exception as e:  # never let a refresh failure take down the whole app
        st.sidebar.error(f"Update failed unexpectedly: {e}")

st.sidebar.subheader("2. Retrain ML models")
st.sidebar.caption("Models are trained independently per game -- retraining one game "
                    "never changes another game's picks.")
retrain_games = st.sidebar.multiselect("Games to retrain", config.GAMES, default=stale_games or config.GAMES,
                                        format_func=lambda g: config.NAMES[g])
quick_retrain = st.sidebar.checkbox("Quick retrain (fewer estimators/epochs, coarser backtest)", value=True)
skip_bt_default = len(retrain_games) < len(config.GAMES)
skip_backtest_retrain = st.sidebar.checkbox("Skip backtest (faster -- just refresh model artifacts)",
                                             value=skip_bt_default)
st.sidebar.caption("A full run over all 7 games (with backtest) takes ~5-8 min quick / ~15 min full. "
                    "Skipping the backtest or retraining fewer games is much faster.")
train_status = training_control.status()
if training_control.is_running():
    st.sidebar.info(f"Training in progress (stage: {train_status.get('stage', '?') if train_status else '?'})...")
    if st.sidebar.button("Refresh status"):
        st.rerun()
    with st.sidebar.expander("Training log (tail)"):
        st.code(training_control.log_tail(30) or "(no output yet)")
else:
    if train_status and train_status.get('status') == 'done':
        st.sidebar.success(f"Last trained: {train_status.get('finished_at', '?')} "
                            f"({'quick' if train_status.get('quick') else 'full'}, "
                            f"games={','.join(train_status.get('games', []) or config.GAMES)})")
    elif train_status and train_status.get('status') == 'error':
        st.sidebar.error(f"Last run failed: {train_status.get('error', '?')}")
    elif training_control.is_stale(train_status):
        st.sidebar.warning("A previous training run looks stuck (no update in 20+ min).")
        if st.sidebar.button("Clear stuck status"):
            training_control.clear_stuck_status()
            st.rerun()
    if st.sidebar.button("Retrain selected games"):
        if not retrain_games:
            st.sidebar.error("Select at least one game.")
        elif training_control.start(quick=quick_retrain, games=retrain_games,
                                     skip_backtest=skip_backtest_retrain):
            st.sidebar.success("Retrain started in the background.")
            st.rerun()
        else:
            st.sidebar.error("A training run is already in progress.")

# ---------------------------------------------------------------- tabs
tab_pattern, tab_ml, tab_perf, tab_explore, tab_charts, tab_method = st.tabs(
    ["Pattern Analysis", "ML Models", "Model Performance", "Data Explorer", "Chart Relationships", "Methodology"])


def _game_date_picker(key_prefix):
    col1, col2, col3 = st.columns(3)
    game = col1.selectbox("Game", config.GAMES, format_func=lambda g: config.NAMES[g], key=f"{key_prefix}_game")
    use_date = col3.checkbox("Predict for a specific past date instead of 'next draw'", key=f"{key_prefix}_usedate")
    upto = None
    if use_date:
        upto = st.date_input("As of date (only draws before this date are used)",
                              value=draws[-1]['date'], min_value=draws[0]['date'], key=f"{key_prefix}_date")
        upto = dt.date.fromisoformat(str(upto))
    return col2, game, upto


def _picks_header(game, upto, sc, mode):
    ranked = sorted(sc.items(), key=lambda t: (-t[1], t[0]))
    picks10 = [n for n, _ in ranked[:10]]
    next_draw_date = upto if upto else next_date_for(game, dt.date.today())
    st.subheader(f"{config.NAMES[game]} -- draw of {next_draw_date}")
    c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.2])
    c1.metric("2-sure", " · ".join(map(str, picks10[:2])))
    c2.metric("3-direct", " · ".join(map(str, picks10[:3])))
    c3.metric("5 picks", " · ".join(map(str, picks10[:5])))
    c4.metric("Alternates", " · ".join(map(str, picks10[5:10])))
    st.plotly_chart(fig_bar_scores(sc, title=f"Top-15 scores ({mode})"), width='stretch')


with tab_pattern:
    st.header("Pattern Analysis")
    st.caption(
        "The reading-strategy system: chart relationships, machine-number affinity, "
        "pair-tracing (find when a current pair last repeated and see what followed), "
        "the 'addition' technique (plus one-up/one-down neighbor keys), 'terminal' "
        "(last-digit groups, e.g. the 7s: 7/17/27/.../87 -- crediting other members of "
        "the group, never a number for sharing a group with itself), 'group' (the same "
        "idea for digital-root groups: repeatedly sum a number's digits to one digit), "
        "**trend-similarity** (profile the last few draws as a trend and trace the most "
        "similar older trend-windows to see what won right after each), **conditional "
        "rates** (each number's next-draw rate measured separately under the current "
        "draw's own conditions: sum band, parity, span, repeat-from-previous), "
        "**cross-game transfer** (numbers drawn in the other games since this game's "
        "last draw, weighted by each source game's measured transfer rate -- the "
        "'today's results point at tomorrow's game' reading), **machine-trace** (the "
        "current machine pairs/triplets traced against historical machine draws), "
        "**yearly** (does the number recur within a week of this same calendar date "
        "across previous years -- finally meaningful with the archive reaching 1962), and a "
        "recency-weighted frequency baseline -- composed live from history on every "
        "request. 'Positional stickiness' and 'lap' (does a physical draw-slot repeat) "
        "are computed and shown in 'Show your work' too, but are diagnostic-only: "
        "they're excluded from the score itself, because they structurally only ever "
        "concern last week's own 5 numbers (see the Methodology tab for the measured "
        "reason why). Component weights among the scored components are not fixed: "
        "they're re-derived every call from which ones actually placed the most "
        "score-mass on real winners over the last 30 draws (see the Methodology tab "
        "for the honest caveats on doing that in a game that's provably random). "
        "**No training step**: run `predictor.py update` and this tab reflects the "
        "new draw immediately, unlike the ML Models tab."
    )
    col2, game, upto = _game_date_picker("pattern")
    mode = col2.selectbox("Strategy", list(PATTERN_MODES), index=list(PATTERN_MODES).index('pattern'), key="pattern_mode")
    seq = [d for d in draws if d['code'] == game and (upto is None or d['date'] < upto)]
    all_seq = [d for d in draws if upto is None or d['date'] < upto]
    if len(seq) < 30:
        st.warning(f"Not enough history for {config.NAMES[game]} before {upto}.")
    else:
        sc = get_scores_any(seq, mode, game, all_draws=all_seq)
        _picks_header(game, upto, sc, mode)
        st.caption("These picks are ranked by the selected strategy's score, not a probability of "
                   "winning. See the Methodology tab: no strategy has shown a real edge over random chance.")

        picks5 = [n for n, _ in sorted(sc.items(), key=lambda t: (-t[1], t[0]))[:5]]
        diag = pattern_analysis.combination_diagnostics(seq, picks5)
        st.caption(
            f"Combination diagnostic (descriptive only -- not part of any score): this 5-pick set sums "
            f"to {diag['sum']} (history mean {diag['history_mean_sum']:.0f}), has {diag['odd_count']}/5 "
            f"odd numbers (history mean {diag['history_mean_odd_count']:.1f}/5), and spans {diag['span']} "
            f"across {diag['decade_buckets_used']}/5 decade buckets (history mean span "
            f"{diag['history_mean_span']:.0f}) -- shown only so you can see whether the picks are a "
            f"structural outlier relative to {diag['n_historical_draws']} real draws, not because a "
            f"'typical-looking' combination is any more likely to win."
        )

        if mode == 'pattern':
            st.divider()
            st.subheader("Show your work")
            st.caption(
                "Each of the 5 predicted numbers gets its own tab below, showing the real historical "
                "evidence behind its score -- the actual past draws, dates and counts, never just an "
                "opaque number. Start with the **evidence map**: it shows which kind of evidence is "
                "carrying each pick."
            )
            with st.expander("How to read these tables (30 seconds)"):
                st.markdown(
                    "- **Measured rate** -- how often something actually happened in the recorded draws.\n"
                    "- **Chance** -- what pure luck would produce. An evidence line only means something "
                    "if its measured rate clearly beats its chance rate.\n"
                    "- **Cases / n** -- how many past examples were checked. A pattern seen 3 times is a "
                    "coincidence; the tables always tell you the count.\n"
                    "- **Score credit** -- how much the evidence actually adds to the number's score. "
                    "It is automatically reduced when there are few cases, so thin evidence can't "
                    "outshout well-tested evidence.\n"
                    "- **Share of score** -- what fraction of this pick's final score came from that "
                    "evidence type.\n\n"
                    "Honest bottom line: the draws are random, so every measured rate tends to sit at "
                    "its chance rate -- these tables let you *see* that for yourself, pick by pick."
                )

            def _fmt_positions(d):
                # pyarrow can't serialize dict columns with int keys for st.dataframe
                return ", ".join(f"#{k}@pos{v}" for k, v in d.items())

            def _render_explanation(num, exp=None, rank=None):
                if exp is None:
                    exp = pattern_analysis.explain(seq, num, all_draws=all_seq)
                contrib_sorted = sorted(exp['contribution'].items(),
                                        key=lambda kv: -kv[1]['contribution_pct'])
                drivers = [(PLAIN_LABELS.get(n, n), v['contribution_pct'])
                           for n, v in contrib_sorted[:2] if v['contribution_pct'] > 0]
                if rank:
                    st.markdown(f"### Pick #{rank} of 5 — number {num}")
                if drivers:
                    d_txt = " and ".join(f"**{n}**" for n, _ in drivers)
                    st.markdown(f"{num} is here mainly because of {d_txt} — together "
                                f"{sum(p for _, p in drivers):.0f}% of its score. "
                                f"The tables below show the actual past draws behind that.")
                with st.expander(f"The full story of {num}, in words"):
                    st.markdown(exp['narrative'])

                contrib_df = pd.DataFrame([
                    {'evidence': PLAIN_LABELS.get(name, name),
                     'share of score': f"{v['contribution_pct']:.0f}%",
                     'note': ("doesn't apply to this number" if v['excluded'] else "")}
                    for name, v in contrib_sorted
                ])
                st.write(f"**What's driving {num}'s score** (all evidence types, biggest first):")
                st.dataframe(contrib_df, width='stretch', hide_index=True)
                if any(v['excluded'] for v in exp['contribution'].values()):
                    st.caption(f"{num} was in the last draw itself, so 'family' evidence can't apply to "
                               f"it (a number can't support itself) — its score is shared out among the "
                               f"evidence that does apply.")
                fig_c = go.Figure(go.Bar(x=[PLAIN_LABELS.get(n, n) for n, _ in contrib_sorted],
                                          y=[v['contribution_pct'] for _, v in contrib_sorted],
                                          marker_color="#4C78A8"))
                fig_c.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                                     yaxis_title=f"% of {num}'s score")
                st.plotly_chart(fig_c, width='stretch')

                colA, colB = st.columns(2)
                with colA:
                    st.write(f"**Chart pointers → {num}** — last draw's numbers that 'point to' {num} "
                             f"on the traditional charts:")
                    if exp['chart_hits']:
                        st.dataframe(pd.DataFrame([
                            {'chart': p['chart'], 'pointing number': p['from'],
                             'from': 'winning' if p['from_kind'] == 'win' else 'machine',
                             'how often this chart comes true': f"{p['transfer_rate']:.1%}",
                             'note': 'points to itself — not counted' if p['self_pointer'] else ''}
                            for p in exp['chart_hits']]), width='stretch', hide_index=True)
                        st.caption("Pure chance would be 5.6% — compare each chart's real record to that.")
                    else:
                        st.caption(f"None of the last draw's numbers point to {num} on any chart.")
                    st.write(f"**{num} as a machine number** — did winning follow soon after? "
                             f"({exp['machine_number_confidence']}):")
                    if exp['machine_number_events']:
                        st.dataframe(pd.DataFrame([
                            {'was machine number on': e['date'],
                             'won soon after?': ('yes, ' + str(e['lag']) + ' draw(s) later on ' + str(e['hit_date'])) if e['hit'] else 'no'}
                            for e in exp['machine_number_events']]), width='stretch', hide_index=True)
                    else:
                        st.caption(f"{num} has never appeared as a machine number in this game.")
                    st.caption(
                        "The two lines below are shown for transparency but do **not** count toward the "
                        "score: they only ever describe last week's own 5 numbers, so scoring them would "
                        "just replay last week's draw (we measured exactly that happening — see the "
                        "Methodology tab)."
                    )
                    if exp['positional_trials']:
                        st.write(f"*Position note (not scored)*: numbers in the position {num} held last "
                                 f"draw come back within 5 draws {exp['positional_rate']:.0%} of the time "
                                 f"({exp['positional_trials']:,} cases; luck alone gives "
                                 f"{exp['positional_chance']:.0%}).")
                    if exp['lap_trials']:
                        st.write(f"*Lap note (not scored)*: the draw-slot {num} came out of last time "
                                 f"repeats into the very next draw {exp['lap_rate']:.0%} of the time "
                                 f"({exp['lap_trials']:,} cases).")
                with colB:
                    st.write(f"**This draw's pairs seen before — did {num} follow?** "
                             f"({exp['pattern_trace_total_repeats']} past repeat(s) of this draw's pairs/"
                             f"triplets found; {exp['pattern_trace_confidence']}):")
                    if exp['pattern_trace_hits']:
                        rows = [{'group that repeated': " & ".join(str(g) for g in h['groups']),
                                 'repeated on': h['repeated_on'],
                                 f'{num} followed': f"{h['lag']} draw(s) later, {h['followup_date']}"}
                                for h in exp['pattern_trace_hits']]
                        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
                        st.caption(f"When a pair or triplet from the current draw appeared together in "
                                   f"the past, these are the times {num} won shortly after.")
                    else:
                        st.caption(f"This draw's pairs have repeated before, but {num} never followed them.")
                    st.write(f"**Simple arithmetic that lands on {num}** "
                             f"({exp['transform_distinct_rules']} different trick(s) agree):")
                    if exp['transform_hits']:
                        st.dataframe(pd.DataFrame([
                            {'trick': h['rule'], 'using': str(h['group']),
                             'how often this trick works': f"{h['rate']:.1%}",
                             'past cases': f"{h['trials']:,}",
                             'score credit': f"{h['weight']:.1%}"}
                            for h in exp['transform_hits']]), width='stretch', hide_index=True)
                        st.caption("Tricks like adding a pair, doubling, mirroring a number across the "
                                   "board (23 becomes 68), one-up/one-down. "
                                   "Chance is ~5.6% — each trick's real record is shown next to it.")
                    else:
                        st.caption(f"No sum, double, mirror or neighbor move on the current draw "
                                   f"produces {num}.")
                    th, gh = exp['terminal_hit'], exp['group_hit']
                    if th:
                        if th['self_credit']:
                            st.caption(f"**Last-digit family**: {num} was in the last draw itself, so its "
                                       f"own family can't be used as evidence for it.")
                        else:
                            st.write(f"**Last-digit family**: {num} ends in **{th['class']}**, same as a "
                                     f"number in the last draw. Historically, when a '…{th['class']}' "
                                     f"number is drawn, another one follows within 5 draws "
                                     f"{th['rate']:.0%} of the time ({th['trials']:,} past cases).")
                    if gh:
                        if gh['self_credit']:
                            st.caption(f"**Digital-root family**: {num} was in the last draw itself, so "
                                       f"its own family can't be used as evidence for it.")
                        else:
                            st.write(f"**Digital-root family**: {num}'s digits reduce to **{gh['class']}** "
                                     f"(e.g. 67→6+7=13→4), same as a number in the last draw. Another "
                                     f"member of that family has followed within 5 draws "
                                     f"{gh['rate']:.0%} of the time ({gh['trials']:,} past cases).")
                    if not th and not gh:
                        st.caption("This number shares no terminal or digital-root group with the current draw.")

                st.divider()
                colC, colD = st.columns(2)
                with colC:
                    st.write(f"**Past stretches that looked like the current run** — the last few draws "
                             f"form a 'trend'; these are the most similar stretches in history where "
                             f"{num} won right after ({len(exp['trend_hits'])} of the top "
                             f"{exp['trend_top_matches']} matches):")
                    if exp['trend_hits']:
                        trows = [{'similar stretch': f"{h['window_start']} → {h['window_end']}",
                                  'how similar': f"{h['similarity']:.0%}",
                                  'numbers it shared with now': ", ".join(map(str, h['shared_numbers'])) or '—',
                                  f'{num} won on': h['followup_date']} for h in exp['trend_hits']]
                        st.dataframe(pd.DataFrame(trows), width='stretch', hide_index=True)
                        st.caption("Similarity mixes the shape of the draws (sums, spread, odd/even, "
                                   "digit families) with the actual numbers shared. Stretches "
                                   "overlapping the present are never used, so this can't just copy "
                                   "last week.")
                    else:
                        st.caption(f"{num} didn't follow any of the stretches most similar to the "
                                   f"current one.")
                    st.write(f"**{num} in the other games this week** — the 'today's results point at "
                             f"tomorrow's game' reading:")
                    if exp['cross_game_hits']:
                        crows = [{'game': config.NAMES.get(h['source_game'], h['source_game']),
                                  'drew it on': h['source_date'],
                                  'how often that game "sends" numbers here': f"{h['rate']:.1%}",
                                  'past cases': f"{h['trials']:,}"} for h in exp['cross_game_hits']]
                        st.dataframe(pd.DataFrame(crows), width='stretch', hide_index=True)
                        st.caption("Chance is 5.6% — check whether the sending game's real record "
                                   "beats it.")
                    else:
                        st.caption(f"No other game has drawn {num} since this game's last draw.")
                with colD:
                    st.write(f"**How often {num} follows a draw like this one** — this draw's profile "
                             f"(total, odd/even mix, spread, repeat or not) matched against history:")
                    if exp['conditional_hits']:
                        cond_names = {'sum_band': 'draw total', 'parity': 'odd/even mix',
                                      'span_band': 'spread', 'repeat_prev': 'repeat from last draw'}
                        krows = [{'this draw is': f"{cond_names.get(h['dim'], h['dim'])}: {h['value']}",
                                  f'{num} followed such draws': f"{h['rate']:.1%} of the time",
                                  'past cases': f"{h['trials']:,}"} for h in exp['conditional_hits']]
                        st.dataframe(pd.DataFrame(krows), width='stretch', hide_index=True)
                        st.caption("Chance is 5.6% per draw — a rate only matters if it clearly beats "
                                   "that across many cases.")
                    else:
                        st.caption(f"{num} has never followed a draw with this profile.")
                    st.write(f"**Machine pairs seen before — did {num} follow?** "
                             f"({exp['mach_trace_total_repeats']} past repeat(s) of this draw's machine "
                             f"pairs; {exp['mach_trace_confidence']}):")
                    if exp['mach_trace_hits']:
                        mrows = [{'machine group that repeated': " & ".join(str(g) for g in h['groups']),
                                  'repeated on': h['repeated_on'],
                                  f'{num} won': f"{h['lag']} draw(s) later, {h['followup_date']}"}
                                 for h in exp['mach_trace_hits']]
                        st.dataframe(pd.DataFrame(mrows), width='stretch', hide_index=True)
                    else:
                        st.caption(f"No repeat of this draw's machine pairs was followed by {num} winning.")
                    yi = exp['yearly_info']
                    if yi and yi['eligible_years']:
                        st.write(f"**{num} around this date in past years**: it appeared within a week "
                                 f"of this same calendar date in **{yi['hits']} of "
                                 f"{yi['eligible_years']}** years "
                                 f"({(yi['hits']/yi['eligible_years']):.0%} — luck alone would give "
                                 f"about {yi['chance_per_window']:.0%}).")
                        if yi['hit_years']:
                            st.caption("The years it showed up: " +
                                       ", ".join(f"{y} ({d})" for y, d in yi['hit_years'][-8:]))

            top5 = [n for n, _ in sorted(sc.items(), key=lambda t: (-t[1], t[0]))[:5]]
            with st.spinner("Tracing the evidence behind each pick..."):
                exps = {n: pattern_analysis.explain(seq, n, all_draws=all_seq) for n in top5}

            st.write(f"**Evidence map — what supports each of the 5 picks** "
                     f"(each cell: that evidence type's share of the pick's score):")
            map_rows = []
            for i, n in enumerate(top5, 1):
                row = {'pick': f"#{i} — {n}"}
                for name, v in exps[n]['contribution'].items():
                    row[PLAIN_LABELS.get(name, name)] = f"{v['contribution_pct']:.0f}%"
                map_rows.append(row)
            st.dataframe(pd.DataFrame(map_rows), width='stretch', hide_index=True)
            st.caption("Read it row by row: the biggest percentages are why that number made the top 5. "
                       "Open the number's tab below to see the actual draws behind each column.")

            pick_tabs = st.tabs([f"#{i} · {n}" for i, n in enumerate(top5, 1)])
            for i, (pt, n) in enumerate(zip(pick_tabs, top5), 1):
                with pt:
                    _render_explanation(n, exp=exps[n], rank=i)

            st.divider()
            st.write("**Explain any other number:**")
            other_num = st.number_input("Number", min_value=1, max_value=90, value=top5[0], step=1, key="pattern_explain_other")
            if int(other_num) not in top5:
                _render_explanation(int(other_num))
            else:
                st.caption("That number is already shown in the tabs above.")

            st.divider()
            st.write("**All recent pattern-trace events for this draw's pairs/triplets** "
                      "(not filtered to a single number):")
            events = pattern_analysis.pattern_trace_events(seq, top_n=15)
            if events:
                ev_rows = [{'groups': e['groups'], 'positions_then': _fmt_positions(e['seed_positions_then']),
                            'positions_now': _fmt_positions(e['seed_positions_now']), 'repeated_on': e['repeat_date'],
                            'followed_by (in order)': [f['win'] for f in e['followups']]} for e in events]
                st.dataframe(pd.DataFrame(ev_rows), width='stretch', hide_index=True)
            else:
                st.caption("No historical repeats found for this draw's pairs/triplets yet.")

            st.divider()
            st.write("**Transform-rule reliability** (measured across this game's entire history):")

            def _rule_group_size(name):
                if name in te_mod.SINGLE_RULES:
                    return 1
                return 3 if name in te_mod.TRIPLET_RULES else 2

            rule_rates = te_mod.measure_rule_rates(seq)
            rule_df = pd.DataFrame([{'rule': name, 'measured_rate': f"{rate:.2%}", 'trials': trials,
                                      'group_size': _rule_group_size(name)}
                                     for name, (rate, trials) in rule_rates.items()]).sort_values('measured_rate', ascending=False)
            st.dataframe(rule_df, width='stretch', hide_index=True)
            st.caption("Luck alone would put every trick at about 5.6% — tricks sitting on that line have "
                       "no real edge, they just exist as reading traditions. The score credit a trick "
                       "actually gets is its measured rate reduced for sample size, so a trick tested a "
                       "handful of times can never outshout one tested thousands of times.")
        elif mode == 'spatial':
            from lottery_core import spatial_engine as se
            SPATIAL_PLAIN = {
                'ncc': 'Look-alike weeks', 'diagonal': 'Diagonal lines', 'box': 'Box partners',
                'vshape': 'V-shapes', 'keys': 'Position equations', 'mach_cross': 'Machine crossover',
                'conditional': 'Draws exactly like this', 'periodic': 'Counting-weeks rhythm',
            }
            st.divider()
            st.subheader("Show your work — the chart-reading engine")
            st.caption(
                "This engine reads the results history like a paper chart: it looks for weeks that "
                "**look like** the recent ones, lines and shapes running across the chart, position "
                "arithmetic, and numbers returning on a steady rhythm. Start with the **evidence "
                "map**: it shows which kind of chart evidence is carrying each of the 5 picks."
            )
            with st.expander("What each kind of evidence means (30 seconds)"):
                st.markdown(
                    "- **Look-alike weeks** — history's stretches most similar to the last 3 weeks; "
                    "whatever dropped right after each one gets a vote.\n"
                    "- **Diagonal lines** — a number stepping +1/−1 (or ±2) week after week; the next "
                    "step of an unfinished line is suggested.\n"
                    "- **Box partners** — numbers that used to be drawn together with one of last "
                    "draw's numbers; the missing partner is suggested.\n"
                    "- **V-shapes** — numbers descending then rising symmetrically on the chart.\n"
                    "- **Position equations** — arithmetic linking one week's positions to the "
                    "next (like 1st number + 2nd = next week's 5th).\n"
                    "- **Machine crossover** — machine numbers turning into winners some draws later.\n"
                    "- **Draws exactly like this** — past draws matching this one's whole profile "
                    "(total, odd/even mix, spread, repeat) and who followed them.\n"
                    "- **Counting-weeks rhythm** — a number returning every g draws, whose next beat "
                    "lands on the coming draw.\n\n"
                    "Every line shows its real historical success rate next to what luck alone would "
                    "give (usually 5.6%) and how many past cases were checked — that comparison is "
                    "the whole story."
                )
            state = se.spatial_state(seq)

            _, sp_comps = se.spatial_scores(seq, state=state)
            sp_active = {n: w for n, w in se.SPATIAL_WEIGHTS.items() if n in sp_comps and w > 0}
            sp_norm = {n: ens_mod.normalize(s) for n, s in sp_comps.items()}
            sp_last = set(seq[-1]['win'])

            def _sp_contrib(num):
                excl = {'box'} if num in sp_last else set()
                local = {n: w for n, w in sp_active.items() if n not in excl} or sp_active
                tw = sum(local.values())
                contrib = {n: (w / tw) * sp_norm[n].get(num, 0.0) for n, w in local.items()}
                tot = sum(contrib.values()) or 1e-9
                return {n: c / tot * 100 for n, c in contrib.items()}

            sp_top5 = [n for n, _ in sorted(sc.items(), key=lambda t: (-t[1], t[0]))[:5]]
            st.write("**Evidence map — what supports each of the 5 picks** "
                     "(each cell: that evidence type's share of the pick's score):")
            sp_map = []
            for i, n in enumerate(sp_top5, 1):
                c = _sp_contrib(n)
                row = {'pick': f"#{i} — {n}"}
                for key_, label in SPATIAL_PLAIN.items():
                    row[label] = f"{c.get(key_, 0.0):.0f}%"
                sp_map.append(row)
            st.dataframe(pd.DataFrame(sp_map), width='stretch', hide_index=True)
            st.caption("Read it row by row: the biggest percentages are why that number made the "
                       "top 5. The tables below hold the actual past draws behind each column.")

            ncc_events, ncc_total, _ = se.ncc_template_matches(seq)
            st.write(f"**Look-alike weeks** — the {len(ncc_events)} stretches (of {ncc_total} in this "
                     f"game's history) most similar to the last 3 weeks, and what dropped right after "
                     f"each:")
            if ncc_events:
                st.dataframe(pd.DataFrame([
                    {'similar stretch': f"{e['window_start']} → {e['window_end']}",
                     'how similar': f"{e['score']:.0%}",
                     'numbers in common': e['overlap_cells'],
                     'what dropped next (the vote)': ", ".join(map(str, e['drop_win'])),
                     'on': e['drop_date']} for e in ncc_events]),
                    width='stretch', hide_index=True)
                st.caption("A pick supported by 'Look-alike weeks' appeared often in these "
                           "drop rows — check the last two columns for your numbers.")
            else:
                st.caption("Not enough history to compare stretches yet.")

            colE, colF = st.columns(2)
            with colE:
                st.write("**Diagonal lines** — a number stepping the same amount each week. How often "
                         "the next step of a 2-week line actually lands (luck alone: ~5.6%):")
                st.dataframe(pd.DataFrame([
                    {'step per week': f"{'+' if d > 0 else ''}{d}", 'next step lands': f"{r:.1%}",
                     'past cases': f"{t:,}"}
                    for d, (r, t) in state['diag_rates'].items()]), width='stretch', hide_index=True)
                projections = se.diagonal_projections(seq, rates=state['diag_rates'])
                if projections:
                    st.write("Lines running through the last two draws, and the number each one "
                             "suggests next:")
                    st.dataframe(pd.DataFrame([
                        {'line so far': " → ".join(str(c) for _, c in p['run']),
                         'step': f"{'+' if p['delta'] > 0 else ''}{p['delta']}",
                         'suggests': p['projected']}
                        for p in projections]), width='stretch', hide_index=True)
                else:
                    st.caption("No diagonal line runs into the current draw.")

                full_v, partial_v, v_rate, v_trials = se.v_shape_events(seq)
                st.write(f"**V-shapes**: {len(full_v)} complete V pattern(s) exist in this game's "
                         f"entire history.")
                if partial_v:
                    st.dataframe(pd.DataFrame([{'step': p['delta'], 'suggests': p['projected']}
                                               for p in partial_v]), width='stretch', hide_index=True)
                else:
                    st.caption("No V shape is forming right now. With 5 numbers out of 90 per draw, "
                               "complete V's almost never exist — we show that truth rather than "
                               "invent one.")

            with colF:
                box_rate, box_trials = state['box_rate']
                st.write(f"**Box partners** — when a number appears, its old draw-mates follow within "
                         f"{se.TRACE_LOOKAHEAD} draws {box_rate:.0%} of the time "
                         f"({box_trials:,} past cases; luck alone: "
                         f"{1 - (85/90)**se.TRACE_LOOKAHEAD:.0%}). The strongest partner suggestions "
                         f"right now:")
                _, box_detail = se.box_score(seq, rate_trials=state['box_rate'])
                box_rows = sorted(((k, v) for k, v in box_detail.items() if v and not v[0]['self_credit']),
                                  key=lambda kv: -sum(h['weight'] for h in kv[1]))[:10]
                if box_rows:
                    st.dataframe(pd.DataFrame([
                        {'suggested number': k,
                         'old draw-mate(s) in last draw': ", ".join(map(str, sorted({h['partner_of'] for h in v}))),
                         'times drawn together before': sum(h['pair_rows'] for h in v)}
                        for k, v in box_rows]),
                        width='stretch', hide_index=True)

                st.write("**Machine crossover** — how often machine numbers become winners some draws "
                         "later (luck alone: ~5.6%):")
                st.dataframe(pd.DataFrame([
                    {'draws later': tau, 'how often': f"{r:.1%}", 'past cases': f"{t:,}"}
                    for tau, (r, t) in state['mach_rates'].items()]), width='stretch', hide_index=True)

                from lottery_core import trend_analysis as ta_core
                _cscores, cmeta = ta_core.joint_conditional_score(seq, seq[-1],
                                                                  rates_and_bounds=state['cond_rates'])
                if cmeta['key']:
                    cond_names = {'low': 'low total', 'mid': 'medium', 'high': 'high total',
                                  'tight': 'tight spread', 'wide': 'wide spread',
                                  'odd_heavy': 'mostly odd', 'even_heavy': 'mostly even',
                                  'balanced': 'balanced odd/even',
                                  'repeat': 'repeated a number', 'no_repeat': 'no repeat'}
                    key_str = ", ".join(cond_names.get(v, v) for v in cmeta['key'])
                    st.write(f"**Draws exactly like this one** — same profile ({key_str}) has happened "
                             f"**{cmeta['cohort']}** time(s) before. Who showed up right after:")
                    if cmeta['detail']:
                        top_cond = sorted(cmeta['detail'].items(),
                                          key=lambda kv: -kv[1][0]['appearances'])[:10]
                        st.dataframe(pd.DataFrame([
                            {'number': k,
                             'followed such draws': f"{v[0]['appearances']} of {v[0]['cohort']} times",
                             'rate': f"{v[0]['rate']:.0%}"}
                            for k, v in top_cond]), width='stretch', hide_index=True)
                        st.caption(f"Luck alone gives each number about 5.6%. With only "
                                   f"{cmeta['cohort']} matching draws, a striking-looking rate can "
                                   f"easily be luck — the score credit is automatically cut down for "
                                   f"the small sample.")
                    else:
                        st.caption("No number has ever followed a draw with this exact profile.")
                else:
                    st.caption("**Draws exactly like this one**: not enough history to compare yet.")

                _pscores, pdetail = ta_core.counting_week_score(seq, stats=state['cw_stats'])
                cw2 = state['cw_stats'].get(2, (0.0, 0))
                cw3 = state['cw_stats'].get(3, (0.0, 0))
                st.write(f"**Counting-weeks rhythm** — numbers that have been returning on a steady "
                         f"beat (every g draws) whose next beat lands exactly on the coming draw:")
                firing = sorted(pdetail.items(), key=lambda kv: -sum(h['weight'] for h in kv[1]))[:10]
                if firing:
                    st.dataframe(pd.DataFrame([
                        {'number': k,
                         'its rhythm': f"every {v[0]['gap']} draws, kept up {v[0]['chain_length']} times",
                         'rhythm ran': f"{v[0]['chain_dates'][0]} → {v[0]['chain_dates'][-1]}"}
                        for k, v in firing]),
                        width='stretch', hide_index=True)
                    st.caption(f"Reality check, measured over this game's whole history: rhythms like "
                               f"these keep going {cw2[0]:.1%}–{cw3[0]:.1%} of the time "
                               f"({cw2[1]:,}+ past cases) — right at the 5.6% luck line. A rhythm "
                               f"'landing on today' feels meaningful but isn't; the score credit "
                               f"reflects the measured rate, not the feeling.")
                else:
                    st.caption("No number's rhythm lands on the coming draw this week.")

            st.divider()
            kr = state['key_report']
            st.write(f"**Position equations ('keys')** — arithmetic rules linking one week's "
                     f"positions to the next (like *2nd winning number − 1 = next week's 5th*). "
                     f"Out of {kr['tested']:,} possible rules tested, {len(kr['keys'])} currently "
                     f"look better than luck:")
            st.caption(f"⚠️ Important: when you test {kr['tested']:,} rules, about "
                       f"{kr['expected_false_positives']:.0f} will look this good by pure luck. So a "
                       f"rule on this list is a *candidate*, not a discovery — the noise test below "
                       f"is the real judge.")
            if kr['keys']:
                st.dataframe(pd.DataFrame([
                    {'rule': f"{k['src']} —{k['op']}→ {k['target']}",
                     'came true': f"{k['hits']} of {k['trials']} weeks",
                     'rate': f"{k['rate']:.1%} (luck: 1.1%)"} for k in kr['keys']]),
                    width='stretch', hide_index=True)
                if st.button("Noise test: do these rules also 'work' on fake random results?",
                             key="spatial_bootstrap"):
                    rows = []
                    with st.spinner("Re-testing each rule on shuffled, meaningless results..."):
                        for k in kr['keys'][:5]:
                            bp = se.bootstrap_key_pvalue(seq, k, iterations=300)
                            rows.append({'rule': f"{k['src']} —{k['op']}→ {k['target']}",
                                         'also works on fake results': f"{bp:.0%} of the time",
                                         'verdict': 'discard — luck can do this' if bp > 0.05
                                                    else 'rare in fake results — but remember the warning above'})
                    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
                    st.caption("We shuffle the entire history into meaningless random results and check "
                               "whether each rule still 'works'. A rule luck can reproduce is not a "
                               "rule. The definitive all-rules-at-once test runs from the command "
                               "line: `python predictor.py keys " + game + "`.")
            else:
                st.caption("No rule currently beats the luck line on this game's history.")

        elif mode == 'plans':
            import importlib
            from lottery_core import plan_engine as pl
            # Streamlit hot-reloads app.py but keeps imported modules cached from server
            # start, so an engine updated since then would render against a stale API and
            # crash on fields it doesn't have yet. Reload it explicitly and cheaply.
            if not hasattr(pl, 'plan_name'):
                pl = importlib.reload(pl)
            st.divider()
            st.subheader("Show your work — the plans it learned")
            st.caption(
                "This engine doesn't just find weeks that look like this one and copy what dropped "
                "after them. It goes one step further: it takes what dropped, **traces each of those "
                "numbers back** to where it came from in the weeks before (up to 10 weeks back), and "
                "the way they kept arriving becomes a **plan**. Every plan is then replayed on the "
                "current week to give its own numbers. Several plans are learned — the strongest one "
                "leads, the rest are all kept and shown below with their own numbers and record.\n\n"
                "It starts the way a reader starts: **take a pair from this week's draw and hunt for "
                "the weeks where that same pair came out together.** Those anchored weeks are then "
                "narrowed further by structure — weeks built like this one, with the same spacing, "
                "spread, clustering and draw-order shape. It combines **temporal** behaviour (how "
                "numbers travel from week to week, up to ten weeks back) with **spatial** structure "
                "(where numbers sit on the chart grid, which slot they came out of)."
            )
            with st.spinner("Learning plans from the chart…"):
                plans_all = pl.discover_plans(seq)
                _psc, prep = pl.plan_scores(seq, plans=plans_all)

            best = prep['best']
            if not best:
                st.warning("Not enough history in this game to learn a plan yet.")
            else:
                st.markdown("## The plan it trusts most")
                st.markdown(f"### {pl.plan_name(best)}")
                st.caption(f"Found by looking at {pl.matcher_short(best)}  ·  internal name: "
                           f"`{best['id']}`")
                c1, c2, c3 = st.columns(3)
                c1.metric("Numbers it gives for this week", " · ".join(map(str, prep['best_numbers'])))
                c2.metric("How often it was right", f"{best['rate']:.1%}", f"luck gets 5.6%")
                c3.metric("Learned from", f"{best['situations']} past weeks")

                st.write("**How it works, step by step:**")
                for i, s in enumerate(prep.get('best_steps') or [], 1):
                    st.markdown(f"{i}. {s}")

                if best.get('anchored') and best.get('top_anchors'):
                    st.info("**Where the search started:** the pairs from this week's draw that were "
                            "used to find lookalike weeks — "
                            + ", ".join(f"**{' & '.join(map(str, a['numbers']))}** "
                                        f"({a['weeks']} week" + ("" if a['weeks'] == 1 else "s") + ")"
                                        for a in best['top_anchors'][:4]))

                st.write("**The past weeks it learned this from:**")
                if best['evidence']:
                    ev_rows = []
                    for e in best['evidence']:
                        row = {'the week': e['drop_date']}
                        # Only anchored plans have a pair to show; for shape-matched plans
                        # the column would be nothing but dashes, so it is left out.
                        if best.get('anchored'):
                            row['found using this pair'] = " & ".join(map(str, e.get('anchor') or []))
                        row.update({
                            'numbers that dropped': " · ".join(map(str, e['dropped'])),
                            'the ones this plan got right': " · ".join(map(str, e['explained'])),
                            'which came from the draw of': e['source_date'],
                            'how many it suggested that week': e['n_proposed']})
                        ev_rows.append(row)
                    st.dataframe(pd.DataFrame(ev_rows), width='stretch', hide_index=True)
                    if not best.get('anchored'):
                        st.caption("This plan found its lookalike weeks by the **shape** of the draw "
                                   "rather than by starting from a pair, so there is no anchoring "
                                   "pair to show. The best pair-anchored plan is shown below.")
                    st.caption("Read one row like this: on that date, those numbers dropped — and the "
                               "ones in the third column can be worked out from the earlier draw in "
                               "the fourth column, using this plan's method. Seeing that happen again "
                               "and again is how the plan was learned, and doing the same thing to "
                               "this week's draw is where its numbers above come from.")

                st.divider()
                st.markdown("## The other plans it learned")
                st.write(f"It kept {prep['n_qualifying']} plans that held up (out of "
                         f"{prep['n_plans_evaluated']} it tried). Each one gives its own numbers:")
                st.dataframe(pd.DataFrame([
                    {'the plan': pl.plan_name(p),
                     'looks back': f"{p['lag']} week" + ("" if p['lag'] == 1 else "s"),
                     'finds its weeks by': pl.matcher_short(p),
                     'its numbers for this week': " · ".join(map(str, p['now'])),
                     'how often it was right': f"{p['rate']:.1%}",
                     'no. of weeks it learned from': p['situations'],
                     'no. of numbers it has suggested': f"{p['proposals']:,}"}
                    for p in ([best] + prep['others'])]), width='stretch', hide_index=True)
                st.caption("Luck alone gets 5.6% right, so compare every plan to that. 'Weeks it "
                           "learned from' matters just as much as the percentage: a plan built on a "
                           "handful of weeks is a coincidence dressed up as a method, and the app "
                           "trusts it less accordingly.")

                ba_plan = prep.get('best_anchored')
                if ba_plan and ba_plan is not best:
                    st.divider()
                    st.markdown("## The best plan that started from a pair")
                    st.markdown(f"### {pl.plan_name(ba_plan)}")
                    st.caption(f"Found by looking at {pl.matcher_short(ba_plan)}  ·  internal name: "
                               f"`{ba_plan['id']}`")
                    st.caption("Shown separately because starting from a pair of this week's own "
                               "numbers is the way the reading is meant to begin — so it stays "
                               "visible even when a shape-matched plan happens to score higher.")
                    a1, a2, a3 = st.columns(3)
                    a1.metric("Numbers it gives", " · ".join(map(str, ba_plan['now'])))
                    a2.metric("How often it was right", f"{ba_plan['rate']:.1%}", "luck gets 5.6%")
                    a3.metric("Learned from", f"{ba_plan['situations']} past weeks")
                    if ba_plan.get('top_anchors'):
                        st.info("**The pairs it started from:** "
                                + ", ".join(f"**{' & '.join(map(str, a['numbers']))}** "
                                            f"({a['weeks']} week" + ("" if a['weeks'] == 1 else "s") + ")"
                                            for a in ba_plan['top_anchors'][:4]))
                    if ba_plan['evidence']:
                        st.dataframe(pd.DataFrame([
                            {'the week': e['drop_date'],
                             'found using this pair': " & ".join(map(str, e.get('anchor') or [])),
                             'numbers that dropped': " · ".join(map(str, e['dropped'])),
                             'the ones this plan got right': " · ".join(map(str, e['explained'])),
                             'which came from the draw of': e['source_date']}
                            for e in ba_plan['evidence']]), width='stretch', hide_index=True)

                ba, bu = prep.get('best_anchored'), prep.get('best_unanchored')
                if ba and bu:
                    lead = ("Starting from a pair is ahead here — the anchor is doing real work."
                            if ba['weight'] >= bu['weight'] else
                            "Searching without an anchor is ahead here — meaning starting from a pair "
                            "isn't adding anything on this game right now. Worth knowing rather than "
                            "hiding.")
                    st.caption(
                        f"**Does starting from a pair actually help?** Best plan that began by hunting "
                        f"a pair from this week's draw: {ba['rate']:.1%}. Best plan that searched "
                        f"without any anchor: {bu['rate']:.1%}. {lead}")

                bs, bf = prep.get('best_structural'), prep.get('best_surface')
                if bs and bf:
                    lead = ("Shape-based matching is ahead here."
                            if bs['weight'] >= bf['weight'] else
                            "Number-based matching is ahead here.")
                    st.caption(
                        f"**And does the shape of the week matter?** Best plan matching weeks by how "
                        f"they were BUILT (spacing, spread, clustering, draw order): {bs['rate']:.1%}. "
                        f"Best plan matching by the numbers themselves: {bf['rate']:.1%}. {lead}")

                if prep.get('matchers_skipped'):
                    st.caption("Not enough history to learn from: "
                               + ", ".join(f"`{m}`" for m in prep['matchers_skipped'])
                               + ". (Sharing three or more numbers with the current draw, for "
                                 "instance, happens too rarely to build a plan on — so no plan is "
                                 "offered rather than one built on a couple of coincidences.)")

                st.divider()
                st.markdown("## Why this plan and not the others")
                comparison = prep.get('comparison') or {'why_primary': '', 'differences': []}
                st.markdown(comparison['why_primary'])
                if comparison['differences']:
                    st.write("**How the others differ from it:**")
                    st.dataframe(pd.DataFrame([
                        {'the plan': d.get('name', d['plan']),
                         'its numbers': " · ".join(map(str, d['numbers'])),
                         'how it differs': d['difference']}
                        for d in comparison['differences']]),
                        width='stretch', hide_index=True)

                derivs = prep.get('derivations') or {}
                top5_plans = [n for n, _ in sorted(_psc.items(), key=lambda t: (-t[1], t[0]))[:5]]
                st.divider()
                st.markdown("## The final numbers, and where each one came from")
                st.caption("These are the picks once all the plans are weighed together. Every number "
                           "here can be traced — if nothing could explain a number, it doesn't get "
                           "picked.")
                rows = []
                for num in top5_plans:
                    ds = derivs.get(num, [])
                    if ds:
                        d0 = ds[0]
                        rows.append({'number': num,
                                     'plans that back it': len(ds),
                                     'best reason for it': d0['mechanism'],
                                     'worked out from the draw of': f"{d0['from_draw']} "
                                                                    f"({d0['lag']} weeks back)",
                                     'how often that plan was right': f"{d0['rate']:.1%}"})
                if rows:
                    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

                if prep['echoes']:
                    with st.expander("Plans that were thrown out for just copying last week"):
                        st.dataframe(pd.DataFrame([
                            {'the plan': pl.plan_name(p), 'how often it was right': f"{p['rate']:.1%}"}
                            for p in prep['echoes']]), width='stretch', hide_index=True)
                        st.caption("Some of these look excellent — but 'the same number comes back "
                                   "from one week ago' just hands you last week's draw again. That "
                                   "isn't a forecast, so these are measured and shown here, but never "
                                   "allowed to choose numbers.")

                st.divider()
                st.markdown("#### ⚠️ The test that matters most")
                st.caption(
                    f"This engine tried **{prep['n_plans_evaluated']} different plans** and showed you "
                    f"the best one. Try that many plans on *anything* — even numbers pulled out of a "
                    f"hat — and a few will look brilliant purely by luck. So the fair question isn't "
                    f"'does the winning plan look good?' It's: **would a plan this good still turn up "
                    f"if the results were completely made up?** The button below checks exactly that."
                )
                if st.button("Run the noise test on the whole plan search", key="plans_bootstrap"):
                    with st.spinner("Re-learning plans from scratch on shuffled fake histories…"):
                        bres = pl.bootstrap_best_plan_pvalue(seq, iterations=40)
                    st.write(f"Fake random histories that produced an **even better** best plan: "
                             f"**{bres['n_noise_wins']} of {bres['iterations']}** "
                             f"({bres['bootstrap_p']:.0%}).")
                    if bres['bootstrap_p'] > 0.05:
                        st.error(f"Verdict: {bres['verdict']}")
                    else:
                        st.warning(f"Verdict: {bres['verdict']}")

        else:
            st.divider()
            st.subheader("Show your work")
            top5 = [n for n, _ in sorted(sc.items(), key=lambda t: (-t[1], t[0]))[:5]]

            def _render_stat_explanation(num):
                if mode == 'charts':
                    exp = pattern_analysis.explain_charts(seq, num)
                    st.markdown(f"**{exp['narrative']}**")
                    if exp['chart_hits']:
                        st.write(f"**Chart pointers landing on {num}** (from the last draw's numbers):")
                        st.dataframe(pd.DataFrame(exp['chart_hits']), width='stretch', hide_index=True)
                elif mode == 'charts2':
                    from lottery_core import chart_analysis
                    exp = chart_analysis.explain_number(seq, num, all_draws=all_seq)
                    st.markdown(f"**{exp['narrative']}**")
                    if exp['chart_hits']:
                        st.write(f"**Chart pointers landing on {num}** (each with its OWN measured entry "
                                 f"record, pooled across all 7 games; machine-sourced pointers at their "
                                 f"own measured rates, not an assumed 0.5x):")
                        st.dataframe(pd.DataFrame([
                            {'chart': p['chart'], 'from': p['from'], 'source': p['from_kind'],
                             'entry_record': f"{p['entry_hits']}/{p['entry_trials']}",
                             'chart_avg': (f"{p['chart_rate']:.2%}" if p['chart_rate'] is not None else '—'),
                             'shrunk_weight': f"{p['weight']:.2%}", 'self_pointer': p['self_pointer']}
                            for p in exp['chart_hits']]), width='stretch', hide_index=True)
                else:
                    exp = classic.explain(seq, num, mode)
                    st.markdown(f"**{exp['narrative']}**")
                    st.caption(exp['formula'])
                    st.dataframe(pd.DataFrame([{
                        'drawn (all-time)': exp['freq_all'], 'drawn (last 30)': exp['freq_30'],
                        'recency-weighted freq': round(exp['wfreq'], 3),
                        'draws since last seen': exp['gap'], 'expected gap': round(exp['expected_gap'], 1),
                        'last seen on': exp['last_seen_date'],
                    }]), width='stretch', hide_index=True)

            captions = {
                'hot': "Every number below is traced back to its raw drawn-count -- 'hot' has no recency "
                       "or gap weighting at all, so this is the whole story.",
                'recent': "Every number below is traced back to its recency-weighted frequency (half-life "
                          "60 draws) -- older wins count for progressively less than recent ones.",
                'overdue': "Every number below is traced back to how many draws it's gone unseen versus the "
                           "expected gap for a 5/90 game (90/5 = 18 draws) -- purely a gap calculation.",
                'blend': "Every number below is traced back to its recency-weighted frequency plus the "
                         "overdue bonus (capped at 45%) that combines with it.",
                'charts': "Every number below is traced back to which chart(s) point to it from the last "
                          "draw's numbers (with that chart's measured historical transfer rate), plus the "
                          "recency-weighted-frequency baseline blended in alongside it.",
                'charts2': "The upgraded chart strategy: every pointer scores at its OWN entry's measured "
                           "transfer record (pooled across all 7 games, ~150 trials per entry), shrunk "
                           "toward the chart average by sample size; machine-sourced pointers score at "
                           "their own measured rates instead of the legacy assumed 0.5x. The legacy "
                           "'charts' mode is kept unchanged as the backtest baseline.",
            }
            st.caption(captions.get(mode, "") +
                       " See the Methodology tab: none of these strategies has shown a real edge over chance.")

            pick_tabs = st.tabs([str(n) for n in top5])
            for pt, n in zip(pick_tabs, top5):
                with pt:
                    _render_stat_explanation(n)

            st.divider()
            st.write("**Explain any other number:**")
            other_num = st.number_input("Number", min_value=1, max_value=90, value=top5[0], step=1,
                                         key=f"{mode}_explain_other")
            if int(other_num) not in top5:
                _render_stat_explanation(int(other_num))
            else:
                st.caption("That number is already shown in the tabs above.")

with tab_ml:
    st.header("ML Models")
    st.caption(
        "The statistical/machine-learning system: a legacy NumPy logistic regression plus a "
        "scikit-learn ensemble (Random Forest, HistGradientBoosting, MLP) and a PyTorch LSTM. "
        "**One independent model per game** -- a data update for one game only ever retrains "
        "and changes that game's own picks, never another game's. **Requires training**: after "
        "`predictor.py update`, re-run `python train.py` (optionally `--games MS` to retrain just "
        "one) or these picks are stale."
    )
    col2b, game_ml, upto_ml = _game_date_picker("ml")
    mode_ml = col2b.selectbox("Strategy", list(ML_MODES), index=list(ML_MODES).index('ensemble'), key="ml_mode")
    if mode_ml != 'ml':
        game_meta = artifacts.game_artifact_meta(game_ml, 'rf')
        game_fp_now = artifacts.game_data_fingerprint(game_ml)
        if game_meta is None:
            st.warning(f"No trained artifact for {config.NAMES[game_ml]} yet -- retrain in the sidebar.")
        elif game_meta.get('game_fingerprint') != game_fp_now:
            st.warning(f"{config.NAMES[game_ml]}'s artifact is stale (its data changed since "
                       f"{game_meta.get('trained_at', '?')}) -- retrain in the sidebar.")
        else:
            st.caption(f"{config.NAMES[game_ml]}'s model last trained {game_meta.get('trained_at', '?')} "
                       f"({'quick' if game_meta.get('quick') else 'full'}).")
    seq_ml = [d for d in draws if d['code'] == game_ml and (upto_ml is None or d['date'] < upto_ml)]
    if len(seq_ml) < 30:
        st.warning(f"Not enough history for {config.NAMES[game_ml]} before {upto_ml}.")
    else:
        with st.spinner(f"Scoring with strategy '{mode_ml}'..."):
            sc_ml = get_scores_any(seq_ml, mode_ml, game_ml)
        _picks_header(game_ml, upto_ml, sc_ml, mode_ml)
        st.caption("These picks are ranked by the selected strategy's score, not a probability of "
                   "winning. See the Methodology tab: no strategy has shown a real edge over random chance.")

with tab_perf:
    st.header("Model Performance")
    if not bt_cache:
        st.warning("No backtest cache found. Run `python train.py` to generate one.")
    else:
        results = bt_cache['results']
        meta = bt_cache.get('meta', {})
        st.caption(f"{meta.get('n_draws', '?')} draws, min_hist={meta.get('min_hist', '?')}, "
                   f"quick={meta.get('quick', '?')}")
        col1, col2 = st.columns(2)
        k = col1.selectbox("Number of picks (k)", [2, 3, 4, 5], index=3)
        game_filter = col2.selectbox("Game filter", ["All games (pooled)"] + config.GAMES,
                                      format_func=lambda g: "All games (pooled)" if g == "All games (pooled)" else config.NAMES[g])
        pg = None if game_filter == "All games (pooled)" else game_filter
        st.plotly_chart(fig_hitrate_comparison(results, k, per_game=pg), width='stretch')
        st.plotly_chart(fig_auc_brier(results, per_game=pg), width='stretch')
        st.caption("Error bars are 95% Wilson confidence intervals. Where a strategy's bar and error bar "
                   "overlap the random-chance line, that strategy has not demonstrated a real edge.")

with tab_explore:
    st.header("Data Explorer")
    games_sel = st.multiselect("Games", config.GAMES, default=config.GAMES,
                                format_func=lambda g: config.NAMES[g])
    filtered = [d for d in draws if d['code'] in games_sel]
    df = pd.DataFrame([{
        'date': d['date'], 'game': config.NAMES[d['code']], 'code': d['code'],
        'w1': d['win'][0], 'w2': d['win'][1], 'w3': d['win'][2], 'w4': d['win'][3], 'w5': d['win'][4],
        'm1': (d['mach'][0] if d['mach'] else None), 'm2': (d['mach'][1] if d['mach'] else None),
        'm3': (d['mach'][2] if d['mach'] else None), 'm4': (d['mach'][3] if d['mach'] else None),
        'm5': (d['mach'][4] if d['mach'] else None),
        'sum': sum(d['win']),
    } for d in filtered])
    st.dataframe(df.sort_values('date', ascending=False), width='stretch', height=300)

    col1, col2 = st.columns(2)
    with col1:
        freq = {k: 0 for k in range(1, 91)}
        for d in filtered:
            for n in d['win']:
                freq[n] += 1
        grid = [[freq[r * 10 + c + 1] for c in range(10)] for r in range(9)]
        fig = go.Figure(go.Heatmap(z=grid, x=[str(i) for i in range(1, 11)],
                                    y=[f"{r*10+1}-{r*10+10}" for r in range(9)], colorscale="Blues"))
        fig.update_layout(title="Winning-number frequency heatmap (1-90)", height=400, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width='stretch')
    with col2:
        fig2 = go.Figure(go.Histogram(x=df['sum'], nbinsx=30, marker_color="#4C78A8"))
        fig2.update_layout(title="Draw-sum distribution", height=400, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, width='stretch')

    st.subheader("Machine numbers")
    st.caption("Ghana NLA draws publish 5 winning numbers plus 5 'machine' numbers. A common belief among "
               "lottery-paper readers is that a number drawn as a machine number tends to reappear as a "
               "winning number soon after. Below is that belief tested directly against this game's history, "
               "not assumed -- these numbers also feed the rf/gbm/mlp/deep/ensemble strategies as features.")
    mach_present = [d for d in filtered if d['mach']]
    if not mach_present:
        st.info("No machine-number data for the selected games/date range (published from Aug 2018 onward).")
    else:
        col3, col4 = st.columns(2)
        with col3:
            mfreq = {k: 0 for k in range(1, 91)}
            for d in mach_present:
                for n in d['mach']:
                    mfreq[n] += 1
            mgrid = [[mfreq[r * 10 + c + 1] for c in range(10)] for r in range(9)]
            figm = go.Figure(go.Heatmap(z=mgrid, x=[str(i) for i in range(1, 11)],
                                         y=[f"{r*10+1}-{r*10+10}" for r in range(9)], colorscale="Oranges"))
            figm.update_layout(title="Machine-number frequency heatmap (1-90)", height=400,
                                margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(figm, width='stretch')
        with col4:
            from lottery_core import features as feat_mod
            seq_sorted = sorted(mach_present, key=lambda d: d['date'])
            rates, trials = feat_mod.mach_to_win_affinity(seq_sorted)
            aff_df = pd.DataFrame([{'number': k, 'rate': rates[k], 'trials': trials[k]}
                                    for k in range(1, 91) if trials[k] >= 5]).sort_values('rate', ascending=False).head(20)
            chance_rate = 1 - (85 / 90) ** feat_mod.MACH_LOOKAHEAD
            fig4 = go.Figure(go.Bar(x=aff_df['number'].astype(str), y=aff_df['rate'] * 100,
                                     text=aff_df['trials'], marker_color="#F58518"))
            fig4.add_hline(y=chance_rate * 100, line_dash="dash", line_color="#E45756",
                            annotation_text="random chance")
            fig4.update_layout(title=f"P(machine number -> winning number within {feat_mod.MACH_LOOKAHEAD} draws), "
                                      f"min 5 occurrences", yaxis_title="%", height=400,
                                margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig4, width='stretch')
            st.caption("Bar labels show sample size (number of times each number appeared as a machine number). "
                       "With this few occurrences per number, individual bars are noisy -- compare against the "
                       "random-chance line rather than reading any single bar as a signal.")

    st.subheader("Gap since last seen")
    if filtered:
        seq_sorted = sorted(filtered, key=lambda d: d['date'])
        last_seen = {}
        for i, d in enumerate(seq_sorted):
            for n in d['win']:
                last_seen[n] = i
        n_total = len(seq_sorted)
        gaps = {k: n_total - last_seen.get(k, -1) - 1 for k in range(1, 91)}
        gap_df = pd.DataFrame(sorted(gaps.items(), key=lambda t: -t[1])[:20], columns=['number', 'draws_since_seen'])
        fig3 = go.Figure(go.Bar(x=gap_df['number'].astype(str), y=gap_df['draws_since_seen'], marker_color="#F58518"))
        fig3.update_layout(title="Most 'overdue' numbers (draws since last seen)", height=350,
                            margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig3, width='stretch')
        st.caption("'Overdue' is a gambler's-fallacy framing for independent draws -- shown here as a "
                   "descriptive stat only, not a prediction signal (the backtest confirms it carries none).")

with tab_charts:
    st.header("Chart Relationships (folk numerology)")
    st.caption("Traditional Ghana lotto 'chart' relationships from justlottoo.blogspot.com. "
               "Measured against 8+ years of draws, none of these transfer at a rate different from chance (5.56%).")
    col1, col2 = st.columns(2)
    chart_name = col1.selectbox("Chart", list(classic.CHARTS.keys()))
    number = col2.number_input("Number", min_value=1, max_value=90, value=1, step=1)
    partner = classic.CHARTS[chart_name].get(number)
    st.metric(f"{chart_name} partner of {number}", partner if partner else "none")

    rates = classic.chart_transfer_rates(draws)
    rate_df = pd.DataFrame([{'chart': name, 'transfer_rate_%': rate * 100} for name, rate in rates.items()])
    rate_df = pd.concat([rate_df, pd.DataFrame([{'chart': 'random chance', 'transfer_rate_%': 100 / 18}])], ignore_index=True)
    fig = go.Figure(go.Bar(x=rate_df['chart'], y=rate_df['transfer_rate_%'],
                            marker_color=["#E45756" if c == 'random chance' else "#4C78A8" for c in rate_df['chart']]))
    fig.update_layout(title="Measured chart transfer rate vs. chance (5.56%)", yaxis_title="%",
                       height=400, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("Entry-level chart analysis (the 'charts2' upgrade)")
    from lottery_core import chart_analysis as ca
    st.caption(
        "The legacy measurement above gives each chart ONE pooled rate; the upgraded "
        "'charts2' strategy measures every individual entry (a → b) on its own record, "
        "pooled across all 7 games (~150 trials per entry), and shrinks thin entries "
        "toward their chart's average. Below: the transfer-rate curve by lag (does the "
        "partner come next draw, or 'within a few weeks'?), and the individually "
        "best-looking entries — with the selection-effect warning they require."
    )
    curves = ca.lag_curves(draws)
    fig_lag = go.Figure()
    for name, by_lag in curves.items():
        fig_lag.add_trace(go.Scatter(name=name, x=list(by_lag.keys()),
                                     y=[r * 100 for r, _ in by_lag.values()], mode='lines+markers'))
    fig_lag.add_hline(y=100 * 5 / 90, line_dash="dash", line_color="#E45756")
    fig_lag.update_layout(title="Transfer rate by exact lag (dashed = 5.56% chance)",
                          xaxis_title="draws after source", yaxis_title="%",
                          height=400, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_lag, width='stretch')

    rep = ca.best_entry_report(draws)
    st.write(f"**Individually best-looking entries** ({rep['tested']} entries tested — with that many, "
             f"the best few are GUARANTEED to look impressive by selection alone):")
    st.dataframe(pd.DataFrame([
        {'chart': r['chart'], 'entry': f"{r['from']} → {r['to']}", 'record': f"{r['hits']}/{r['trials']}",
         'rate': f"{r['rate']:.1%}", 'binomial_p': f"{r['p_value']:.5f}"} for r in rep['top']]),
        width='stretch', hide_index=True)
    if st.button("Family-wise bootstrap: is the best entry better than noise's best entry?",
                 key="chart_entry_bootstrap"):
        with st.spinner("Re-running the best-entry search on structure-destroyed synthetic histories..."):
            bres = ca.bootstrap_best_entry_pvalue(draws, iterations=60)
        verdict = ("DISCARD: indistinguishable from apophenia — read every entry as a chart-average "
                   "performer, not a special relationship." if bres['bootstrap_p'] > 0.05 else
                   "the real best entry beats the noise baseline at this iteration count.")
        st.write(f"P(noise produces a better best-entry) = **{bres['bootstrap_p']:.2f}** "
                 f"({bres['iterations']} synthetic histories) → {verdict}")

with tab_method:
    st.header("Methodology & Honesty")
    st.markdown(methodology_text(bt_cache).replace("\n", "  \n"))
    st.divider()
    st.subheader("Why no stacked meta-learner in the ensemble")
    st.write(
        "The ensemble strategy is a fixed, normalized weighted average across strategies, not a "
        "trained meta-model. With only ~2,000 walk-forward test draws, a trainable stacker has enough "
        "free parameters to fit noise in the backtest and manufacture a fake edge -- exactly the "
        "failure mode this project's walk-forward validation is designed to catch."
    )
    st.subheader("Pattern Analysis's dynamic component weights -- a deliberate exception")
    st.write(
        "Unlike the ensemble above, the Pattern Analysis tab's component weights are NOT fixed: "
        "`dynamic_weights()` re-derives them on every call from which of the scored components (ten, "
        "eleven when cross-game data is available) placed the most score-mass on the actual winning "
        "numbers over the last 30 draws. This is a "
        "real departure from the anti-auto-tuning stance just explained, made deliberately -- but it "
        "comes with the exact same risk that stance exists to avoid: over ~2,700 *provably random* "
        "draws, 'which component looked best in the last 30 draws' is noise, not a regime to adapt to, "
        "so the resulting weights can drift toward whichever component's score happens to be less "
        "sparse (and therefore accumulates more incidental score-mass) rather than whichever is more "
        "skillful. It is also ~30x the cost of a fixed-weight score, mitigated here with a small cache "
        "(repeat calls against the same history reuse the last result) and, in the backtest, a "
        "periodic recompute every 20 test points rather than every single one. Treat any single "
        "component's dynamic weight as descriptive, not as evidence that component is actually better."
    )
    st.subheader("Why 'positional' and 'lap' were removed from the score (and a follow-up fix)")
    st.write(
        "Both used to be scored components. Measured on 375 real walk-forward test points across all "
        "7 games: with them included, the pattern-analysis top-5 picks overlapped **last week's own 5 "
        "numbers** at 2.379/5 on average -- 8.5x the ~0.278/5 you'd expect between two independent "
        "random 5-number sets -- while overlap with the **actual next draw** sat at exactly chance "
        "(0.285/5). The system wasn't forecasting; it was substantially replaying last week's draw. "
        "The cause was structural, not incidental: `positional_carryover_score` and `lap_score` are, "
        "by construction, nonzero *only* for the 5 numbers in the immediately preceding draw, and "
        "`terminal`/`group`/`charts`(`turning`) each trivially credit a number for sharing a class with "
        "*itself*. `positional` and `lap` are now diagnostic-only (still computed and shown in 'Show "
        "your work', just excluded from `component_scores()`); `terminal`, `group`, and `charts` still "
        "score, but skip self-credit specifically (see transform_engine.py's `class_carryover_score` "
        "and classic.py's `chart_scores` docstrings)."
    )
    st.write(
        "First attempt at the terminal/group fix overcorrected: since a number can only ever belong to "
        "ONE terminal and ONE digital-root group (its own), excluding self-credit meant last week's own "
        "5 numbers could NEVER get terminal/group credit from any path -- a structural, not probabilistic, "
        "zero on two of the most heavily-weighted components. Re-measured: overlap with last week's own "
        "numbers dropped to exactly 0.000/5 (ranks consistently below median, not neutral) while genuinely "
        "unconnected numbers still had a real chance at credit. `ensemble.blend_scores()` now accepts an "
        "`exclude` map ({number: {component names}}) that renormalizes a number's remaining component "
        "weights instead of counting a structurally-inapplicable component as a literal (worst-possible, "
        "post-normalization) zero. Final measured result: last week's own numbers overlap the new top-5 "
        "at 0.245/5 and the actual next draw at 0.323/5 -- both within normal sampling noise of the "
        "0.278/5 chance level. Neither favored nor penalized, as it should be for a random game."
    )
    st.subheader("How to refresh")
    st.write("After running `python predictor.py update`, run `python train.py` to retrain the "
             "scikit-learn/deep models and refresh this dashboard's numbers.")
