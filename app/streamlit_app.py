"""Interactive dashboard for the pairs-trading study.

Run from the repository root::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pairs import (Rule, Settings, benjamini_hochberg, candidate_pairs, engle_granger, load_prices, metrics,  # noqa: E402
                   run_backtest, scan_pairs, sector_of, split_benchmark, trade_pair)
from pairs.plotting import (AQUA, BLUE, GRAY, NAVY, ORANGE, QUIET, RULE, YELLOW, plotly_layout)  # noqa: E402

st.set_page_config(page_title="Pairs Trading", page_icon="⇄", layout="wide")

# Share of unrelated random walks that the two-direction scan rejects at 5%
# (measured by simulation in scripts/make_figures.py).
SCAN_FALSE_POSITIVE_RATE = 0.091

# --------------------------------------------------------------------------
# Visual identity
# --------------------------------------------------------------------------

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,700&family=Hanken+Grotesk:wght@400;500;600&display=swap');

      html, body, .stApp, .stMarkdown, p, label, li, input, button, textarea {
        font-family: 'Hanken Grotesk', 'Segoe UI', system-ui, sans-serif;
      }
      .block-container {padding-top: 2.6rem; padding-bottom: 4rem; max-width: 1240px;}

      h1, h2, h3 {font-family: 'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif; color: #14213D; letter-spacing: -0.015em;}
      h1 {font-size: 2.9rem; font-weight: 700; line-height: 1.05; margin: 0 0 0.5rem 0; padding: 0;}
      h2 {font-size: 1.55rem; font-weight: 600; margin-top: 2.2rem;}
      h3 {font-size: 1.15rem; font-weight: 600;}

      /* Streamlit sets its own paragraph size, hence the explicit selectors. */
      .stMarkdown div.lede {font-size: 1.14rem; line-height: 1.55; color: #475467; max-width: 46rem; margin-bottom: 1.6rem;}
      .stMarkdown div.verdict {font-family: 'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif; font-size: 1.95rem;
                               font-weight: 600; line-height: 1.2; color: #14213D; max-width: 52rem;
                               margin: 2.4rem 0 1.2rem 0; letter-spacing: -0.012em;}
      .stMarkdown div.note {font-size: 1rem; line-height: 1.6; color: #475467; max-width: 46rem; margin-bottom: 1rem;}

      /* A strip of figures separated by rules: reads as one statement, not as a set of cards. */
      .figures {display: flex; flex-wrap: wrap; border-top: 2px solid #14213D; border-bottom: 1px solid #D9DEE7;
                margin: 1.1rem 0 1.8rem 0;}
      .figures > div {flex: 1 1 150px; padding: 0.9rem 1.2rem 0.95rem 0; margin-right: 1.2rem; border-right: 1px solid #D9DEE7;}
      .figures > div:last-child {border-right: none; margin-right: 0;}
      .figures .value {font-family: 'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif; font-size: 1.85rem; font-weight: 600;
                       color: #14213D; line-height: 1.1; font-variant-numeric: tabular-nums;}
      .figures .value.negative {color: #B54708;}
      .figures .label {font-size: 0.9rem; color: #475467; margin-top: 0.3rem; line-height: 1.3;}

      /* Charts sit on white panels; the opening chart gets a larger one. */
      div[data-testid="stPlotlyChart"] {background: #FFFFFF; border: 1px solid #D9DEE7; border-radius: 12px; overflow: hidden;}
      div[data-testid="stDataFrame"] {border: 1px solid #D9DEE7; border-radius: 10px; overflow: hidden;}

      section[data-testid="stSidebar"] .brand {font-family: 'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif; font-size: 1.5rem;
                                               font-weight: 700; color: #FFFFFF; line-height: 1.1; margin-bottom: 0.2rem;}
      section[data-testid="stSidebar"] .tagline {color: #A9B7D0; font-size: 0.92rem; line-height: 1.4; margin-bottom: 1.4rem;}
      section[data-testid="stSidebar"] hr {border-color: #2E4170;}
      section[data-testid="stSidebar"] div.universe {color: #A9B7D0; font-size: 0.9rem; line-height: 1.5;}
      :focus-visible {outline: 2px solid #2a78d6; outline-offset: 2px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def figures(items: list[tuple[str, str]]) -> None:
    """A horizontal strip of figures: (value, label)."""
    cells = ""
    for value, label in items:
        negative = " negative" if value.startswith(("-", "−")) else ""
        cells += f'<div><div class="value{negative}">{value}</div><div class="label">{label}</div></div>'
    st.markdown(f'<div class="figures">{cells}</div>', unsafe_allow_html=True)


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%".replace("-", "−")


# --------------------------------------------------------------------------
# Data and cached computations
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def get_prices() -> pd.DataFrame:
    prices, _ = split_benchmark(load_prices(ROOT / "data" / "prices.csv"))
    return prices


@st.cache_data(show_spinner="Testing every pair for cointegration…")
def get_scan(start: int, window: int) -> pd.DataFrame:
    prices = get_prices()
    return scan_pairs(np.log(prices.iloc[start:start + window]), candidate_pairs(prices.columns))


@st.cache_data(show_spinner="Running the walk-forward backtest…")
def get_backtest(settings: Settings):
    result = run_backtest(get_prices(), settings)
    return result.returns, result.pairs


prices = get_prices()
sectors = sector_of(prices.columns)
tickers = list(prices.columns)

with st.sidebar:
    st.markdown('<div class="brand">Pairs trading</div>'
                '<div class="tagline">Do stocks that moved together keep moving together?</div>', unsafe_allow_html=True)
    page = st.radio("Section", ["The verdict", "Find pairs", "Pair lab", "Backtest"], label_visibility="collapsed")
    st.divider()
    st.markdown(f'<div class="universe">{len(tickers)} US large caps in {sectors.nunique()} sectors, daily prices '
                f'from {prices.index[0]:%B %Y} to {prices.index[-1]:%B %Y}.</div>', unsafe_allow_html=True)


def pair_chart(y: str, x: str, start: int, formation: int, trading: int, title: str, height: int = 430) -> go.Figure:
    """Two normalised prices with the gap between them shaded: the spread made visible."""
    window = prices[[y, x]].iloc[start - formation:start + trading]
    normalised = window / window.iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normalised.index, y=normalised[x], name=x, line=dict(color=ORANGE, width=2)))
    fig.add_trace(go.Scatter(x=normalised.index, y=normalised[y], name=y, line=dict(color=BLUE, width=2),
                             fill="tonexty", fillcolor="rgba(20, 33, 61, 0.09)"))
    if trading:
        split = prices.index[start]
        fig.add_vline(x=split, line=dict(color=GRAY, dash="dot", width=1.2))
        fig.add_annotation(x=split, y=1.0, yref="paper", text="trading starts", showarrow=False, xanchor="left",
                           xshift=6, font=dict(size=12, color=QUIET))
    fig.update_layout(**plotly_layout(title, height=height, yaxis_title="Price, start = 1"))
    return fig


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------

if page == "The verdict":
    st.markdown("<h1>Two prices,<br>one bet on the gap</h1>", unsafe_allow_html=True)
    st.markdown('<div class="lede">Visa and Mastercard run the same business, and their share prices rarely stray far from '
                'each other. Pairs trading bets that when the gap opens, it will close again. This study tests that '
                'idea on 104 US stocks over fifteen years, with pairs chosen from past data only.</div>',
                unsafe_allow_html=True)

    start = int(prices.index.get_loc(pd.Timestamp("2022-01-06")))
    show(pair_chart("V", "MA", start, 504, 126, "Visa and Mastercard, with the gap between them shaded"))

    base = Settings()
    names = {"Distance": dict(selection="distance"), "Cointegration": dict(selection="cointegration"),
             "Cointegration + Kalman": dict(selection="cointegration", method="kalman"),
             "Random pairs": dict(selection="random")}
    results = {name: get_backtest(replace(base, **options))[0] for name, options in names.items()}
    sharpe = {name: metrics.sharpe_ratio(r) for name, r in results.items()}
    annual = {name: metrics.annualised_return(r) for name, r in results.items()}

    st.markdown(f'<div class="verdict">Pairs picked by a cointegration test earned {pct(annual["Cointegration"])} a year '
                f'after costs: nothing. Pairs picked for simply having stayed close earned {pct(annual["Distance"])}, '
                'and only for some settings.</div>', unsafe_allow_html=True)
    figures([
        (f"{sharpe['Cointegration']:.2f}".replace("-", "−"), "Sharpe ratio, cointegration pairs"),
        (f"{sharpe['Cointegration + Kalman']:.2f}".replace("-", "−"), "with a Kalman-filter hedge ratio"),
        (f"{sharpe['Distance']:.2f}".replace("-", "−"), "distance pairs, base settings"),
        (f"{sharpe['Random pairs']:.2f}".replace("-", "−"), "pairs drawn at random"),
        (f"± {metrics.sharpe_standard_error(results['Cointegration']):.2f}", "statistical uncertainty on each"),
    ])

    colors = {"Distance": BLUE, "Cointegration": ORANGE, "Cointegration + Kalman": AQUA, "Random pairs": GRAY}
    fig = go.Figure()
    for name, series in results.items():
        fig.add_trace(go.Scatter(x=series.index, y=100 * series.cumsum(), name=name,
                                 line=dict(color=colors[name], width=2, dash="dot" if name == "Random pairs" else "solid"),
                                 hovertemplate="%{y:.1f}%"))
    fig.add_hline(y=0, line=dict(color=RULE, width=1))
    fig.update_layout(**plotly_layout("Cumulative return on committed capital, after 5 bp costs",
                                      yaxis_title="Percent", height=420))
    show(fig)

    st.markdown("## Why the statistical test did not help")
    st.markdown('<div class="note">Each two-year window tests 567 same-sector pairs. About 52 would pass a 5% test by '
                'chance alone, and between 44 and 102 do. After correcting for the number of tests, almost none '
                'remain. Pairs that pass in one window are no more likely than any other pair to pass in the next: '
                'the test mostly finds coincidences. <b>Find pairs</b> shows this window by window, <b>Pair lab</b> '
                'lets you try any two stocks, and <b>Backtest</b> lets you change every setting.</div>',
                unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Find pairs
# --------------------------------------------------------------------------

elif page == "Find pairs":
    st.markdown("<h1>Find pairs</h1>", unsafe_allow_html=True)
    st.markdown('<div class="lede">An Engle-Granger test on every pair of stocks from the same sector, over a two-year '
                'window. The question is how many of the pairs it flags are real.</div>', unsafe_allow_html=True)

    window = 504
    starts = list(range(0, len(prices) - window + 1, window))
    labels = [f"{prices.index[s]:%b %Y} to {prices.index[s + window - 1]:%b %Y}" for s in starts]
    choice = st.select_slider("Window", options=labels, value=labels[-2])
    k = labels.index(choice)
    scan = get_scan(starts[k], window)

    significant = scan["p_value"] <= 0.05
    corrected = benjamini_hochberg(scan["p_value"])
    items = [
        (f"{len(scan)}", "pairs tested"),
        (f"{int(significant.sum())}", "pass at the 5% level"),
        (f"{SCAN_FALSE_POSITIVE_RATE * len(scan):.0f}", "would pass by chance alone"),
        (f"{int(corrected.sum())}", "remain after correcting for the number of tests"),
    ]
    later = None
    if k + 1 < len(starts):
        following = np.log(prices.iloc[starts[k + 1]:starts[k + 1] + window])
        later = np.array([engle_granger(following[y].to_numpy(), following[x].to_numpy()).p_value
                          for y, x in zip(scan["y"], scan["x"])])
        items.append((pct((later[significant.to_numpy()] <= 0.05).mean(), 0), "of those still pass in the next window"))
        items.append((pct((later <= 0.05).mean(), 0), "of all pairs pass in the next window"))
    figures(items)

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=scan["p_value"], xbins=dict(start=0, end=1, size=0.05), name="Observed",
                                   marker=dict(color=BLUE, line=dict(color="white", width=1))))
        fig.update_layout(**plotly_layout("Distribution of the p-values", hovermode="closest", bargap=0.04,
                                          xaxis_title="p-value of the cointegration test", yaxis_title="Number of pairs"))
        show(fig)
        st.caption("Each p-value is the smaller of two tests (one per direction of the regression), so they pile up "
                   "near zero even when nothing ties the stocks together. On unrelated random walks, 9% of pairs pass.")
    with right:
        if later is not None:
            fig = go.Figure(go.Scatter(
                x=scan["p_value"], y=later, mode="markers", text=scan["y"] + " / " + scan["x"],
                marker=dict(color=BLUE, size=6, opacity=0.55, line=dict(color="white", width=0.5)),
                hovertemplate="<b>%{text}</b><br>this window %{x:.3f}<br>next window %{y:.3f}<extra></extra>"))
            fig.update_layout(**plotly_layout("This window against the next", hovermode="closest",
                                              xaxis_title="p-value in this window", yaxis_title="p-value in the next window"))
            show(fig)
            st.caption("If cointegration persisted, the points would gather in the bottom-left corner.")
        else:
            st.info("This is the last complete window: there is no following window to compare with.")

    st.markdown("## The most significant pairs")
    table = scan.head(25).copy()
    table.insert(0, "Pair", table["y"] + " / " + table["x"])
    table.insert(1, "Sector", sectors[table["y"]].to_numpy())
    table["Survives correction"] = corrected[:25]
    if later is not None:
        table["p-value next window"] = later[:25]
    st.dataframe(
        table.drop(columns=["y", "x", "alpha", "t_stat"]).rename(columns={
            "beta": "Hedge ratio", "p_value": "p-value", "half_life": "Half-life (days)", "spread_std": "Spread std"}),
        hide_index=True, width="stretch",
        column_config={"Hedge ratio": st.column_config.NumberColumn(format="%.2f"),
                       "p-value": st.column_config.NumberColumn(format="%.4f"),
                       "Half-life (days)": st.column_config.NumberColumn(format="%.0f"),
                       "Spread std": st.column_config.NumberColumn(format="%.3f"),
                       "p-value next window": st.column_config.NumberColumn(format="%.3f")},
    )

# --------------------------------------------------------------------------
# Pair lab
# --------------------------------------------------------------------------

elif page == "Pair lab":
    st.markdown("<h1>Pair lab</h1>", unsafe_allow_html=True)
    st.markdown('<div class="lede">Pick two stocks and a date. The relation is estimated on the two years before the '
                'date and traded during the six months after it.</div>', unsafe_allow_html=True)

    one, two, three, four = st.columns([1, 1, 1.4, 1.2])
    y = one.selectbox("First stock", tickers, index=tickers.index("V"))
    same_sector = [t for t in tickers if sectors[t] == sectors[y] and t != y]
    others = same_sector + [t for t in tickers if t not in same_sector and t != y]
    x = two.selectbox("Second stock", others, index=others.index("MA") if "MA" in others else 0,
                      help="Stocks of the same sector are listed first.")
    formation, trading = 504, 126
    dates = list(range(formation, len(prices) - 1, trading))
    date_labels = [f"{prices.index[d]:%b %Y}" for d in dates]
    chosen_date = three.select_slider("Trading starts", options=date_labels, value="Jan 2022" if "Jan 2022" in date_labels else date_labels[-3])
    method = four.selectbox("Hedge ratio", ["Fixed (regression)", "Adaptive (Kalman filter)", "One for one"])

    a, b, c, d = st.columns(4)
    entry = a.slider("Enter beyond (std)", 1.0, 3.0, 2.0, 0.25)
    exit_level = b.slider("Exit within (std)", 0.0, 0.9, 0.5, 0.1)
    stop = c.slider("Give up beyond (std)", 3.25, 6.0, 4.0, 0.25)
    cost = d.slider("Cost (basis points)", 0, 30, 5)

    start = dates[date_labels.index(chosen_date)]
    formation_prices = prices[[y, x]].iloc[start - formation:start]
    trading_prices = prices[[y, x]].iloc[start - 1:start + trading]
    test = engle_granger(np.log(formation_prices[y].to_numpy()), np.log(formation_prices[x].to_numpy()))
    trade = trade_pair(formation_prices, trading_prices, "kalman" if method.startswith("Adaptive") else "static",
                       Rule(entry, exit_level, stop), cost, unit_hedge=method == "One for one")
    earned = trade.returns.iloc[1:]

    figures([
        (f"{test.p_value:.3f}", "p-value of the cointegration test on the two years before"),
        (f"{test.half_life:.0f} days" if np.isfinite(test.half_life) else "none", "half-life of the spread"),
        (f"{test.beta:.2f}", f"hedge ratio: {x} per unit of {y}"),
        (f"{trade.n_trades}", "trades in the six months"),
        (pct(earned.sum(), 2), "return on capital, after costs"),
    ])

    show(pair_chart(y, x, start, formation, trading, f"{y} and {x}", height=380))

    z = trade.z_score.iloc[1:]
    held = trade.position.iloc[:-1]
    fig = go.Figure()
    for sign, color, label in [(1.0, "rgba(27,175,122,0.20)", f"Long {y}, short {x}"),
                               (-1.0, "rgba(237,161,0,0.24)", f"Short {y}, long {x}")]:
        mask = (held == sign).to_numpy()
        edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
        for begin, finish in zip(edges[::2], edges[1::2]):
            fig.add_vrect(x0=held.index[begin], x1=held.index[min(finish, len(held) - 1)], fillcolor=color, line_width=0)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                 marker=dict(color=color.replace("0.20", "0.9").replace("0.24", "0.9"), size=11, symbol="square")))
    fig.add_trace(go.Scatter(x=z.index, y=z, name="Spread", line=dict(color=NAVY, width=1.8), hovertemplate="%{y:.2f} std"))
    for level, dash in [(entry, "dash"), (-entry, "dash"), (exit_level, "dot"), (-exit_level, "dot")]:
        fig.add_hline(y=level, line=dict(color=GRAY, dash=dash, width=1))
    fig.update_layout(**plotly_layout("The spread during the trading period, and the positions taken",
                                      yaxis_title="Standard deviations", height=360))
    show(fig)

    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Scatter(x=earned.index, y=100 * earned.cumsum(), line=dict(color=BLUE, width=2),
                                   name="Profit", hovertemplate="%{y:.2f}%"))
        fig.add_hline(y=0, line=dict(color=RULE, width=1))
        fig.update_layout(**plotly_layout("Cumulative return", yaxis_title="Percent of capital", height=320, showlegend=False))
        show(fig)
    with right:
        adaptive = trade_pair(formation_prices, trading_prices, "kalman", Rule(entry, exit_level, stop), cost)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=adaptive.hedge_ratio.index, y=adaptive.hedge_ratio, name="Kalman filter",
                                 line=dict(color=AQUA, width=2), hovertemplate="%{y:.3f}"))
        fig.add_trace(go.Scatter(x=adaptive.hedge_ratio.index, y=np.full(len(adaptive.hedge_ratio), test.beta),
                                 name="Regression on the two years before", line=dict(color=GRAY, width=1.6, dash="dash"),
                                 hovertemplate="%{y:.3f}"))
        fig.update_layout(**plotly_layout("Hedge ratio", height=320))
        show(fig)
    st.caption("A position opened at the close of one day earns the returns of the following days: no signal trades "
               "on the price that produced it.")

# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------

else:
    st.markdown("<h1>Backtest</h1>", unsafe_allow_html=True)
    st.markdown('<div class="lede">Every six months, pairs are chosen from the two years before and traded with frozen '
                'parameters. Change a setting and see whether the result survives.</div>', unsafe_allow_html=True)

    one, two, three, four = st.columns(4)
    selection = one.selectbox("How pairs are chosen", ["Cointegration test", "Closest prices (distance)", "At random"])
    kalman = two.selectbox("Hedge ratio", ["Fixed", "Adaptive (Kalman filter)"],
                           disabled=selection != "Cointegration test") == "Adaptive (Kalman filter)"
    n_pairs = three.select_slider("Pairs traded", [5, 10, 20, 40], value=20)
    cost = four.slider("Cost (basis points)", 0, 30, 5)
    a, b, c, d = st.columns(4)
    formation = a.select_slider("Formation window (days)", [252, 504, 756], value=504)
    trading = b.select_slider("Trading period (days)", [63, 126], value=126)
    entry = c.slider("Enter beyond (std)", 1.0, 3.0, 2.0, 0.5)
    stop = d.select_slider("Give up beyond (std)", [3.5, 4.0, 5.0, 50.0], value=4.0,
                           format_func=lambda v: "never" if v >= 50 else f"{v:g}")

    key = {"Cointegration test": "cointegration", "Closest prices (distance)": "distance", "At random": "random"}[selection]
    settings = Settings(formation_days=formation, trading_days=trading, n_pairs=n_pairs, selection=key,
                        method="kalman" if kalman and key == "cointegration" else "static",
                        rule=Rule(entry=entry, stop=stop), cost_bps=float(cost))
    returns, traded = get_backtest(settings)
    placebo, _ = get_backtest(replace(settings, selection="random", method="static"))

    ratio, error = metrics.sharpe_ratio(returns), metrics.sharpe_standard_error(returns)
    figures([
        (pct(metrics.annualised_return(returns), 2), "return per year on committed capital"),
        (pct(metrics.annualised_volatility(returns)), "volatility"),
        (f"{ratio:.2f}".replace("-", "−"), f"Sharpe ratio, give or take {error:.2f}"),
        (pct(metrics.max_drawdown(returns)), "largest drawdown"),
        (pct((traded["profit"] > 0).mean(), 0), "of pair-periods end with a profit"),
    ])
    if abs(ratio) < 2 * error:
        st.markdown('<div class="note">The Sharpe ratio is within two standard errors of zero: with these settings the '
                    'data cannot tell this strategy from one that earns nothing.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="note">The Sharpe ratio is more than two standard errors from zero. Before trusting it, '
                    'change the formation window and the number of pairs: a real effect should not depend on them.</div>',
                    unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=placebo.index, y=100 * placebo.cumsum(), name="Pairs drawn at random, same rule",
                             line=dict(color=GRAY, width=1.6, dash="dot"), hovertemplate="%{y:.1f}%"))
    fig.add_trace(go.Scatter(x=returns.index, y=100 * returns.cumsum(), name=selection,
                             line=dict(color=BLUE, width=2.2), hovertemplate="%{y:.1f}%"))
    fig.add_hline(y=0, line=dict(color=RULE, width=1))
    fig.update_layout(**plotly_layout("Cumulative return after costs", yaxis_title="Percent of capital", height=400))
    show(fig)

    left, right = st.columns([3, 2])
    with left:
        yearly = 100 * returns.groupby(returns.index.year).sum()
        fig = go.Figure(go.Bar(x=yearly.index, y=yearly.values,
                               marker_color=[BLUE if v >= 0 else ORANGE for v in yearly.values],
                               hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
        fig.update_layout(**plotly_layout("Return by calendar year", yaxis_title="Percent", hovermode="closest", height=340))
        show(fig)
    with right:
        st.markdown("### Pairs chosen most often")
        frequent = (traded.assign(Pair=traded["y"] + " / " + traded["x"]).groupby("Pair")
                    .agg(Periods=("profit", "size"), Profit=("profit", "sum"), Trades=("n_trades", "sum"))
                    .sort_values("Periods", ascending=False).head(10).reset_index())
        st.dataframe(frequent, hide_index=True, width="stretch", height=318,
                     column_config={"Profit": st.column_config.NumberColumn("Total return", format="percent")})
