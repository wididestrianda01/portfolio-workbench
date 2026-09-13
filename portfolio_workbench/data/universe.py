"""The frozen universe, the mandate constants and the panel window.

Nothing here is derived at run time: every value was fixed before the build began.
Insertion order is load-bearing. `TICKERS` is built from `SLEEVES`, and every weight
vector in the package is indexed by `TICKERS`, so reordering the sleeve map would
silently misalign weights against returns without raising anything.
"""

# One line per instrument: the sleeve it carries and the currency its quote is in.
# A line survives only if it carries exposure the panel cannot separate otherwise.
# Dropped on that rule: a second World tracker, near-collinear with the first; a
# USD-duration line inside a EUR mandate; a real-estate duplicate on a pence-quoted
# ex-UK index; and the emerging-market line, despite the sleeve being wanted, because
# its adjusted close is dividend-blind and therefore not a total-return series. The
# emerging-market sleeve is absent by decision, not by oversight.
SLEEVES = {
    "IWDA.AS": ("equity_dev", "EUR"),
    "IMEU.AS": ("equity_eu", "EUR"),
    "XACT-NORDEN.ST": ("equity_nordic", "SEK"),
    "IBGL.AS": ("gov_long", "EUR"),
    "IEGE.AS": ("gov_short", "EUR"),
    "IEAC.AS": ("credit_ig", "EUR"),
    "IHYG.L": ("credit_hy", "EUR"),
    "IBCI.AS": ("inflation_linked", "EUR"),
    "4GLD.DE": ("gold", "EUR"),
    "IWDP.AS": ("real_estate", "EUR"),
    "XEON.DE": ("cash", "EUR"),
}

GROUP = {
    "equity_dev": "equity",
    "equity_eu": "equity",
    "equity_nordic": "equity",
    "gov_long": "government",
    "gov_short": "government",
    "inflation_linked": "government",
    "credit_ig": "credit",
    "credit_hy": "credit",
    "gold": "real_and_cash",
    "real_estate": "real_and_cash",
    "cash": "real_and_cash",
}

FX_TICKERS = ("EURUSD=X", "EURSEK=X")

TICKERS = list(SLEEVES)
GROUPS = [GROUP[SLEEVES[t][0]] for t in TICKERS]

# The strategic policy weights, summing to one. They define the benchmark the whole
# comparison is measured against, and they are fixed rather than optimised: a
# benchmark that moved with the data would make every tracking error a moving target.
POLICY_WEIGHTS = {
    "IWDA.AS": 0.30,
    "IMEU.AS": 0.08,
    "XACT-NORDEN.ST": 0.04,
    "IBGL.AS": 0.12,
    "IEGE.AS": 0.05,
    "IEAC.AS": 0.13,
    "IHYG.L": 0.04,
    "IBCI.AS": 0.05,
    "4GLD.DE": 0.05,
    "IWDP.AS": 0.04,
    "XEON.DE": 0.10,
}

# The declared risk budget, in share of portfolio volatility per group. Declared
# before construction so that "consumed by design or by accident" is measurable.
RISK_BUDGET = {"equity": 0.55, "government": 0.20, "credit": 0.15, "real_and_cash": 0.10}

# The price window is the deepest the feed allows for this set; the joined panel
# stops one month earlier because the factor library's newest month is the month
# before the last completed price month.
WINDOW_START, WINDOW_END = "2010-09", "2026-08"
PANEL_START, PANEL_END = "2010-09", "2026-07"

# The fixed constraint set, identical in every cell so that a difference is
# attributable to the method rather than to a constraint.
CAP = 0.35             # per-sleeve cap
BAND = 0.01            # no-trade band, absolute weight move, per sleeve
TURNOVER_CAP = 0.05    # one-way turnover per measured rebalance
COST_BP = 10.0         # per side, charged on traded notional
EST_MONTHS = 60        # rolling estimation window
COST_SENSITIVITY_BP = (5.0, 20.0, 40.0)

SEED = 20260912
