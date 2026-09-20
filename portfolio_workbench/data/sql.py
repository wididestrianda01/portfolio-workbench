"""The table contract, expressed as a query over the snapshot's own files.

The panel is read by pandas, and this module states the same operations the analytics depend on (the
as-of join, the coverage report, and the assembly of the monthly euro excess return table) as SQL,
then checks the statements against each other on the frozen snapshot. The point is not that a query is
faster than a dataframe. It is that the contract another project has to satisfy is a *table* contract,
and a contract written only as pandas code is one the sibling has to re-derive rather than satisfy: a
query names the columns, the date arithmetic, the three-source join and the availability rule in a form
that can be read without reading Python.

**The assembly is the part worth stating this way.** A reader told the package works on a monthly euro
excess return table has, in the pandas path, to follow the currency translation, the return of holding
that currency, the accrual of an annualised overnight rate over a month's own calendar days and the
availability gate through four modules to find out what the table is. Here it is one statement: the
price legs and the currency legs reduced to month-on-month ratios, the cash leg compounded within the
month at /360 on the calendar days the publisher's unit implies, the translation as
`(1 + r) / (1 + fx) - 1` rather than a sum, and every leg gated by the same rule.

**The availability rule is stated in SQL here, not imported from pandas.** `period_month` is the
month a bar describes and `available_from` is the first instant it may be used, and expressing that
arithmetic a second time is the point of the check: if the two statements ever disagree, the check
fails, and a disagreement between them is exactly the look-ahead class of error the rule exists to
prevent. What is *not* duplicated is the rule's definition as a number: the pandas path stays the
authority the analytics run on, and this module's queries are asserted against it rather than adopted
by it.

Nothing in the analytics imports this module. It is a second statement of the data contract for a
reader and for the sibling project, not a second path the results travel down.
"""

import duckdb
import pandas as pd

from . import external, loader, manifest, panel, universe

# The month a row describes and the moment it becomes readable, in the query's own arithmetic: a bar
# labelled with a month is available from the first day of the following month.
AVAILABILITY = (
    "date_trunc('month', \"date\") AS period_month, "
    "date_trunc('month', \"date\") + INTERVAL 1 MONTH AS available_from"
)

# The priced legs, which are the files the table contract is about. The factor archives are monthly
# already and are not part of the instrument-month table.
PRICED_ROLES = ("price", "fx")

# The two overnight legs, under view names of their own rather than the position-derived ones the
# priced legs get: they are one series spliced at a date the plan fixed, so a name carrying their
# role is what makes the splice readable in the query below.
CASH_VIEWS = {"ecb_eonia": "cash_eonia", "ecb_estr": "cash_estr"}

# The euro excess return table, as one statement. Every leg is gated by the availability rule, the
# window is the declared panel window, and the cash leg is compounded within the month on calendar
# days at the annualised quote over the day count the publisher's unit implies.
RETURNS = """
WITH rate AS (
    SELECT TIME_PERIOD::DATE AS "date", OBS_VALUE::DOUBLE AS percent FROM {eonia}
    WHERE TIME_PERIOD < TIMESTAMP '{transition}'
    UNION ALL
    SELECT TIME_PERIOD::DATE AS "date", OBS_VALUE::DOUBLE AS percent FROM {estr}
    WHERE TIME_PERIOD >= TIMESTAMP '{transition}'
),
cash AS (
    SELECT date_trunc('month', "date") AS period_month,
           product(1.0 + percent / 100.0 / {days}) - 1.0 AS cash_return
    FROM rate
    GROUP BY 1
    HAVING date_trunc('month', "date") + INTERVAL 1 MONTH <= TIMESTAMP '{when}'
),
priced AS (
    SELECT instrument, currency, date_trunc('month', "date") AS period_month, adj_close
    FROM ({prices})
    WHERE {window}
),
local AS (
    SELECT instrument, currency, period_month,
           adj_close / lag(adj_close) OVER (PARTITION BY instrument ORDER BY period_month) - 1.0
               AS local_return
    FROM priced
),
quoted AS (
    SELECT substr(instrument, 4, 3) AS currency,
           date_trunc('month', "date") AS period_month,
           close / lag(close) OVER (PARTITION BY instrument ORDER BY period_month) - 1.0 AS fx_return
    FROM ({fx})
    WHERE {window}
),
joined AS (
    SELECT l.instrument, l.currency, l.period_month, l.local_return, q.fx_return, c.cash_return
    FROM local l
    LEFT JOIN quoted q ON q.currency = l.currency AND q.period_month = l.period_month
    LEFT JOIN cash c ON c.period_month = l.period_month
)
SELECT instrument, currency, period_month, local_return, fx_return, cash_return,
       CASE WHEN currency = 'EUR' THEN local_return
            ELSE (1.0 + local_return) / (1.0 + fx_return) - 1.0 END - cash_return AS excess_return
FROM joined
WHERE local_return IS NOT NULL AND cash_return IS NOT NULL
ORDER BY instrument, period_month
"""

# The gate every leg of the assembly carries, and the window the declared panel runs over. Stated once
# because a leg gated differently from its neighbours is a table whose rows mean two things.
WINDOW = (
    "date_trunc('month', \"date\") + INTERVAL 1 MONTH <= TIMESTAMP '{when}' "
    "AND date_trunc('month', \"date\") BETWEEN DATE '{start}-01' AND DATE '{end}-01'"
)

# How far the two paths may differ on a monthly return. The cash leg compounds the same daily factors
# in a different order (a product over the month's rows against pandas' pairwise product), so the
# agreement is a tolerance rather than an identity: a monthly return is of order 1e-2 and this is
# eight orders below it, which is not a rounding difference but a different arithmetic.
RETURN_TOLERANCE = 1e-10


def connection(root=None):
    """A DuckDB connection with one view per leg, and the snapshot's manifest beside it."""
    directory = loader.snapshot_root(root)
    snapshot = manifest.read(directory)
    con = duckdb.connect()
    views = {}
    for position, entry in enumerate(snapshot["files"]):
        role = entry.get("role")
        path = (directory / entry["path"]).as_posix()
        if role in CASH_VIEWS:
            con.execute(f"CREATE VIEW {CASH_VIEWS[role]} AS SELECT * FROM read_csv_auto('{path}')")
            continue
        if role not in PRICED_ROLES:
            continue
        # The leg's name is its role and the file's position, not its path: a path carries separators
        # and dots, and a view name that has to be quoted is a name nobody can read back out of a log.
        # The role is in the name because the assembly below reads the price legs and the currency
        # legs separately, and the table contract reads them together.
        name = f"{role}_{position}"
        con.execute(
            f"CREATE VIEW {name} AS SELECT '{entry['instrument']}' AS instrument, "
            f"'{entry.get('currency', 'EUR')}' AS currency, * FROM read_csv_auto('{path}')"
        )
        views[name] = entry["instrument"]
    absent = sorted(role for role in CASH_VIEWS if role not in {entry.get("role") for entry in snapshot["files"]})
    if absent:
        raise ValueError(f"the snapshot carries no {' or '.join(absent)} leg, so the cash rate cannot be accrued")
    return con, snapshot, views


def as_of(con, views, when):
    """The rows a reader standing at `when` may use, one row per instrument-day, as a query.

    The gate is `available_from <= when` and nothing else: a bar whose month is `when`'s own month is
    excluded, which is the difference between a monthly label and a monthly bar.
    """
    parts = [
        f"SELECT instrument, currency, {AVAILABILITY} FROM {name}" for name in sorted(views)
    ]
    query = f"WITH long AS ({' UNION ALL '.join(parts)}) SELECT * FROM long WHERE available_from <= TIMESTAMP '{when}'"
    return con.execute(query).fetch_df()


def coverage(con, views, when):
    """First and last month per instrument, the months carried, and the months visible at `when`.

    The last column is the one the as-of rule changes and the one a coverage report without it would
    overstate: a panel that runs to the snapshot's own month shows a bar for that month whether or not
    the decision it feeds could have seen it.
    """
    parts = [f"SELECT instrument, {AVAILABILITY} FROM {name}" for name in sorted(views)]
    carried = ", ".join(f"'{instrument}'" for instrument in sorted(set(views.values())))
    query = f"""
        WITH long AS ({' UNION ALL '.join(parts)}),
             months AS (SELECT DISTINCT instrument, period_month FROM long)
        SELECT instrument,
               CAST(min(period_month) AS DATE) AS first_month,
               CAST(max(period_month) AS DATE) AS last_month,
               count(*) AS months,
               count(*) FILTER (WHERE period_month + INTERVAL 1 MONTH <= TIMESTAMP '{when}') AS months_visible
        FROM months
        WHERE instrument IN ({carried})
        GROUP BY instrument
        ORDER BY instrument
    """
    return con.execute(query).fetch_df()


def agreement(document, report, root=None):
    """Whether the query's coverage and the pandas path's agree, per instrument and in total.

    Compared on the months carried and the first and last month, which is what both statements claim to
    describe. A difference in either is refused rather than reported as a note: two statements of one
    contract that disagree mean one of them is wrong about the panel every result is keyed to.
    """
    con, _, views = connection(root)
    queried = {row["instrument"]: row for _, row in coverage(con, views, when=document.as_of).iterrows()}
    carried = panel.coverage_months(document.prices)
    differences = []
    for instrument, block in carried.items():
        row = queried.get(instrument)
        if row is None:
            differences.append(f"{instrument}: the query does not cover it")
            continue
        if str(row["first_month"])[:7] != block["first"] or str(row["last_month"])[:7] != block["last"]:
            differences.append(
                f"{instrument}: query {str(row['first_month'])[:7]}..{str(row['last_month'])[:7]} against "
                f"the loader's {block['first']}..{block['last']}"
            )
        elif int(row["months"]) != block["months"]:
            differences.append(f"{instrument}: query carries {row['months']} months against {block['months']}")
    return {
        "agrees": not differences,
        "differences": differences,
        "instruments": len(carried),
        "months": sum(block["months"] for block in carried.values()),
    }


def returns(con, views, document, when=None):
    """The monthly euro excess return table, as one query over the snapshot's own files.

    The same frame `panel.eur_excess_returns` returns, assembled here from the files: the price and
    currency legs reduced to month-on-month ratios, the currency translation applied as the
    reciprocal multiplication the decomposition uses, and the accrued overnight rate subtracted, so
    the left-hand side of the factor model is an excess return in the statement as well as in the
    dataframe.
    """
    moment = document.as_of if when is None else when

    def legs(role):
        names = [name for name in sorted(views) if name.startswith(f"{role}_")]
        if not names:
            raise ValueError(f"the snapshot carries no {role} leg to assemble the panel from")
        return " UNION ALL ".join(f"SELECT * FROM {name}" for name in names)

    query = RETURNS.format(
        eonia=CASH_VIEWS["ecb_eonia"],
        estr=CASH_VIEWS["ecb_estr"],
        transition=external.RF_TRANSITION,
        days=external.ACCRUAL_DAYS,
        prices=legs("price"),
        fx=legs("fx"),
        when=moment,
        window=WINDOW.format(when=moment, start=universe.PANEL_START, end=universe.WINDOW_END),
    )
    return con.execute(query).fetch_df()


def returns_agreement(document, queried):
    """Whether the query's euro excess returns and the pandas path's agree, month by month.

    Compared cell by cell over the months both statements carry, and on the months themselves: a
    statement that agrees on the overlap while covering a different span is not the same table. The
    tolerance is loose by the standards of the identity checks elsewhere in the package because the
    two paths compound a month's rate in a different order, and a difference above it is not
    rounding.
    """
    frame = document.returns
    wide = queried.assign(month=pd.PeriodIndex(queried["period_month"], freq="M")).pivot(
        index="month", columns="instrument", values="excess_return"
    )
    differences = []
    missing = [str(month) for month in frame.index.difference(wide.index)]
    extra = [str(month) for month in wide.index.difference(frame.index)]
    if missing or extra:
        differences.append(
            f"months only the pandas path carries: {missing[:4] or 'none'}; months only the query "
            f"carries: {extra[:4] or 'none'}"
        )
    months = frame.index.intersection(wide.index)
    columns = frame.columns.intersection(wide.columns)
    worst = float(
        (frame.reindex(index=months, columns=columns) - wide.reindex(index=months, columns=columns))
        .abs()
        .max()
        .max()
    )
    if worst > RETURN_TOLERANCE:
        differences.append(f"the largest difference is {worst:.3e}, above the {RETURN_TOLERANCE:g} tolerance")
    return {
        "agrees": not differences,
        "differences": differences,
        "months": len(months),
        "instruments": len(columns),
        "worst": worst,
    }


def main(root=None):
    """Print the query's coverage beside the loader's, and whether the two statements agree."""
    document = loader.load_panel(root)
    con, _, views = connection(root)
    report = coverage(con, views, when=document.as_of)
    joined = document.months
    print(f"[data] snapshot {document.snapshot_id}, taken as of {document.as_of}")
    print(f"[data] the as-of join in SQL: {len(as_of(con, views, document.as_of))} instrument-days visible")
    print(f"[data] {'instrument':<18s}{'first':>9s}{'last':>9s}{'months':>8s}{'visible':>9s}")
    for _, row in report.iterrows():
        print(
            f"[data] {row['instrument']:<18s}{str(row['first_month'])[:7]:>9s}{str(row['last_month'])[:7]:>9s}"
            f"{int(row['months']):>8d}{int(row['months_visible']):>9d}"
        )
    # The same report read a month before the snapshot was taken: the gate is what removes the bars a
    # decision at that moment could not have seen, and a coverage report without it would show them.
    earlier = coverage(con, views, when=joined.max().to_timestamp())
    hidden = int(earlier["months"].sum() - earlier["months_visible"].sum())
    print(
        f"[data] read at {joined.max()} instead of the snapshot date, the same query hides {hidden} of "
        f"{int(earlier['months'].sum())} instrument-months: a bar is readable only from the month after "
        "the month it describes"
    )
    checked = agreement(document, report, root)
    statement = (
        "the query and the pandas path state one contract"
        if checked["agrees"]
        else "; ".join(checked["differences"])
    )
    print(
        f"[data] the query's coverage against the loader's over {checked['instruments']} instruments and "
        f"{checked['months']} instrument-months: {statement}"
    )
    queried = returns(con, views, document)
    assembled = returns_agreement(document, queried)
    assembly = (
        f"the largest month-instrument difference is {assembled['worst']:.2e}"
        if assembled["agrees"]
        else "; ".join(assembled["differences"])
    )
    print(
        f"[data] the euro excess return table assembled in SQL against the pandas path over "
        f"{assembled['instruments']} instruments and {assembled['months']} months: {assembly}"
    )
    return {"coverage": report, "agreement": checked, "returns": assembled, "document": document}


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
