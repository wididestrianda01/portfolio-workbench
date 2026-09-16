"""The table contract, expressed as a query over the snapshot's own files.

The panel is read by pandas, and this module states the same two operations the analytics depend on -
the as-of join and the coverage report - as SQL, then checks the two statements against each other on
the frozen snapshot. The point is not that a query is faster than a dataframe. It is that the contract
another project has to satisfy is a *table* contract, and a contract written only as pandas code is one
the sibling has to re-derive rather than satisfy: a query names the columns, the date arithmetic and
the availability rule in a form that can be read without reading Python.

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

from . import loader, manifest, panel

# The month a row describes and the moment it becomes readable, in the query's own arithmetic: a bar
# labelled with a month is available from the first day of the following month.
AVAILABILITY = (
    "date_trunc('month', \"date\") AS period_month, "
    "date_trunc('month', \"date\") + INTERVAL 1 MONTH AS available_from"
)

# The priced legs, which are the files the table contract is about. The factor archives are monthly
# already and are not part of the instrument-month table.
PRICED_ROLES = ("price", "fx")


def connection(root=None):
    """A DuckDB connection with one view per priced leg, and the snapshot's manifest beside it."""
    directory = loader.snapshot_root(root)
    document = manifest.read(directory)
    con = duckdb.connect()
    views = {}
    for position, entry in enumerate(document["files"]):
        if entry.get("role") not in PRICED_ROLES:
            continue
        # The leg's name is the file's position and not its path: a path carries separators and dots,
        # and a view name that has to be quoted is a name nobody can read back out of a log.
        name = f"leg_{position}"
        path = (directory / entry["path"]).as_posix()
        con.execute(
            f"CREATE VIEW {name} AS SELECT '{entry['instrument']}' AS instrument, "
            f"'{entry.get('currency', 'EUR')}' AS currency, * FROM read_csv_auto('{path}')"
        )
        views[name] = entry["instrument"]
    return con, document, views


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
    queried = {row["instrument"]: row for _, row in coverage(con, views, when=document["as_of"]).iterrows()}
    carried = panel.coverage_months(document["prices"])
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


def main(root=None):
    """Print the query's coverage beside the loader's, and whether the two statements agree."""
    document = loader.load_panel(root)
    con, _, views = connection(root)
    report = coverage(con, views, when=document["as_of"])
    joined = document["months"]
    print(f"[data] snapshot {document['snapshot_id']}, taken as of {document['as_of']}")
    print(f"[data] the as-of join in SQL: {len(as_of(con, views, document['as_of']))} instrument-days visible")
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
    return {"coverage": report, "agreement": checked, "document": document}


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
