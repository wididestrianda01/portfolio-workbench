"""One engine decides, for every month, which data was available and which model was fitted.

The boundary the whole comparison rests on is one inequality: an estimate formed at the close of a
month cannot read the next month's bar. It is a property of the calendar the windows are cut on, and
it is invisible in a weight path - a cell that looked one month ahead returns plausible weights and a
flattering return series, and nothing in the row would say so. So the engine keeps a **record** rather
than an intention. Every step carries the month traded, the window estimated on, the moment that
window's last bar became readable and which bar set that moment; the run is then checked against the
record, rather than each step being trusted as it goes.

Four things here are decisions rather than mechanics.

**The window ends the month before the month it trades, and nothing else will do.** A window that
stopped a month earlier would be conservative rather than wrong, and it would silently shorten every
estimate in the package; a window that reached into the traded month is the leakage the design exists
to prevent. The engine asserts equality, so neither can happen quietly.

**The estimate is the window's own, the count included.** The number of components a factor
covariance retains is decided inside the trailing window, so the engine carries the count beside the
step it belongs to: the count a step reports is the one its own window decided, and the row says which
window that was. A model fitted on a longer window than the step trades behind would be leakage
whether or not its count differed.

**The availability that gated a step is the traded month's own first day.** A monthly bar labelled
with the month's first day carries that month's last close, so it is readable from the first day of
the following month; the window's last bar therefore becomes readable exactly as the traded month
opens. The rule is written once in the data layer and read here, and the record states the moment it
produced rather than the argument for it.

**The record is what the boundary is checked against, and it is checked on the whole run.** The
assertion is a function over the steps rather than a flag inside the loop, which is what lets a
planted break be caught: a window that reaches its own traded month, a window that stops short of the
last bar the month could have read, an availability that does not follow from the bar it names, a
window that is not the contiguous run it claims, or two counts on one window each fail here without a
weight path having been built first.
"""

import pandas as pd

from ..data import loader, panel
from ..factors import exposures

# The moment the estimate is formed and the model refitted: monthly, on the panel's own calendar.
# Every step of every protocol carries it, because a cell whose refit frequency differed would be a
# second axis moving beside the one under test.
REFIT = "monthly"

# The fields a step record carries, in the order the report prints them. `window` holds the window's
# months themselves and travels beside the record rather than in it, because a reader of the record
# wants the month labels and a column of window objects would have to be unpacked by every one of them.
# The list is the record's declaration, and it is checked on the records themselves in
# `assert_no_look_ahead`: the frame that used to render it was built for every run and read by nobody,
# so the declaration has to be held against something a run actually produced.
FIELDS = (
    "traded",
    "window_start",
    "window_end",
    "months",
    "observations",
    "available_from",
    "gated_by",
    "refit",
    "components",
)

# The protocols the engine runs, mapped to the rule that cuts them. The rolling rule is the factor
# layer's own, read from there so the grid cannot cut its windows differently from the exposures it
# feeds; the expanding rule trades the same months with a window that grows from the panel's first.
PROTOCOLS = {"rolling": exposures.windows, "expanding": exposures.expanding_windows}


def steps(months, kind="rolling", window=exposures.WINDOW):
    """Every step of one protocol as a record: the month traded, the window estimated on and the
    availability that gated it.

    `observations` is left unset until the window is cut, because the panel's first bar carries no
    return and so the first window of a rolling protocol reads one row fewer than it declares. The
    declared length and the observed count are two fields rather than one for that reason: the
    difference is a structural absence in one window, never a gap to be excused elsewhere.
    """
    if kind not in PROTOCOLS:
        raise ValueError(f"the engine runs no protocol called {kind!r}; it declares {sorted(PROTOCOLS)}")
    records = []
    for traded, window_months in PROTOCOLS[kind](pd.PeriodIndex(months, freq="M"), window=window):
        records.append(
            {
                "traded": traded,
                "window": window_months,
                "window_start": window_months[0],
                "window_end": window_months[-1],
                "months": len(window_months),
                "observations": None,
                # The bar that set the gate, and the moment it became readable. Two fields, because
                # the record has to say which bar produced the availability rather than only when.
                "available_from": panel.available_from(window_months)[-1],
                "gated_by": window_months[-1],
                "refit": REFIT,
                "components": None,
            }
        )
    return tuple(records)


def block(returns, step):
    """The rows one step may estimate from, and how many of them there were.

    The count is returned beside the rows rather than measured by the caller, because the caller that
    measured it wrote it into the record from outside the module that owns the record - which is how a
    new field came to be a change in four files with no single validator.

    The check is the cheap half of the boundary and it runs before an optimiser is called, so a window
    that had been mis-cut cannot first produce a plausible book and be caught afterwards. The rule
    that cuts the window is the factor layer's, so the rows an estimate reads cannot differ from the
    rows the exposures read for the same month.
    """
    rows = exposures.window_block(returns, step["window"])
    if len(rows) and rows.index.max() >= step["traded"]:
        raise ValueError(
            f"the window {step['window_start']}..{step['window_end']} reaches {rows.index.max()}, which is "
            f"not before the month it trades, {step['traded']}: an estimate formed at the close of a month "
            f"cannot read that month's own bar"
        )
    return rows, len(rows)


def observed(step, taken, components=None):
    """The step's record completed with the two fields only the run can know.

    `steps` leaves both unset because it cuts no window and reads no estimator: how many rows a window
    actually held is measured when the window is cut, and the component count is the estimator's own
    report. Both are written here, by the module that owns the record, rather than assigned into it
    from the runner - a record half-filled at one call site and half-filled at another has no single
    place that can say what its fields are.
    """
    return {**step, "observations": taken, "components": components}


def assert_no_look_ahead(records, months):
    """The whole run's boundary, checked on the record: every window's last month precedes the month
    it trades, every window ends on the last bar that month could have read, the availability follows
    from the bar it names, the window is the contiguous run it claims, and only the panel's own front
    bar is ever absent.

    `months` is the panel calendar the windows were cut on, and it is what makes the one legitimate
    absence checkable: the panel's first bar carries no return, so a window starting there reads one
    row fewer, and a window starting anywhere else reads what it declares.

    This is the check the design promises rather than a property it assumes. Each of its clauses is a
    failure mode that produces a plausible weight path: a window reaching its own traded month, a
    window quietly shortened by one month, an availability that does not match its bar, a window
    spliced from two ranges, and a hole mid-window excused as a short history.

    The component counts the steps carry are summarised rather than compared. Under a protocol whose
    windows are one month apart a window key cannot repeat, so a comparison between two steps' counts
    could never fire; what the count path shows is that each step's model was fitted on its own
    trailing window, which the boundary clause above is what enforces.

    The record's own shape is checked here too. `FIELDS` declares it, and the frame that used to render
    the declaration was built for every run and read by nobody, so the declaration is checked on the
    records instead - the ones a run produced and the ones a test planted by hand alike, since both
    reach this function and a field added to one and not the other should fail rather than propagate.
    """
    if not len(records):
        raise ValueError("the engine has no steps to check; an empty run is not a boundary")
    declared = set(FIELDS) | {"window"}
    for record in records:
        if set(record) != declared:
            raise ValueError(
                f"the step record for {record.get('traded')} is not the shape the engine declares: "
                f"missing {sorted(declared - set(record))}, unexpected {sorted(set(record) - declared)}"
            )
    traded = [record["traded"] for record in records]
    if any(later <= earlier for earlier, later in zip(traded, traded[1:])):
        raise ValueError(f"the engine's traded months are not strictly increasing: {traded[:3]}")
    # The panel's first bar carries no return, so it is the one row a window may be short of - and
    # only a window that starts on it. Under the expanding protocol every window starts there, which
    # is why the excuse is tied to that month rather than to the first step of a run.
    front = pd.PeriodIndex(months, freq="M")[0]
    windows, counts = set(), []
    for record in records:
        month = record["traded"]
        if record["window_end"] >= month:
            raise ValueError(
                f"the estimate behind {month} reaches {record['window_end']}, so the month traded is "
                f"inside its own estimation window"
            )
        if record["window_end"] != month - 1:
            raise ValueError(
                f"the estimate behind {month} stops at {record['window_end']} rather than at {month - 1}, "
                f"the last bar that month could have read"
            )
        if record["available_from"] != panel.available_from(record["window"])[-1]:
            raise ValueError(
                f"the window behind {month} says its last bar {record['gated_by']} became readable at "
                f"{record['available_from']}, which is not the first day of the month after it"
            )
        contiguous = pd.period_range(record["window_start"], record["window_end"], freq="M")
        if record["window_start"] != record["window"][0] or record["months"] != len(record["window"]) or record["months"] != len(contiguous):
            raise ValueError(
                f"the window behind {month} is not the contiguous run its record claims: "
                f"{record['window_start']}..{record['window_end']} in {record['months']} months"
            )
        if record["observations"] is not None:
            shortfall = record["months"] - record["observations"]
            if shortfall < 0 or shortfall > 1:
                raise ValueError(
                    f"the window behind {month} declares {record['months']} months and read "
                    f"{record['observations']}; at most the panel's first bar may be absent"
                )
            if shortfall == 1 and record["window_start"] != front:
                raise ValueError(
                    f"the window behind {month} is one observation short and starts at "
                    f"{record['window_start']} rather than at the panel's first bar {front}: only that bar "
                    f"has no return to compute, so the row is missing rather than uncomputable"
                )
        windows.add((str(record["window_start"]), str(record["window_end"])))
        if record["components"] is not None:
            counts.append(int(record["components"]))
    return {
        "steps": len(records),
        "first_traded": str(traded[0]),
        "last_traded": str(traded[-1]),
        "windows": len(windows),
        "components": sorted(set(counts)) if counts else None,
        "gate": "every window's last bar becomes readable on the first day of the month it is traded",
        "observations": sum(record["observations"] or 0 for record in records),
    }


def main(root=None):
    """The engine's own report: both protocols re-cut on the frozen snapshot, every step of the rolling
    one printed, and the boundary checked on each."""
    document = loader.load_panel(root)
    returns = document.returns
    for kind in sorted(PROTOCOLS):
        records = []
        for record in steps(document.months, kind):
            # Only the window's count is read here: the rows themselves are what the runner estimates
            # from, and this entry point is checking that the engine cuts the window it claims to.
            records.append(observed(record, block(returns, record)[1]))
        report = assert_no_look_ahead(records, document.months)
        print(
            f"[table] the {kind} protocol: {report['steps']} steps {report['first_traded']}..{report['last_traded']} "
            f"over {report['windows']} windows, {report['observations']} sleeve-months read"
        )
        shown = records if kind == "rolling" else records[:1] + records[-1:]
        for record in shown:
            print(
                f"[table] {record['traded']}  window {record['window_start']}..{record['window_end']}  "
                f"{record['months']:>3d} months, {record['observations']:>3d} observations  "
                f"available {record['available_from']:%Y-%m-%d} from the {record['gated_by']} bar  "
                f"refit {record['refit']}"
            )
        if kind != "rolling":
            print(
                f"[table] the expanding window runs {records[0]['months']} months behind {records[0]['traded']} "
                f"and {records[-1]['months']} months behind {records[-1]['traded']}, trading the same months as the rolling one"
            )
        print(f"[table] {report['gate']}")
    return document


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
