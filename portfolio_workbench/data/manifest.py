"""The snapshot manifest, and the check that the snapshot is the one it claims to be.

The panel is fetched once and then frozen; every result is keyed to the snapshot id it
was read from. The check is fail-closed in the strict sense: a checksum, row-count or
date-range mismatch raises instead of warning, because a panel that has silently
changed underneath a result makes the result uninterpretable, not merely stale.
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "manifest.json"

# Every file entry carries these; `instrument` and `vintage` are added where they apply.
FILE_FIELDS = ("path", "role", "source", "url", "retrieved", "rows", "first", "last", "sha256")


class ManifestError(Exception):
    """The snapshot does not match its manifest. Never caught and patched downstream."""


def sha256(path):
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def measure(path):
    """Row count and date range of one artifact, read from the artifact itself.

    Both the central-bank CSVs and the price files carry one ISO date per row, so the
    range is a string comparison and no date library is needed to verify a file.
    The factor archives are counted on their monthly rows, which is the count the
    analytics consume.

    A file that does not parse is refused here rather than left to raise: the parser's
    own fault type is translated into this module's, so every way a snapshot can be wrong
    arrives at the caller as one vocabulary.
    """
    path = Path(path)
    if path.suffix == ".zip":
        from .external import SourceFormatError, parse_french_zip

        try:
            frame = parse_french_zip(path)
        except SourceFormatError as fault:
            raise ManifestError(f"{path.name}: {fault}") from fault
        if frame.empty:
            raise ManifestError(f"{path.name}: the factor archive holds no monthly rows")
        return len(frame), str(frame.index[0]), str(frame.index[-1])

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise ManifestError(f"{path.name}: empty file")
        column = "TIME_PERIOD" if "TIME_PERIOD" in header else "date"
        if column not in header:
            raise ManifestError(f"{path.name}: no date column in {header[:5]}")
        index = header.index(column)
        dates = [row[index] for row in reader if len(row) > index and row[index].strip()]
    if not dates:
        raise ManifestError(f"{path.name}: no dated rows")
    return len(dates), min(dates), max(dates)


def describe(root, relative, role, source, url, retrieved, **extra):
    """One manifest entry for one file, measured rather than asserted."""
    path = Path(root) / relative
    if not path.exists():
        raise ManifestError(f"{relative}: listed for the snapshot but not present")
    rows, first, last = measure(path)
    entry = {
        "path": relative,
        "role": role,
        "source": source,
        "url": url,
        "retrieved": retrieved,
        "rows": rows,
        "first": first,
        "last": last,
        "sha256": sha256(path),
    }
    entry.update(extra)
    return entry


def build(root, entries, instruments, extra=None):
    """Write the manifest. `entries` come from `describe`, so nothing is hand-typed."""
    root = Path(root)
    document = {
        "snapshot_id": root.name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": entries,
        "instruments": instruments,
    }
    if extra:
        document.update(extra)
    (root / MANIFEST_NAME).write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")
    return document


def read(root):
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        raise ManifestError(f"{root}: no {MANIFEST_NAME}; the snapshot is not frozen")
    return json.loads(path.read_text())


def verify(root):
    """Re-derive every entry and refuse to proceed on any difference.

    The whole snapshot is verified on every load: it is a few dozen small files, and
    a partial check is the failure mode this module exists to prevent.
    """
    root = Path(root)
    document = read(root)
    for entry in document["files"]:
        missing = [field for field in FILE_FIELDS if field not in entry]
        if missing:
            raise ManifestError(f"{entry.get('path', '?')}: manifest entry lacks {missing}")
        path = root / entry["path"]
        if not path.exists():
            raise ManifestError(f"{entry['path']}: in the manifest, absent from the snapshot")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise ManifestError(
                f"{entry['path']}: sha256 {actual[:12]} does not match the manifest "
                f"{entry['sha256'][:12]}; the snapshot is not the one the results were keyed to"
            )
        rows, first, last = measure(path)
        if (rows, first, last) != (entry["rows"], entry["first"], entry["last"]):
            raise ManifestError(
                f"{entry['path']}: manifest says {entry['rows']} rows {entry['first']}..{entry['last']}, "
                f"file holds {rows} rows {first}..{last}"
            )
    return document
