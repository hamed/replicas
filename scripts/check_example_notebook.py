"""Execute an example notebook and compare its outputs with the committed ones.

The example notebook drives the public API the way the README tells a user to:
pandas bootstrap, Spark bootstrap, all three metric functions, and both plot
helpers. Running it in CI catches two kinds of regression that the unit tests
miss.

1. A cell raises. The package broke in a way no test covers.
2. A cell still runs but prints different numbers. The committed outputs are
   part of the artifact (rule 2 of ``docs/reference-design.md``), so a silent
   change of results is a defect until someone re-executes the notebook on
   purpose.

Only ``execute_result`` and ``display_data`` payloads are compared. ``stderr``
streams carry Spark log lines with timestamps and host names, and every repr of
a live object carries its address; both are normalized away or ignored.

Figure dimensions are normalized away too. Matplotlib and seaborn move the
pixel size of a figure between releases -- CI once produced ``730x600`` where
this machine produced ``718x600``, purely from legend spacing -- so comparing
them reports a dependency bump as a defect. The axes count in the same repr is
kept: a facet that stops rendering, or a grid that changes shape, still fails.

Usage:

    python scripts/check_example_notebook.py examples/quickstart.ipynb
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

# `<seaborn.axisgrid.FacetGrid at 0x7f3c1a2b4d90>` embeds a memory address that
# changes on every run. The lookbehind matters: without it the `0x600` inside
# `<Figure size 730x600 ...>` also matches, and the checker rewrites a figure
# dimension into the address placeholder.
_ADDRESS = re.compile(r"(?<![0-9A-Za-z])0x[0-9a-fA-F]+")
# `<Figure size 718x600 with 1 Axes>` -> `<Figure size ... with 1 Axes>`.
_FIGURE_SIZE = re.compile(r"(<Figure size )[\d.]+x[\d.]+( with )")
_COMPARED = ("execute_result", "display_data")


def _normalize(text: str) -> str:
    return _FIGURE_SIZE.sub(r"\1...\2", _ADDRESS.sub("0xADDR", text))


def _payloads(cell) -> list:
    """Return the comparable text of one cell, with addresses normalized."""
    texts = []
    for output in cell.get("outputs", []):
        if output.get("output_type") not in _COMPARED:
            continue
        text = output.get("data", {}).get("text/plain")
        if text:
            joined = text if isinstance(text, str) else "".join(text)
            texts.append(_normalize(joined))
    return texts


def _errors(notebook) -> list:
    """Return `(index, ename, evalue)` for every cell that raised."""
    failures = []
    for index, cell in enumerate(notebook.cells):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                failures.append((index, output.get("ename", ""), output.get("evalue", "")))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="per-cell execution timeout in seconds (default: 1800)",
    )
    arguments = parser.parse_args()

    committed = nbformat.read(arguments.notebook, as_version=4)
    executed = nbformat.read(arguments.notebook, as_version=4)

    print(f"executing {arguments.notebook}", flush=True)
    NotebookClient(
        executed,
        timeout=arguments.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(arguments.notebook.parent)}},
        allow_errors=True,
    ).execute()

    failures = _errors(executed)
    if failures:
        print(f"\nFAIL: {len(failures)} cell(s) raised during execution:", file=sys.stderr)
        for index, ename, evalue in failures:
            print(f"  cell {index}: {ename}: {evalue}", file=sys.stderr)
        return 1

    mismatches = 0
    for index, (before, after) in enumerate(zip(committed.cells, executed.cells)):
        if before.get("cell_type") != "code":
            continue
        expected, actual = _payloads(before), _payloads(after)
        if expected == actual:
            continue
        mismatches += 1
        print(f"\nFAIL: cell {index} produced different output:", file=sys.stderr)
        diff = difflib.unified_diff(
            "\n".join(expected).splitlines(),
            "\n".join(actual).splitlines(),
            fromfile="committed",
            tofile="executed",
            lineterm="",
        )
        for line in diff:
            print(f"  {line}", file=sys.stderr)

    if mismatches:
        print(
            f"\n{mismatches} cell(s) drifted. If the change is intended, re-execute "
            f"{arguments.notebook} and commit the new outputs.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {arguments.notebook} ran clean and matches its committed outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
