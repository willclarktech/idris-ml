"""mltools — internal Python package shared by the top-level scripts/.

The package lives at scripts/mltools/ and is reached from bash via
`PYTHONPATH=$REPO/scripts python3 -m mltools.<module>` (see
scripts/perf_lib.sh) or from Python siblings via `from mltools.<module>
import ...`.

The package is the home for shared logic that multiple scripts need
(perf-log entry construction, header parsing, sweep grid expansion).
Single-script logic stays in the script.
"""
