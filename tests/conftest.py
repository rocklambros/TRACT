"""Pytest configuration shared by the whole suite.

Pins a headless matplotlib backend before any test imports pyplot.

The failure this prevents is not a test failure. On a machine with DISPLAY set --
the Jetson that drives the RunPod fleet runs with DISPLAY=:1 -- matplotlib
resolves the default backend to qtagg, and the first `plt.subplots()` call tries
to create a QApplication. That aborts the interpreter with SIGABRT, so pytest
dies at whatever percentage it had reached and prints no summary line at all.
Measured 2026-08-26: the suite died at ~44% inside
tests/test_nb_helpers.py::TestStyleAxes::test_applies_title_and_labels, exit 134,
no tally. A run that ends that way reads as an infrastructure hiccup rather than
as a suite that never finished, which is the shape of failure worth a file.

CI never saw it. The runners have no DISPLAY, and the default `test` job does not
install matplotlib, so the modules that need it skip at import. Green CI was
therefore never evidence that this path worked -- it was evidence that it was
never taken.

`matplotlib.use` has to run before `matplotlib.pyplot` is first imported.
conftest.py is imported before any test module is collected, which is early
enough. MPLBACKEND is set as well, so anything the suite shells out to inherits
the same choice rather than re-deriving it from DISPLAY.
"""

from __future__ import annotations

import os

# setdefault, not assignment: an operator who exports MPLBACKEND deliberately
# (to render a figure while debugging one test) keeps their choice.
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import matplotlib
except ImportError:
    # The default CI test job installs no matplotlib and the modules that need
    # it skip at import. There is no backend to pin.
    pass
else:
    matplotlib.use(os.environ["MPLBACKEND"])
