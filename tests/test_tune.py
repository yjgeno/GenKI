"""Quarantine marker for the Ray-Tune integration.

GenKI.tune relies on the pre-Ray-2 Tune API (`tune.run`, `tune.report`,
`tune.checkpoint_dir`), which was removed in Ray >= 2.x. The rewrite is
deferred to a later modernization stage; this test documents the gap so
the suite stays green and the omission is visible.
"""

import pytest


@pytest.mark.skip(reason="GenKI.tune uses removed Ray<2 Tune API; rewrite deferred")
def test_tune_main_runs():
    raise AssertionError("placeholder until GenKI.tune is ported to Ray 2.x")
