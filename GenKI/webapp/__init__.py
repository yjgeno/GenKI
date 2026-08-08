"""Local web UI for GenKI.

Install with the ``web`` extra and launch:

    pip install "GenKI[web]"
    genki-ui

Serves a single-page app (static/) backed by a small FastAPI JSON API
(main.py) that runs the GenKI workflow (GenKI/api.py) as background jobs.
"""

from .main import create_app

__all__ = ["create_app"]
