"""``genki-ui`` console entry point: launch the local web UI in a browser."""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="genki-ui", description="Run the local GenKI web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8931, help="bind port (default: 8931)")
    parser.add_argument("--grn-dir", default="GRNs", help="directory to cache built GRNs in")
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    args = parser.parse_args(argv)

    import uvicorn

    from .main import create_app

    app = create_app(grn_dir=args.grn_dir)
    url = f"http://{args.host}:{args.port}"

    if not args.no_browser:
        def _open_browser():
            time.sleep(1.0)  # give uvicorn a moment to start listening
            webbrowser.open(url)

        threading.Thread(target=_open_browser, daemon=True).start()

    print(f"GenKI UI: {url}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
