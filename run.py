import os
import threading
import time
from app.app import app

# Every this-many seconds the local server checks whether a game with
# pending tracked bets kicks off soon and, if so, records its closing odds
# (app.capture_closing_lines_now). Set CLOSING_CAPTURE=0 to turn it off.
CLOSING_CAPTURE_INTERVAL_SECONDS = 10 * 60


def _closing_capture_loop():
    from app import cache_utils
    from app.app import capture_closing_lines_now
    while True:
        time.sleep(CLOSING_CAPTURE_INTERVAL_SECONDS)
        cache_utils.set_mode("active")
        try:
            capture_closing_lines_now()
        except Exception as e:  # never let the timer thread die
            print(f"[closing-capture] {type(e).__name__}: {e}")
        finally:
            cache_utils.set_mode("passive")

if __name__ == "__main__":
    # Normal local dev: 127.0.0.1, debug on (auto-reload). For phone/LAN
    # testing, set HOST=0.0.0.0 so other devices on the same WiFi can reach
    # it -- debug mode is force-disabled whenever bound beyond localhost,
    # since Werkzeug's interactive debugger shouldn't be reachable by
    # anything other than this machine.
    host = os.environ.get("HOST", "127.0.0.1")
    debug = host == "127.0.0.1"
    port = int(os.environ.get("PORT", "5057"))
    # With the debug reloader, only the child process (the one that serves
    # requests) runs the timer -- otherwise it would run twice.
    if os.environ.get("CLOSING_CAPTURE", "1") != "0" and (not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true"):
        threading.Thread(target=_closing_capture_loop, daemon=True).start()
    app.run(host=host, port=port, debug=debug)
