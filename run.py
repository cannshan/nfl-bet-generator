import os
from app.app import app

if __name__ == "__main__":
    # Normal local dev: 127.0.0.1, debug on (auto-reload). For phone/LAN
    # testing, set HOST=0.0.0.0 so other devices on the same WiFi can reach
    # it -- debug mode is force-disabled whenever bound beyond localhost,
    # since Werkzeug's interactive debugger shouldn't be reachable by
    # anything other than this machine.
    host = os.environ.get("HOST", "127.0.0.1")
    debug = host == "127.0.0.1"
    port = int(os.environ.get("PORT", "5057"))
    app.run(host=host, port=port, debug=debug)
