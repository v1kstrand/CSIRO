import os
import sys
import json
import requests
import websocket
from urllib.parse import urljoin
from jupyter_client import KernelManager

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: main_init.py <run|cfg|vit> [arg]")
    mode = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if mode in ("run", "cfg") and not arg:
        raise RuntimeError(f"Usage: main_init.py {mode} <arg>")

    km = KernelManager()
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    token = os.environ.get("JUPYTER_TOKEN") or os.environ.get("JPY_API_TOKEN")

    if not token:
        raise RuntimeError("Jupyter token not found in environment")

    # Launch terminal
    base_url = "http://127.0.0.1:8888"
    resp = requests.post(urljoin(base_url, "/api/terminals"), headers={"Authorization": f"token {token}"})
    resp.raise_for_status()
    term_name = resp.json()['name']

    def on_open(ws):
        print("[Terminal] Connected.")
        ws.send(json.dumps(["stdin", "bash\r"]))
        setup_dir = "/notebooks/setups"
        if mode == "vit":
            ws.send(json.dumps(["stdin", f"bash {setup_dir}/main_init.sh vit\r"]))
        elif mode == "run":
            ws.send(json.dumps(["stdin", f"bash {setup_dir}/main_init.sh run {arg}\r"]))
        else:
            ws.send(json.dumps(["stdin", f"bash {setup_dir}/main_init.sh cfg {arg}\r"]))

    def on_error(ws, error):
        print("[Terminal] Error:", error)

    def on_close(ws, close_status_code, close_msg):
        print("[Terminal] Closed.")

    ws = websocket.WebSocketApp(
        f"ws://127.0.0.1:8888/terminals/websocket/{term_name}?token={token}",
        on_open=on_open,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()
