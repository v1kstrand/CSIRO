import os
import sys
import json
import shlex
import threading
import requests
import websocket
from urllib.parse import urljoin
from jupyter_client import KernelManager

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: terminal_launch.py <init|run|vit|backup> [arg]")
    mode = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if mode == "run" and not arg:
        raise RuntimeError("Usage: terminal_launch.py run <arg>")

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

    setup_dir = "/notebooks/setups"
    if mode == "init":
        launch_cmd = f"python {setup_dir}/scheduler_loop.py"
    elif mode == "backup":
        launch_cmd = f"bash {setup_dir}/gpu_backup.sh"
    elif mode == "vit":
        launch_cmd = f"bash {setup_dir}/venv_run.sh vit"
    elif mode == "run":
        launch_cmd = f"bash {setup_dir}/venv_run.sh run {shlex.quote(arg)}"
    else:
        raise RuntimeError(f"Unsupported mode: {mode}")

    ready_token = "__TERMINAL_READY__"
    command_sent = {"done": False}
    timer_holder = {"timer": None}

    def on_open(ws):
        print("[Terminal] Connected.")
        ws.send(json.dumps(["stdin", "bash\r"]))
        ws.send(json.dumps(["stdin", f"echo {ready_token}\r"]))

    def on_message(ws, message):
        if command_sent["done"]:
            return
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return
        if not isinstance(data, list) or len(data) < 2:
            return
        payload = str(data[1])
        if ready_token in payload:
            command_sent["done"] = True
            if timer_holder["timer"] is not None:
                timer_holder["timer"].cancel()
            print("[Terminal] Ready. Launching command.")
            ws.send(json.dumps(["stdin", f"{launch_cmd}\r"]))

    def on_error(ws, error):
        print("[Terminal] Error:", error)

    def on_close(ws, close_status_code, close_msg):
        print("[Terminal] Closed.")

    def on_timeout(ws):
        if command_sent["done"]:
            return
        print("[Terminal] Ready timeout; closing.")
        try:
            ws.close()
        except Exception:
            pass

    ws = websocket.WebSocketApp(
        f"ws://127.0.0.1:8888/terminals/websocket/{term_name}?token={token}",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    timer_holder["timer"] = threading.Timer(20.0, on_timeout, args=(ws,))
    timer_holder["timer"].start()
    ws.run_forever()
