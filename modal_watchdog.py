import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


DEFAULT_APP = "minimax-h3-sglang"
DEFAULT_VOLUME = "minimax-h3-models"
DEFAULT_REMOTE_PATH = "/minimax/logs/modal_watchdog.log"
LOG_DIR = Path("logs")


def modal_binary() -> str:
    local_modal = Path(".venv/bin/modal")
    if local_modal.exists():
        return str(local_modal)
    found = shutil.which("modal")
    if found:
        return found
    raise RuntimeError("Could not find modal. Activate .venv or install modal first.")


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sync_to_modal_volume(log_file: Path, volume_name: str, remote_path: str, timeout: int = 60) -> bool:
    if not log_file.exists():
        return False

    cmd = [
        modal_binary(),
        "volume",
        "put",
        volume_name,
        str(log_file),
        remote_path,
        "--force",
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"[{stamp()}] watchdog: volume sync timed out after {timeout}s", flush=True)
        return False
    if result.returncode != 0:
        print(f"[{stamp()}] watchdog: volume sync failed:\n{result.stdout}", end="", flush=True)
        return False
    print(f"[{stamp()}] watchdog: synced log to {volume_name}:{remote_path}", flush=True)
    return True


def sync_loop(
    log_file: Path,
    volume_name: str,
    remote_path: str,
    sync_interval: int,
    dirty_event: threading.Event,
    stop_event: threading.Event,
) -> None:
    interval = max(sync_interval, 1)
    while not stop_event.is_set():
        dirty_event.wait(interval)
        if not dirty_event.is_set():
            continue
        dirty_event.clear()
        sync_to_modal_volume(log_file, volume_name, remote_path)

    if dirty_event.is_set():
        dirty_event.clear()
        sync_to_modal_volume(log_file, volume_name, remote_path)


def stream_logs(
    app_name: str,
    log_file: Path,
    reconnect_delay: int,
    volume_name: str,
    remote_path: str,
    sync_interval: int,
) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    cmd = [modal_binary(), "app", "logs", app_name, "--timestamps"]
    dirty_event = threading.Event()
    stop_event = threading.Event()
    sync_thread = threading.Thread(
        target=sync_loop,
        args=(log_file, volume_name, remote_path, sync_interval, dirty_event, stop_event),
        daemon=True,
    )
    sync_thread.start()

    try:
        with log_file.open("a", encoding="utf-8") as out:
            while True:
                header = f"\n[{stamp()}] watchdog: starting {' '.join(cmd)}\n"
                print(header, end="", flush=True)
                out.write(header)
                out.flush()
                dirty_event.set()

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=os.environ.copy(),
                )

                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    out.write(line)
                    out.flush()
                    dirty_event.set()

                code = process.wait()
                footer = f"[{stamp()}] watchdog: modal logs exited with {code}\n"
                print(footer, end="", flush=True)
                out.write(footer)
                out.flush()
                dirty_event.set()

                if reconnect_delay < 0:
                    break

                msg = f"[{stamp()}] watchdog: reconnecting in {reconnect_delay}s\n"
                print(msg, end="", flush=True)
                out.write(msg)
                out.flush()
                dirty_event.set()
                time.sleep(reconnect_delay)
    finally:
        stop_event.set()
        sync_thread.join(timeout=65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail Modal logs and mirror them to a local file.")
    parser.add_argument("--app", default=DEFAULT_APP, help="Modal app name or app ID.")
    parser.add_argument("--log-file", default=str(LOG_DIR / "modal_watchdog.log"))
    parser.add_argument("--volume", default=DEFAULT_VOLUME, help="Modal Volume to mirror the log into.")
    parser.add_argument("--remote-path", default=DEFAULT_REMOTE_PATH, help="Path inside the Modal Volume.")
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=5,
        help="Seconds between Modal Volume uploads while new logs arrive. Use 0 to sync every log line.",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=int,
        default=5,
        help="Seconds before reconnecting if the log stream exits. Use -1 to disable.",
    )
    args = parser.parse_args()

    try:
        stream_logs(
            args.app,
            Path(args.log_file),
            args.reconnect_delay,
            args.volume,
            args.remote_path,
            args.sync_interval,
        )
    except KeyboardInterrupt:
        sync_to_modal_volume(Path(args.log_file), args.volume, args.remote_path)
        print("\nwatchdog: stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
