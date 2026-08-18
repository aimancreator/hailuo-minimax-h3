import argparse
import base64
import mimetypes
import shutil
import socket
import subprocess
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import quote

OUTPUT_DIR = Path("outputs")
REPORT_DIR = Path("reports")
INPUT_DIR = Path("inputs")
DEFAULT_VOLUME = "minimax-h3-models"
DEFAULT_REMOTE_ROOT = "/minimax/generations"
DEFAULT_INPUT_REMOTE_ROOT = "/minimax/inputs"
MODAL_VOLUME_MOUNT = "/models"
DEFAULT_REQUEST_TIMEOUT = 3600


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MiniMax H3 Local WebUI</title>
  <style>
    :root { color-scheme: dark light; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; }
    body { margin: 0; background: #111318; color: #eef1f6; }
    main { max-width: 980px; margin: 0 auto; padding: 28px; }
    h1 { font-size: 24px; margin: 0 0 18px; font-weight: 650; }
    label { display: block; font-size: 13px; color: #b9c0cc; margin: 16px 0 6px; }
    input, textarea, select, button {
      width: 100%; box-sizing: border-box; border: 1px solid #363b46; border-radius: 6px;
      background: #191d25; color: #eef1f6; padding: 10px 12px; font: inherit;
    }
    textarea { min-height: 220px; resize: vertical; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    button { margin-top: 16px; cursor: pointer; background: #f26f3f; border-color: #f26f3f; color: white; font-weight: 650; }
    button:disabled { opacity: .6; cursor: wait; }
    pre { white-space: pre-wrap; background: #191d25; border: 1px solid #363b46; border-radius: 6px; padding: 12px; min-height: 80px; }
    a { color: #8cc8ff; }
    @media (max-width: 760px) { main { padding: 18px; } .grid { grid-template-columns: 1fr 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>MiniMax H3 Local WebUI</h1>
  <label for="server">Modal SGLang URL</label>
  <input id="server" value="__SERVER_URL__">

  <label for="prompt">Prompt</label>
  <textarea id="prompt">integrated_multimodal_description: A cinematic 5 second video of a futuristic city at sunrise, slow forward camera move, reflective glass towers, soft atmospheric haze.
overall_soundscape: Gentle city ambience with distant traffic and a warm breeze.
non_diegetic_music: Minimal hopeful synth pad, slow tempo.</textarea>

  <div class="grid">
    <div><label for="task">Task</label><select id="task"><option value="t2va">t2va</option><option value="i2va">i2va</option><option value="fl2va">fl2va</option><option value="ref2va">ref2va</option></select></div>
    <div><label for="duration">Seconds</label><input id="duration" type="number" min="4" max="15" step="1" value="5"></div>
    <div><label for="aspect">Aspect</label><input id="aspect" value="16:9"></div>
    <div><label for="seed">Seed</label><input id="seed" type="number" value="1101"></div>
  </div>

  <label for="firstFrame">First frame image</label>
  <input id="firstFrame" type="file" accept="image/png,image/jpeg,image/webp">

  <label for="json">Conditions JSON</label>
  <textarea id="json">[]</textarea>

  <button id="go">Generate</button>
  <label>Status</label>
  <pre id="status">Idle</pre>
</main>
<script>
const go = document.querySelector("#go");
const statusBox = document.querySelector("#status");
const firstFrame = document.querySelector("#firstFrame");

const readFileAsDataUrl = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve(reader.result);
  reader.onerror = () => reject(reader.error);
  reader.readAsDataURL(file);
});

go.onclick = async () => {
  go.disabled = true;
  statusBox.textContent = "Submitting... First cold start can take several minutes while H3 loads on Modal.";
  try {
    const selectedTask = document.querySelector("#task").value;
    const upload = firstFrame.files.length ? {
      name: firstFrame.files[0].name,
      data_url: await readFileAsDataUrl(firstFrame.files[0])
    } : null;
    const requestTask = selectedTask === "i2va" ? "fl2va" : selectedTask;
    const conditions = selectedTask === "i2va" ? [] : JSON.parse(document.querySelector("#json").value);
    if (selectedTask === "i2va" && !upload) throw new Error("Choose a first frame image for i2va.");

    const payload = {
      server_url: document.querySelector("#server").value.trim(),
      mode: selectedTask,
      first_frame_upload: upload,
      request: {
        model: "MiniMaxAI/MiniMax-H3",
        prompt: document.querySelector("#prompt").value,
        seconds: Number(document.querySelector("#duration").value),
        task: requestTask,
        conditions,
        target: {
          short_edge: 768,
          aspect_ratio: selectedTask === "i2va" ? "auto" : document.querySelector("#aspect").value,
          duration_seconds: Number(document.querySelector("#duration").value)
        },
        num_outputs_per_prompt: 1,
        num_inference_steps: 50,
        flow_shift: 12.0,
        audio_flow_shift: 3.0,
        seed: Number(document.querySelector("#seed").value)
      }
    };
    const res = await fetch("/generate", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || JSON.stringify(data));
    statusBox.innerHTML = `Completed: <a href="${data.output_url}" target="_blank">${data.output_file}</a>
Uploaded report: ${data.remote_report}
Uploaded video: ${data.remote_video}`;
  } catch (err) {
    statusBox.textContent = "Error: " + err.message;
  } finally {
    go.disabled = false;
  }
};
</script>
</body>
</html>
"""


def request_json(
    url: str,
    method: str = "GET",
    payload: Optional[dict] = None,
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def wait_server_ready(server_url: str) -> dict:
    return request_json(f"{server_url}/health")


def modal_binary() -> str:
    local_modal = Path(".venv/bin/modal")
    if local_modal.exists():
        return str(local_modal)
    found = shutil.which("modal")
    if found:
        return found
    raise RuntimeError("Could not find modal. Activate .venv or install modal first.")


def upload_to_volume(local_path: Path, volume_name: str, remote_path: str) -> None:
    command = [modal_binary(), "volume", "put", volume_name, str(local_path), remote_path, "--force"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Modal Volume upload failed for {local_path}:\n{result.stdout}")


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(name).name)


def save_upload(upload: dict, prefix: str) -> tuple[Path, str, str]:
    data_url = upload["data_url"]
    if "," not in data_url:
        raise RuntimeError("Invalid upload data URL.")

    header, encoded = data_url.split(",", 1)
    if ";base64" not in header:
        raise RuntimeError("Only base64 file uploads are supported.")

    content_type = header.removeprefix("data:").split(";", 1)[0] or "application/octet-stream"
    suffix = Path(upload.get("name", "")).suffix.lower()
    if not suffix:
        suffix = mimetypes.guess_extension(content_type) or ".bin"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise RuntimeError(f"Unsupported image extension: {suffix}")

    INPUT_DIR.mkdir(exist_ok=True)
    filename = f"{prefix}_{safe_name(upload.get('name') or 'first-frame')}"
    if not Path(filename).suffix:
        filename = f"{filename}{suffix}"
    path = INPUT_DIR / filename
    path.write_bytes(base64.b64decode(encoded))
    return path, content_type, filename


class Handler(BaseHTTPRequestHandler):
    server_url = ""
    media_url = ""
    volume_name = DEFAULT_VOLUME
    remote_root = DEFAULT_REMOTE_ROOT

    def do_GET(self):
        if self.path == "/":
            body = HTML.replace("__SERVER_URL__", self.server_url).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/outputs/"):
            name = Path(self.path.removeprefix("/outputs/")).name
            path = OUTPUT_DIR / name
            if not path.exists():
                self.send_error(404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404)

    def do_POST(self):
        if self.path != "/generate":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode())
            server_url = payload["server_url"].rstrip("/")
            request_payload = payload["request"]
            mode = payload.get("mode", request_payload.get("task"))
            uploaded_inputs = []

            if mode == "i2va":
                upload = payload.get("first_frame_upload")
                if not upload:
                    raise RuntimeError("i2va requires a first frame image.")

                upload_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
                first_frame_path, content_type, first_frame_name = save_upload(upload, f"i2va_{upload_id}")
                remote_first_frame = f"{DEFAULT_INPUT_REMOTE_ROOT}/{first_frame_name}"
                upload_to_volume(first_frame_path, self.volume_name, remote_first_frame)
                media_url = self.media_url or server_url.replace("-serve-h3.", "-media.")
                first_frame_uri = f"{media_url}?path={quote(remote_first_frame.lstrip('/'), safe='')}"
                request_payload["task"] = "fl2va"
                request_payload["conditions"] = [
                    {
                        "type": "image",
                        "uri": first_frame_uri,
                        "role": "keyframe",
                        "frame_index": 0,
                    }
                ]
                request_payload.setdefault("target", {})["aspect_ratio"] = "auto"
                uploaded_inputs.append(
                    {
                        "local_path": str(first_frame_path),
                        "content_type": content_type,
                        "remote_path": f"{self.volume_name}:{remote_first_frame}",
                        "uri": first_frame_uri,
                    }
                )

            health = wait_server_ready(server_url)
            created = request_json(f"{server_url}/v1/videos", "POST", request_payload)
            video_id = created["id"]

            while True:
                status = request_json(f"{server_url}/v1/videos/{quote(video_id)}", timeout=300)
                state = status.get("status")
                if state == "completed":
                    break
                if state in {"failed", "cancelled", "canceled"}:
                    raise RuntimeError(json.dumps(status))
                time.sleep(5)

            OUTPUT_DIR.mkdir(exist_ok=True)
            REPORT_DIR.mkdir(exist_ok=True)
            output_file = f"{video_id}.mp4".replace("/", "_")
            output_path = OUTPUT_DIR / output_file
            with urllib.request.urlopen(
                f"{server_url}/v1/videos/{quote(video_id)}/content",
                timeout=DEFAULT_REQUEST_TIMEOUT,
            ) as response:
                output_path.write_bytes(response.read())

            safe_video_id = output_path.stem
            report_file = f"{safe_video_id}.json"
            report_path = REPORT_DIR / report_file
            completed_at = datetime.now().astimezone().isoformat(timespec="seconds")
            report = {
                "video_id": video_id,
                "completed_at": completed_at,
                "server_url": server_url,
                "health_before_submit": health,
                "mode": mode,
                "uploaded_inputs": uploaded_inputs,
                "request": request_payload,
                "create_response": created,
                "final_status": status,
                "local_video": str(output_path),
                "local_report": str(report_path),
            }
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

            remote_base = f"{self.remote_root.rstrip('/')}/{safe_video_id}"
            remote_video = f"{remote_base}/{output_file}"
            remote_report = f"{remote_base}/{report_file}"
            upload_to_volume(report_path, self.volume_name, remote_report)
            upload_to_volume(output_path, self.volume_name, remote_video)

            self.respond_json(
                {
                    "video_id": video_id,
                    "output_file": output_file,
                    "output_url": f"/outputs/{output_file}",
                    "report_file": report_file,
                    "remote_report": f"{self.volume_name}:{remote_report}",
                    "remote_video": f"{self.volume_name}:{remote_video}",
                }
            )
        except (
            KeyError,
            json.JSONDecodeError,
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            RuntimeError,
            subprocess.TimeoutExpired,
        ) as exc:
            self.respond_json({"error": str(exc)}, status=500)

    def respond_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="", help="Modal URL printed by `modal deploy` for serve_h3.")
    parser.add_argument("--media-url", default="", help="Modal media URL. Defaults by replacing -serve-h3 with -media.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--volume", default=DEFAULT_VOLUME, help="Modal Volume to upload generation reports/results into.")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT, help="Root folder inside the Modal Volume.")
    args = parser.parse_args()

    Handler.server_url = args.server_url
    Handler.media_url = args.media_url.rstrip("/")
    Handler.volume_name = args.volume
    Handler.remote_root = args.remote_root
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Local WebUI: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
