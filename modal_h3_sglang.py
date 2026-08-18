import os
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "minimax-h3-sglang"
MODEL_ID = "MiniMaxAI/MiniMax-H3"
MODEL_VOLUME_NAME = "minimax-h3-models"
MODEL_VOLUME_PATH = Path("/models")
HF_HOME = MODEL_VOLUME_PATH / "hf"
PORT = 30010
WARMUP_SECONDS = int(os.environ.get("H3_WARMUP_SECONDS", "2"))

# FL2VA serves t2va and fl2va. Deploy a second copy with MODEL_VARIANT=ref2va
# if you need reference/video/audio-conditioned requests.
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "fl2va")
GPU = os.environ.get("H3_GPU", "H100:4")
NUM_GPUS = os.environ.get("H3_NUM_GPUS", "4")

volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "curl", "ffmpeg", "libgl1", "libglib2.0-0")
    .uv_pip_install(
        "huggingface_hub[cli]",
        "hf_transfer",
        "sglang[diffusion]",
        pre=True,
    )
    .env(
        {
            "HF_HOME": str(HF_HOME),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install("fastapi"),
    volumes={str(MODEL_VOLUME_PATH): volume},
    timeout=300,
    cpu=1,
    memory=1024,
)
@modal.fastapi_endpoint(method="GET", label="media")
def media(path: str):
    """Serve uploaded input media from the Modal Volume for SGLang conditions."""
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    volume.reload()
    safe_path = Path("/" + path.lstrip("/")).resolve()
    root = Path("/minimax/inputs").resolve()
    if root not in safe_path.parents and safe_path != root:
        raise HTTPException(status_code=403, detail="Path is outside /minimax/inputs")

    file_path = MODEL_VOLUME_PATH / safe_path.relative_to("/")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Missing media file: {path}")
    return FileResponse(file_path)


@app.function(
    image=image,
    volumes={str(MODEL_VOLUME_PATH): volume},
    timeout=24 * 60 * 60,
    cpu=8,
    memory=32768,
    ephemeral_disk=1024 * 1024,
    retries=0,
)
def download_model(include_ref2va: bool = False) -> dict:
    """Download H3 tensors into the Modal Volume-backed Hugging Face cache."""
    from huggingface_hub import snapshot_download

    patterns = ["model_index.json", "FL2VA/*"]
    if include_ref2va:
        patterns.append("Ref2VA/*")

    path = snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=patterns,
        cache_dir=str(HF_HOME / "hub"),
        resume_download=True,
    )
    volume.commit()

    total_bytes = 0
    file_count = 0
    for file_path in Path(path).rglob("*"):
        if file_path.is_file():
            file_count += 1
            total_bytes += file_path.stat().st_size

    return {
        "repo_id": MODEL_ID,
        "snapshot_path": path,
        "include_ref2va": include_ref2va,
        "files": file_count,
        "bytes": total_bytes,
    }


@app.function(
    image=image,
    gpu=GPU,
    volumes={str(MODEL_VOLUME_PATH): volume},
    timeout=24 * 60 * 60,
    cpu=16,
    memory=393216,
    ephemeral_disk=1024 * 1024,
    scaledown_window=30 * 60,
    max_containers=1,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=16)
@modal.web_server(PORT, startup_timeout=60 * 60)
def serve_h3():
    """Expose SGLang's OpenAI-compatible /v1/videos API on Modal."""
    volume.reload()

    command = [
        "sglang",
        "serve",
        "--model-path",
        MODEL_ID,
        "--model-variant",
        MODEL_VARIANT,
        "--num-gpus",
        NUM_GPUS,
        "--tp-size",
        "2",
        "--ulysses-degree",
        "2",
        "--performance-mode",
        "speed",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
    ]

    print("Starting:", " ".join(command), flush=True)
    subprocess.Popen(command)
    print(f"Warm-up hold: {WARMUP_SECONDS}s", flush=True)
    time.sleep(WARMUP_SECONDS)
