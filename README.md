# MiniMax H3 on Modal

This workspace deploys MiniMax H3 Base through SGLang on Modal and keeps the Hugging Face tensor cache on a Modal Volume named `minimax-h3-models`.

MiniMax H3 has its own license and usage restrictions. Review the current model card/license before downloading or running the weights.

## 1. Download weights to Modal storage

FL2VA is enough for text-to-video-audio (`t2va`) and first/last-frame (`fl2va`):

```bash
.venv/bin/modal run modal_h3_sglang.py::download_model
```

To also download Ref2VA:

```bash
.venv/bin/modal run modal_h3_sglang.py::download_model --include-ref2va true
```

## 2. Deploy the SGLang server

```bash
.venv/bin/modal deploy modal_h3_sglang.py
```

The deploy output prints a URL for `serve_h3`. Use that URL in the local WebUI.

## 3. Run the local WebUI

```bash
.venv/bin/python local_webui.py --server-url <modal-serve-h3-url>
```

Open `http://127.0.0.1:7860`.

When a generation completes, the WebUI downloads the MP4 locally and uploads both the MP4 and a JSON report to Modal storage:

```text
minimax-h3-models:/minimax/generations/<video_id>/
```

For I2VA, choose `i2va` in the WebUI and select a first-frame image. The WebUI uploads that image to:

```text
minimax-h3-models:/minimax/inputs/
```

Then it sends SGLang an FL2VA request with a first-frame keyframe condition.

## 4. Watch Modal logs locally

Run this in a second terminal while generating:

```bash
.venv/bin/python modal_watchdog.py
```

It streams new Modal logs with timestamps and also writes them to `logs/modal_watchdog.log`.
It also mirrors that file into Modal storage at `minimax-h3-models:/minimax/logs/modal_watchdog.log`.

## Notes

- Default deployment is `MODEL_VARIANT=fl2va`, `H3_GPU=H100:4`, `H3_NUM_GPUS=4`.
- Modal retries are disabled on the download function (`retries=0`). Modal web endpoints reject the retries setting, so `serve_h3` has no retry policy configured.
- The SGLang web function has Modal CPU memory snapshot and experimental GPU snapshot enabled.
- The server holds for `H3_WARMUP_SECONDS=2` seconds after launching SGLang.
- Override before deploy if needed, for example:

```bash
MODEL_VARIANT=ref2va H3_GPU=H100:4 H3_NUM_GPUS=4 H3_WARMUP_SECONDS=2 .venv/bin/modal deploy modal_h3_sglang.py
```

- Local media paths in `conditions[].uri` must be paths visible inside the Modal SGLang container. The current local WebUI supports text/JSON submission and downloads completed MP4 files into `outputs/`.
