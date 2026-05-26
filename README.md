# PiD — Pixel Diffusion Decoder (Pinokio Launcher)

One-click Pinokio launcher for [NVIDIA PiD](https://github.com/nv-tlabs/PiD): a plug-and-play diffusion decoder that replaces VAE/RAE decoders and turns latent representations into super-resolved pixels in a single pass.

- **Paper:** [arXiv:2605.23902](https://arxiv.org/abs/2605.23902)
- **Project page:** [research.nvidia.com/labs/sil/projects/pid](https://research.nvidia.com/labs/sil/projects/pid/)
- **Model weights:** [huggingface.co/nvidia/PiD](https://huggingface.co/nvidia/PiD) (NSCLv1 — non-commercial research/evaluation only)

## What this launcher provides

1. **Gradio Web UI** — Z-Image text-to-image with live previews, then PiD 4× super-resolution. Compare native VAE vs PiD with a slider.
2. **Flux CLI demo** — Runs the official `from_ldm_flux` inference script and saves side-by-side outputs under `results/demo-flux/`.
3. **Checkpoint download** — Pulls Flux/Z-Image-compatible PiD weights from Hugging Face.

## Requirements

- **NVIDIA GPU** with CUDA (roughly 13 GB+ VRAM for 512→2048; more for 1024→4K)
- **Pinokio** with `uv`, `git`, and `hf` CLI
- Disk space for PiD checkpoints and Hugging Face model caches

## How to use

1. Click **Install** — clones [nv-tlabs/PiD](https://github.com/nv-tlabs/PiD) into `app/`, creates a Python venv, installs PyTorch and dependencies, and runs `verify_env.py`.
2. Click **Download Checkpoints** — downloads Flux-compatible PiD weights into `app/checkpoints/`.
3. Click **Start** — launches the Gradio UI at `http://127.0.0.1:<port>`.
4. Optional: **Flux CLI Demo** — batch inference without the web UI.

On first **Start**, Z-Image weights are fetched from Hugging Face automatically.

## CLI usage (advanced)

After install, you can run any upstream script from a Pinokio terminal:

```bash
cd app
PYTHONPATH=. python -m pid._src.inference.from_ldm_flux \
  --prompt "A photorealistic cat" \
  --ldm_inference_steps 28 --save_xt_steps 24 \
  --output_dir ../results/demo \
  --cfg_scale 1 --pid_inference_steps 4 --scale 4
```

For 4K decoding:

```bash
PYTHONPATH=. python -m pid._src.inference.from_ldm_flux \
  --prompt "A photorealistic cat" \
  --resolution 1024 --pid_ckpt_type 2kto4k \
  --ldm_inference_steps 28 --save_xt_steps 24 \
  --output_dir ../results/demo_4k \
  --cfg_scale 1 --pid_inference_steps 4 --scale 4
```

See the [PiD README](https://github.com/nv-tlabs/PiD) for all backbones (`flux2`, `sd3`, `zimage`, `dinov2`, `siglip`, etc.).

## Programmatic access

### Python (Gradio UI server)

The launcher starts `gradio_app.py`, which exposes a Gradio app on localhost. Once running, use the Gradio client or HTTP API:

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860")
result = client.predict(
    "A photorealistic cat on a wooden table",
    28,   # num_inference_steps
    5.0,  # guidance_scale
    0,    # seed
    512,  # resolution
    True, # randomize_seed
)
```

### Python (upstream PiD API)

```python
import subprocess

subprocess.run([
    "python", "-m", "pid._src.inference.from_ldm_flux",
    "--prompt", "A photorealistic cat",
    "--ldm_inference_steps", "28",
    "--save_xt_steps", "24",
    "--output_dir", "./results/demo",
    "--cfg_scale", "1",
    "--pid_inference_steps", "4",
    "--scale", "4",
], cwd="app", env={"PYTHONPATH": "."})
```

### cURL (Gradio HTTP API)

After the Web UI is running, list routes:

```bash
curl http://127.0.0.1:7860/info
```

Then POST to the documented prediction endpoint (route name depends on Gradio version).

## Launcher scripts

| Script | Purpose |
|--------|---------|
| `install.js` | Clone PiD, install deps, verify environment |
| `download.js` | Download Flux/Z-Image PiD checkpoints |
| `start.js` | Launch Gradio UI |
| `demo-flux.js` | Run official Flux CLI demo |
| `update.js` | Pull latest PiD source and refresh deps |
| `reset.js` | Remove `env/` virtual environment |
| `link.js` | Deduplicate venv packages to save disk |
| `torch.js` | Cross-platform PyTorch install helper |

## License

- **PiD source code:** Apache 2.0
- **PiD model weights:** NSCLv1 (non-commercial)
- **This launcher:** follow the repository license
