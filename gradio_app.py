"""Local Gradio UI for Z-Image + PiD, adapted from multimodalart/pid for Pinokio."""

import argparse
import os
import random
import sys
import threading
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
if not APP_DIR.exists():
    raise SystemExit("PiD source not found. Run Install first.")

os.chdir(APP_DIR)
sys.path.insert(0, str(APP_DIR))

import gradio as gr
import numpy as np
import torch
from PIL import Image

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU required. PiD inference does not run on CPU.")

from diffusers import AutoencoderTiny
from huggingface_hub import snapshot_download
from pid._src.inference.checkpoint_registry import get_pid_checkpoint
from pid._src.inference.pipeline_registry import (
    decode_with_pipeline_vae,
    extract_latent,
    load_pipeline,
)
from pid._src.utils.model_loader import load_model_from_checkpoint

DTYPE = torch.bfloat16
BACKBONE = "zimage"
SR_SCALE = 4
PID_INFERENCE_STEPS = 4


def ensure_checkpoints():
    snapshot_download(
        repo_id="nvidia/PiD",
        local_dir=str(APP_DIR),
        allow_patterns=[
            "checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/*",
            "checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/*",
            "checkpoints/ae.safetensors",
        ],
    )


def load_models():
    ensure_checkpoints()

    print("[pid] loading Z-Image pipeline...", flush=True)
    pipeline, pipe_cfg = load_pipeline(BACKBONE, dtype=DTYPE)
    pipeline.to("cuda")

    print("[pid] loading TAEF1 preview decoder...", flush=True)
    taef1 = AutoencoderTiny.from_pretrained(
        "madebyollin/taef1", torch_dtype=DTYPE, low_cpu_mem_usage=False
    ).to("cuda")
    taef1.eval()

    def _load_pid(ckpt_type: str):
        meta = get_pid_checkpoint(BACKBONE, ckpt_type)
        print(f"[pid] loading PiD decoder ({ckpt_type})...", flush=True)
        model, _ = load_model_from_checkpoint(
            experiment_name=meta.experiment,
            checkpoint_path=meta.checkpoint_path,
            config_file="pid/_src/configs/pid/config.py",
            enable_fsdp=False,
            strict=False,
        )
        model.eval()
        return model

    pid_models = {
        "2k": _load_pid("2k"),
        "2kto4k": _load_pid("2kto4k"),
    }
    print("[pid] ready", flush=True)
    return pipeline, pipe_cfg, taef1, pid_models


PIPELINE, PIPE_CFG, TAEF1, PID_MODELS = load_models()


def _pick_pid_model(resolution: int):
    return PID_MODELS["2kto4k"] if resolution > 512 else PID_MODELS["2k"]


def _taef1_preview(packed_latent: torch.Tensor, height: int, width: int) -> Image.Image:
    with torch.no_grad():
        unpacked = extract_latent(PIPELINE, SimpleNamespace(images=packed_latent), PIPE_CFG, height, width)
        scale = PIPELINE.vae.config.scaling_factor
        shift = getattr(PIPELINE.vae.config, "shift_factor", None) or 0.0
        denorm = unpacked.to(dtype=DTYPE) / scale + shift
        img = TAEF1.decode(denorm).sample
        img = (img.float().clamp(-1, 1) + 1) / 2
        arr = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)


def _pid_pixel_to_pil(x: torch.Tensor) -> Image.Image:
    arr = ((x[0].float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(arr)


def _pid_stream(pid_model, latent, baseline_01, sigma, caption, num_steps=PID_INFERENCE_STEPS):
    from contextlib import nullcontext

    caption_embs, _ = pid_model._encode_text_raw([caption])
    caption_embs = caption_embs.to(**pid_model.tensor_kwargs)

    lq_h, lq_w = baseline_01.shape[-2], baseline_01.shape[-1]
    img_h, img_w = lq_h * SR_SCALE, lq_w * SR_SCALE

    lq_video_or_image = (baseline_01 * 2.0 - 1.0).to(dtype=DTYPE, device="cuda")
    lq_latent = latent.to(dtype=DTYPE, device="cuda")
    degrade_sigma_tensor = torch.tensor([sigma], device="cuda", dtype=torch.float32)

    gen = torch.Generator(device="cuda").manual_seed(0)
    noise = torch.randn(1, 3, img_h, img_w, device="cuda", generator=gen)

    t_list = pid_model._get_t_list(device=torch.device("cuda"), num_steps=num_steps)
    autocast_ctx = (
        torch.autocast("cuda", dtype=pid_model.autocast_dtype)
        if pid_model.autocast_dtype
        else nullcontext()
    )
    net = pid_model.net
    net.eval()
    timescale = pid_model.fm_trainer.timescale
    student_sample_type = pid_model.config.student_sample_type
    prediction_type = pid_model.config.prediction_type

    x = noise
    with torch.no_grad(), autocast_ctx:
        steps_total = len(t_list) - 1
        for step_idx, (t_cur, t_next) in enumerate(zip(t_list[:-1], t_list[1:])):
            t_cur_batch = t_cur.expand(1)
            t_cur_scaled = t_cur_batch * timescale
            v_pred = net(
                x,
                t_cur_scaled,
                caption_embs,
                lq_video_or_image=lq_video_or_image,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma_tensor,
            )
            if t_next.item() > 0:
                if student_sample_type == "ode":
                    v_for_step = pid_model._net_output_to_velocity(x, v_pred, t_cur_batch, prediction_type)
                    dt = t_next - t_cur
                    x = x + dt * v_for_step
                else:
                    x0_pred = pid_model._velocity_to_x0(x, v_pred, t_cur_batch)
                    eps_infer = torch.randn(
                        x0_pred.shape, device=x0_pred.device, dtype=x0_pred.dtype, generator=gen
                    )
                    s = [1] + [1] * (x.ndim - 1)
                    t_next_bcast = t_next.reshape(1).expand(s)
                    x = (1.0 - t_next_bcast) * x0_pred + t_next_bcast * eps_infer
            else:
                x = pid_model._velocity_to_x0(x, v_pred, t_cur_batch)
            yield step_idx + 1, steps_total, x.clone()


def generate(
    prompt,
    num_inference_steps,
    guidance_scale,
    seed,
    resolution,
    randomize_seed,
):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    if randomize_seed:
        seed = random.randint(0, 2**31 - 1)
    seed = int(seed)
    num_inference_steps = int(num_inference_steps)
    height = width = int(resolution)

    yield gr.update(visible=True, value=None, label="Generating Z-Image…"), gr.update(visible=False, value=None), gr.update(value=seed)

    preview_q = Queue()
    done = object()

    def streaming_cb(pipe, step_index, timestep, callback_kwargs):
        try:
            preview = _taef1_preview(callback_kwargs["latents"], height, width)
            preview_q.put((step_index, preview))
        except Exception as exc:
            print(f"[pid] taef1 preview failed at step {step_index}: {exc}", flush=True)
        return callback_kwargs

    def run_pipeline():
        gen_torch = torch.Generator(device="cuda").manual_seed(seed)
        gen_kwargs = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=float(guidance_scale),
            num_images_per_prompt=1,
            output_type="latent",
            generator=gen_torch,
            callback_on_step_end=streaming_cb,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        gen_kwargs.update(PIPE_CFG.extra_generate_kwargs)
        try:
            with torch.no_grad():
                out = PIPELINE(**gen_kwargs)
            preview_q.put((done, out))
        except Exception as exc:
            preview_q.put((done, exc))

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    raw_output = None
    while True:
        step_index, payload = preview_q.get()
        if step_index is done:
            if isinstance(payload, Exception):
                raise payload
            raw_output = payload
            break
        label = f"Generating Z-Image — step {step_index + 1}/{num_inference_steps}"
        yield gr.update(visible=True, value=payload, label=label), gr.update(visible=False), gr.update()

    thread.join()
    final_latent = extract_latent(PIPELINE, raw_output, PIPE_CFG, height, width)

    yield gr.update(visible=True, label="Decoding final Z-Image…"), gr.update(visible=False), gr.update()
    with torch.no_grad():
        baseline_01 = decode_with_pipeline_vae(PIPELINE, final_latent, PIPE_CFG)
        zimage_img = Image.fromarray(
            (baseline_01[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
        )

    torch.cuda.empty_cache()

    final_sigma = float(PIPELINE.scheduler.sigmas[-1].item())
    pid_img = None
    pid_model = _pick_pid_model(height)
    for step, total, x in _pid_stream(pid_model, final_latent, baseline_01, final_sigma, prompt):
        pid_img = _pid_pixel_to_pil(x)
        yield (
            gr.update(visible=True, value=pid_img, label=f"Upscaling with PiD — step {step}/{total}"),
            gr.update(visible=False),
            gr.update(),
        )

    yield (
        gr.update(visible=False, value=None),
        gr.update(visible=True, value=(zimage_img, pid_img)),
        gr.update(),
    )


DESCRIPTION = """
# PiD — Pixel Diffusion Decoder for Z-Image

Runs [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image), then
[NVIDIA PiD](https://github.com/nv-tlabs/PiD)'s 4-step pixel-diffusion decoder for a 4×
super-resolved result. Drag the slider to compare the native VAE output to the PiD upscale.
"""

CSS = """
.gradio-container { max-width: 1200px !important; margin: auto !important; }
"""

with gr.Blocks(title="PiD", css=CSS) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        prompt = gr.Textbox(
            show_label=False,
            placeholder="Describe what you want to generate…",
            value="A photorealistic close-up of a brown tabby cat wearing a woolen hat sitting on a rustic wooden table, morning light, detailed fur",
            lines=1,
            scale=4,
            container=False,
        )
        run = gr.Button("Run", variant="primary", scale=1)

    live_preview = gr.Image(label="Z-Image with PiD", visible=True, type="pil", height=720)
    slider = gr.ImageSlider(
        label="Z-Image (left) ↔ PiD 4× upscale (right)",
        visible=False,
        type="pil",
        height=720,
        max_height=720,
    )

    with gr.Accordion("Advanced settings", open=False):
        with gr.Row():
            resolution = gr.Radio(
                label="Z-Image resolution",
                choices=[512, 1024],
                value=512,
                info="512 → 2048² (PiD 2k); 1024 → 4096² (PiD 2kto4k)",
            )
            num_inference_steps = gr.Slider(label="Z-Image steps", minimum=8, maximum=50, step=1, value=28)
        with gr.Row():
            guidance_scale = gr.Slider(label="Guidance", minimum=1.0, maximum=10.0, step=0.5, value=5.0)
            seed = gr.Number(label="Seed", value=0, precision=0)
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

    run.click(
        fn=generate,
        inputs=[prompt, num_inference_steps, guidance_scale, seed, resolution, randomize_seed],
        outputs=[live_preview, slider, seed],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo.queue().launch(server_name=args.host, server_port=args.port)
