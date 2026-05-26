module.exports = {
  run: [
    {
      method: "notify",
      params: {
        html: "Installing NVIDIA PiD (Pixel Diffusion Decoder)..."
      }
    },
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/nv-tlabs/PiD.git app || git -C app pull"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install \"diffusers>=0.37.0\" \"transformers>=4.57\" safetensors sentencepiece huggingface_hub gradio",
          "uv pip install numpy pandas pillow imageio opencv-python-headless einops",
          "uv pip install hydra-core omegaconf pyyaml attrs",
          "uv pip install loguru termcolor fvcore iopath wandb packaging",
          "uv pip install boto3 botocore",
          "uv pip install -e ."
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python verify_env.py"
        ]
      }
    },
    {
      method: "fs.link",
      params: {
        venv: "env"
      }
    },
    {
      method: "notify",
      params: {
        html: "Installed. Download checkpoints next, then Start the Gradio UI. Z-Image weights download on first run."
      }
    }
  ]
}
