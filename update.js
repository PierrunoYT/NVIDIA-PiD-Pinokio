module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: [
          "git -C app pull"
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
      method: "notify",
      params: {
        html: "Updated PiD source and Python dependencies."
      }
    }
  ]
}
