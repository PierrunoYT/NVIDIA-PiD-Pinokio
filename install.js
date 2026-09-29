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
        venv_python: "3.12",
        path: "app",
        message: [
          "python --version"
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
          "uv pip install -e . gradio hf_transfer"
        ]
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
        venv: "app/env"
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
