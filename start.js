module.exports = {
  daemon: true,
  run: [
    {
      when: "{{!exists('app/checkpoints/ae.safetensors')}}",
      method: "script.start",
      params: {
        uri: "download.js"
      }
    },
    {
      method: "notify",
      params: {
        html: "Starting PiD Gradio UI. First launch loads Z-Image and PiD decoders into VRAM."
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        env: {
          HF_HUB_ENABLE_HF_TRANSFER: "1",
          HF_HUB_DOWNLOAD_TIMEOUT: "300",
          PYTHONUTF8: "1",
          PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
        },
        message: [
          "python gradio_app.py --host 127.0.0.1 --port {{port}}"
        ],
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    },
    {
      method: "notify",
      params: {
        html: "PiD is running. Generate with Z-Image, then compare native VAE vs PiD 4x upscale."
      }
    }
  ]
}
