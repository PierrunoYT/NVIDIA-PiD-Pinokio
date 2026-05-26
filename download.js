module.exports = {
  run: [
    {
      method: "notify",
      params: {
        html: "Downloading PiD Flux / Z-Image checkpoints from Hugging Face. This can take several minutes."
      }
    },
    {
      method: "shell.run",
      params: {
        path: "app",
        env: {
          HF_HUB_ENABLE_HF_TRANSFER: "1",
          HF_HUB_DOWNLOAD_TIMEOUT: "300"
        },
        message: [
          "hf download nvidia/PiD --local-dir . --include \"checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/*\" --include \"checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/*\" --include \"checkpoints/ae.safetensors\""
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "Checkpoints downloaded. Click Start to launch the Gradio UI."
      }
    }
  ]
}
