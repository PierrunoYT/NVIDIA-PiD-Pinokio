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
      method: "notify",
      params: {
        html: "Updated PiD source and Python dependencies."
      }
    }
  ]
}
