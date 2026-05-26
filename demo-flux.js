module.exports = {
  run: [
    {
      method: "notify",
      params: {
        html: "Running Flux text-to-image demo with PiD 4x decode. Outputs go to results/demo-flux/."
      }
    },
    {
      when: "{{!exists('app/checkpoints/ae.safetensors')}}",
      method: "script.start",
      params: {
        uri: "download.js"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          HF_HUB_ENABLE_HF_TRANSFER: "1",
          PYTHONUTF8: "1"
        },
        message: [
          "PYTHONPATH=. python -m pid._src.inference.from_ldm_flux --prompt \"A photorealistic half-body portrait of a brown tabby cat with bold stripes sitting attentively on a rustic wooden kitchen table, soft morning light, detailed fur, photorealistic\" --ldm_inference_steps 28 --save_xt_steps 24 --output_dir ../results/demo-flux --cfg_scale 1 --pid_inference_steps 4 --scale 4"
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "Done. Open the results/demo-flux folder to view side-by-side VAE vs PiD outputs."
      }
    },
    {
      method: "fs.open",
      params: {
        path: "results/demo-flux"
      }
    }
  ]
}
