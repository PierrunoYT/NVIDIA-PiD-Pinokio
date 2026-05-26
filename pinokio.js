module.exports = {
  version: "5.0",
  title: "PiD",
  description: "NVIDIA PiD — Pixel Diffusion Decoder. Z-Image generation with 4× PiD super-resolution. Weights: nvidia/PiD (NSCLv1, non-commercial).",
  menu: async (kernel, info) => {
    let installed = info.exists("env") && info.exists("app/verify_env.py")
    let checkpoints = info.exists("app/checkpoints/ae.safetensors")
    let running = {
      install: info.running("install.js"),
      download: info.running("download.js"),
      start: info.running("start.js"),
      demo: info.running("demo-flux.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js"),
      link: info.running("link.js")
    }
    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js",
      }]
    } else if (installed) {
      if (running.start) {
        let local = info.local("start.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Web UI",
            href: local.url,
          }, {
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: "start.js",
          }]
        } else {
          return [{
            default: true,
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: "start.js",
          }]
        }
      } else if (running.download) {
        return [{
          default: true,
          icon: "fa-solid fa-cloud-arrow-down",
          text: "Downloading Checkpoints",
          href: "download.js",
        }]
      } else if (running.demo) {
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Running Flux Demo",
          href: "demo-flux.js",
        }]
      } else if (running.update) {
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Updating",
          href: "update.js",
        }]
      } else if (running.reset) {
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Resetting",
          href: "reset.js",
        }]
      } else if (running.link) {
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Deduplicating",
          href: "link.js",
        }]
      } else {
        let menu = []
        if (checkpoints) {
          menu.push({
            default: true,
            icon: "fa-solid fa-power-off",
            text: "Start",
            href: "start.js",
          })
        } else {
          menu.push({
            default: true,
            icon: "fa-solid fa-cloud-arrow-down",
            text: "Download Checkpoints",
            href: "download.js",
          })
          menu.push({
            icon: "fa-solid fa-power-off",
            text: "Start",
            href: "start.js",
          })
        }
        menu.push({
          icon: "fa-solid fa-wand-magic-sparkles",
          text: "<div><strong>Flux CLI Demo</strong><div>Official from_ldm_flux script</div></div>",
          href: "demo-flux.js",
        })
        menu.push(
          {
            icon: "fa-solid fa-plug",
            text: "Update",
            href: "update.js",
          },
          {
            icon: "fa-solid fa-plug",
            text: "Install",
            href: "install.js",
          },
          {
            icon: "fa-solid fa-file-zipper",
            text: "<div><strong>Save Disk Space</strong><div>Deduplicates redundant library files</div></div>",
            href: "link.js",
          },
          {
            icon: "fa-regular fa-circle-xmark",
            text: "<div><strong>Reset</strong><div>Remove virtual environment</div></div>",
            href: "reset.js",
            confirm: "Are you sure you wish to reset the app?"
          }
        )
        return menu
      }
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      }]
    }
  }
}
