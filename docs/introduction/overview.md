<!-- Add class immediately so secondary sidebar is hidden from first paint (no flash) -->
<script>document.body.classList.add("rc-overview-page");</script>
<script>
  document.addEventListener("DOMContentLoaded", () => {
    document.body.classList.add("rc-overview-page");

    // Copy buttons for citation blocks
    const section = document.getElementById("citation");
    if (section) {
      section.querySelectorAll(".highlight-bibtex").forEach((block) => {
        const pre = block.querySelector("pre");
        if (!pre) return;
        const text = pre.textContent || "";
        const wrapper = block.querySelector(".highlight") || block;
        wrapper.style.position = "relative";
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "rc-citation-copy-btn";
        btn.setAttribute("aria-label", "Copy citation");
        btn.title = "Copy to clipboard";
        btn.innerHTML = "<svg class=\"rc-citation-copy-icon\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><rect x=\"9\" y=\"9\" width=\"13\" height=\"13\" rx=\"2\" ry=\"2\"/><path d=\"M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1\"/></svg>";
        btn.addEventListener("click", () => {
          navigator.clipboard.writeText(text).then(() => {
            const label = btn.getAttribute("aria-label");
            btn.setAttribute("aria-label", "Copied!");
            btn.classList.add("rc-copied");
            btn.innerHTML = "<svg class=\"rc-citation-copy-icon\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><polyline points=\"20 6 9 17 4 12\"/></svg>";
            setTimeout(() => {
              btn.setAttribute("aria-label", label);
              btn.classList.remove("rc-copied");
              btn.innerHTML = "<svg class=\"rc-citation-copy-icon\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><rect x=\"9\" y=\"9\" width=\"13\" height=\"13\" rx=\"2\" ry=\"2\"/><path d=\"M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1\"/></svg>";
            }, 1500);
          });
        });
        wrapper.appendChild(btn);
      });
    }
  });
</script>

# Overview

**RoboCasa** is a large-scale simulation framework for training generally capable robots to perform everyday tasks. It was [originally released](https://robocasa.ai/assets/robocasa_rss24.pdf) in 2024 by UT Austin researchers. The latest iteration, **RoboCasa365**, builds upon the original release with significant new functionalities to support large-scale training and benchmarking in sim. Four pillars underlie RoboCasa365:
- **Diverse tasks**: 365 tasks created with the guidance of large language models
- **Diverse assets**: including 2,500+ kitchen scenes and 3,200+ 3D objects
- **High-quality demonstrations**: including 600+ hours of human demonstrations in addition to 1,600+ hours of robot datasets created with automated trajectory tools
- **Benchmarking support**: popular policy learning methods including Diffusion Policy, Pi, and GR00T

<p align="center">
  <img class="overview-banner-image" style="width: 95%" src="../_static/robocasa-banner.webp" alt="RoboCasa 365 overview">
</p>

This documentation guide contains information about installation, getting started, and additional use cases such as accessing datasets, policy learning, and API docs.

### Citation

**RoboCasa365:**

```bibtex
@inproceedings{robocasa365,
  title={RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots},
  author={Soroush Nasiriany and Sepehr Nasiriany and Abhiram Maddukuri and Yuke Zhu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

**RoboCasa (Original Release):**

```bibtex
@inproceedings{robocasa2024,
  title={RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots},
  author={Soroush Nasiriany and Abhiram Maddukuri and Lance Zhang and Adeet Parikh and Aaron Lo and Abhishek Joshi and Ajay Mandlekar and Yuke Zhu},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2024}
}
```

