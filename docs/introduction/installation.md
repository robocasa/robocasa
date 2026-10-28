<script>
  document.addEventListener("DOMContentLoaded", () => {
    document.body.classList.add("rc-installation-page");
  });
</script>

# Upstream installation reference

For **Unitree G1 with SONIC**, start with the dedicated
[G1 installation guide](../g1/installation.md). It specifies the compatible
simulation dependencies, G1 assets, and SONIC decoder for this checkout.

The steps below are retained as a general upstream RoboCasa reference. Their
default-branch checkouts do not specify the G1 dependency versions. They use the
[Anaconda](https://www.anaconda.com/) package management system:
1. Set up conda environment:

   ```sh
   conda create -c conda-forge -n robocasa python=3.11
   ```
2. Activate conda environment:
   ```sh
   conda activate robocasa
   ```
3. Clone and setup robosuite dependency:

   <div class="admonition warning">
   <p class="admonition-title">Upstream reference</p>
   This command clones the upstream robosuite default branch. For Unitree G1,
   use the compatible fork and revision in the dedicated G1 installation guide
   linked above.
   </div>
   
   ```sh
   git clone https://github.com/ARISE-Initiative/robosuite
   cd robosuite
   pip install -e .
   ```
4. Clone and setup this repo:

   ```sh
   cd ..
   git clone https://github.com/robocasa/robocasa
   cd robocasa
   pip install -e .
   ```
   <div class="admonition note">
   <p class="admonition-title">Installation troubleshooting</p>

   If you encounter issues with installing the numpy or numba dependencies, please try installing through conda instead: `conda install -c numba numba=0.56.4 -y`

   </div>

5. Install the package and download assets:
   ```sh
   python -m robocasa.scripts.setup_macros              # Set up system variables.
   python -m robocasa.scripts.download_kitchen_assets   # Caution: Assets to be downloaded are around 10GB.
   ```
