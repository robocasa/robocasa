<script>
  document.addEventListener("DOMContentLoaded", () => {
    document.body.classList.add("rc-installation-page");
  });
</script>

# Installation

RoboCasa works across all major computing platforms. The easiest way to set up is through the [Anaconda](https://www.anaconda.com/) package management system. Follow the instructions below to install:
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
   <p class="admonition-title">Important!</p>
   The latest version of RoboCasa uses the master branch of robosuite
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
   python robocasa/scripts/download_kitchen_assets.py   # Caution: Assets to be downloaded are around 5GB.
   python robocasa/scripts/setup_macros.py              # Set up system variables.
   ```

