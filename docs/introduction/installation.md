
# Installation

RoboCasa works across all major computing platforms. The easiest way to set up is through the [Anaconda](https://www.anaconda.com/) package management system. Follow the instructions below to install:
1. Set up conda environment:

   ```sh
   conda create -c conda-forge -n robocasa python=3.9
   ```
2. Activate conda environment:
   ```sh
   conda activate robocasa
   ```
3. Clone and setup robosuite dependency:

   <div class="admonition warning">
   <p class="admonition-title">Important!</p>
   Use the robocasa_v0.1 branch of robosuite!
   </div>
   
   ```sh
   git clone https://github.com/ARISE-Initiative/robosuite -b robocasa_v0.1
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
5. Install the package and download assets:
   ```sh
   conda install -c numba numba -y
   python robocasa/scripts/download_kitchen_assets.py   # Caution: Assets to be downloaded are around 5GB.
   python robocasa/scripts/setup_macros.py              # Set up system variables.
   ```

