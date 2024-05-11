# read the contents of your README file
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="robocasa",
    packages=[package for package in find_packages() if package.startswith("robocasa")],
    install_requires=[
        "numpy==1.23.2",
        "numba>=0.49.1",
        "scipy>=1.2.3",
        "mujoco==3.1.1",
        "pygame",
        "Pillow",
        "opencv-python",
        "pyyaml",
        "pynput",
        "tqdm",
        "termcolor",
        "imageio",
        "h5py",
        "lxml",
        "hidapi",
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="robosuite: A Modular Simulation Framework and Benchmark for Robot Learning",
    author="Yuke Zhu, Josiah Wong, Ajay Mandlekar, Roberto Martín-Martín",
    url="https://github.com/ARISE-Initiative/robosuite",
    author_email="soroush@cs.utexas.edu",
    version="0.4.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
