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
        "numpy==2.2.5",
        "numba==0.61.2",
        "scipy==1.15.3",
        "mujoco==3.3.1",
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
        "tianshou==0.4.10",
        "lerobot==0.3.3",
        "gymnasium",
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots",
    author="Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri, Yuke Zhu",
    url="https://github.com/robocasa/robocasa",
    author_email="soroush@cs.utexas.edu",
    version="1.0.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
