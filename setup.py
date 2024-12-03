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
        "numpy==1.23.3",
        "numba==0.56.4",
        "scipy>=1.2.3",
        "mujoco==3.2.6",
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
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="RoboCasa: Large-Scale Simulation of Household Tasks for Generalist Robots",
    author="Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, Yuke Zhu",
    url="https://github.com/robocasa/robocasa",
    author_email="soroush@cs.utexas.edu",
    version="0.2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
