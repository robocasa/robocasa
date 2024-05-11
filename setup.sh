#!/usr/bin/env bash
# ------------------
# script to setup repo, download assets, and setup macros
# ------------------
pip install -e .
conda install -c numba numba -y
python robocasa/scripts/download_kitchen_assets.py
python robocasa/scripts/setup_macros.py
