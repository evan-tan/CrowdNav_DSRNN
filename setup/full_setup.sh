#!/usr/bin/env bash

# main folder where all repos will be cloned into
REPOS_PATH="$HOME/repos/fyp_repos"
# conda environment name
ENV_NAME="dsrnn"
# conda python version, minimum version 3.6
PYTHON_VER="3.8"
CWD="$(pwd)"

# miniconda 4.5.4 to prevent breaking on Ubuntu 18.04
sh ./install_conda.sh
sh ./install_cmake.sh

# $REPOS_PATH/
# ├── baselines
# ├── CrowdNav_DSRNN
# ├── PythonRobotics
# ├── Python-RVO2
# └── socialforce

if [ -d "$REPOS_PATH" ]; then
    echo "$REPOS_PATH already exists!"
else
    mkdir "$REPOS_PATH"
fi

# clone all repos
cd "$REPOS_PATH"
git clone git@github.com:evan-tan/CrowdNav_DSRNN.git
git clone https://github.com/sybrenstuvel/Python-RVO2.git
git clone https://github.com/openai/baselines.git
git clone https://github.com/ChanganVR/socialforce.git

sudo apt-get install -y python3-tk

alias pip3="python3 -m pip"

#### Create conda environment ####
source $HOME/miniconda3/etc/profile.d/conda.sh
conda create -n $ENV_NAME python=$PYTHON_VER -y &&
    #### ACTIVATE CONDA ENVIRONMENT ####
    conda activate $ENV_NAME

# install rvo2
cd $REPOS_PATH/Python-RVO2 &&
    pip3 install cython && # old cython version will break
    python3 setup.py build &&
    python3 setup.py install &&
    pip3 install -e .

# install openai baselines
cd $REPOS_PATH/baselines &&
    sudo apt-get update &&
    sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev &&
    pip3 install -e .

# install everything else for DSRNN
cd $REPOS_PATH/CrowdNav_DSRNN &&
    pip3 install -r requirements.txt &&
    pip3 install -e .

# # install socialforce (UNUSED)
# cd $REPOS_PATH/socialforce &&
#     pip3 install -e '.[test,plot]'
