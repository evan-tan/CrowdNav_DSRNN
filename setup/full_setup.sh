#!/usr/bin/env bash

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh

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
if [ -d "$REPOS_PATH/CrowdNav_DSRNN" ]; then
    echo "$REPOS_PATH/CrowdNav_DSRNN already exists!"
else
    git clone git@github.com:evan-tan/CrowdNav_DSRNN.git
fi
if [ -d "$REPOS_PATH/Python-RVO2" ]; then
    echo "$REPOS_PATH/Python-RVO2 already exists!"
else
    git clone https://github.com/sybrenstuvel/Python-RVO2.git
fi
if [ -d "$REPOS_PATH/baselines" ]; then
    echo "$REPOS_PATH/baselines already exists!"
else
    git clone https://github.com/openai/baselines.git
fi
if [ -d "$REPOS_PATH/socialforce" ]; then
    echo "$REPOS_PATH/socialforce already exists!"
else
    git clone https://github.com/ChanganVR/socialforce.git
fi

sudo apt-get install -y python3-tk

alias pip3="python3 -m pip"

#### Create conda environment ####
conda create -n $ENV_NAME python=$PYTHON_VER -y &&
    #### ACTIVATE CONDA ENVIRONMENT ####
    conda activate $ENV_NAME

# install cuspatial
# dependencies [cudatoolkit, cudf, rmm]
conda install -y -c anaconda cudatoolkit=11.0
conda install -y -c conda-forge -c rapidsai cudf rmm cudatoolkit=11.0
conda install -y -c conda-forge -c rapidsai cuspatial

# install rvo2
cd $REPOS_PATH/Python-RVO2 &&
    # DO NOT FOLLOW README and install old cython, will break
    conda install -y -c conda-forge cython &&
    python3 setup.py build &&
    python3 setup.py install &&
    pip3 install -e .

# install everything else for DSRNN
cd $REPOS_PATH/CrowdNav_DSRNN
# split install into multiple commands so ...
# solving environment doesn't take forever
pip install tensorflow tensorboard
conda install -y -c conda-forge gym numpy pandas
conda install -y -c conda-forge matplotlib shapely
conda install -y -c conda-forge scipy scikit-image
conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

# install openai baselines
cd $REPOS_PATH/baselines &&
    sudo apt-get update &&
    sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev &&
    pip3 install -e .

# # install socialforce (UNUSED)
# cd $REPOS_PATH/socialforce &&
#     pip3 install -e '.[test,plot]'

# go back to cwd
cd $CWD
