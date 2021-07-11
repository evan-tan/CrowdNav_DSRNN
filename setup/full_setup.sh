#!/usr/bin/env bash

# main folder where all repos will be cloned into
REPOS_PATH="$HOME/repos/fyp_repos"
# conda environment name
ENV_NAME="dsrnn"
# conda python version, minimum version 3.6
PYTHON_VER="3.6"
CWD="$(pwd)"

#### BASE OF ANY ENVIRONMENT ####
cd $CWD
sh ./install_cmake.sh
sh ./install_conda.sh

if [ -d "$REPOS_PATH" ]; then
    echo "$REPOS_PATH already exists!"
else
    mkdir $REPOS_PATH
fi

# clone all repos
cd $REPOS_PATH && \
git clone git@github.com:evan-tan/CrowdNav_DSRNN.git
git clone https://github.com/sybrenstuvel/Python-RVO2.git
git clone https://github.com/openai/baselines.git
git clone https://github.com/ChanganVR/socialforce.git


sudo apt-get install -y python3-tk

alias pip3="python3 -m pip"

#### Create conda environment ####
source $HOME/miniconda3/etc/profile.d/conda.sh
conda create -n $ENV_NAME python=$PYTHON_VER -y && \

#### ACTIVATE CONDA ENVIRONMENT ####
conda activate $ENV_NAME

# install rvo2
cd $REPOS_PATH/Python-RVO2 && \
pip3 install cython && \
python3 setup.py build_ext --inplace && \
pip3 install -e .

# install pytorch
conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

# install socialforce
cd $REPOS_PATH/socialforce && \
pip3 install -e '.[test,plot]'

# install openai baselines
cd $REPOS_PATH/baselines && \
sudo apt-get update && \
sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
conda install -y tensorflow cudatoolkit=11.0 -c anaconda && \
pip3 install -e .

# install everything else for DSRNN
cd $REPOS_PATH/CrowdNav_DSRNN && \
pip3 install -r requirements.txt && \
pip3 install -e .

# sphinx documentation
pip3 install sphinx sphinx_rtd_theme

# PyTorch patches
# patch -p0 $PATCH_PATH/rnn.py < GRU.patch
# patch -p0 $PATCH_PATH/linear.py < Linear.patch
# patch -p0 $PATCH_PATH/activation.py < ReLU.patch
# patch -p0 $PATCH_PATH/activation.py < Tanh.patch
# patch -p0 $PATCH_PATH/container.py < ModuleList.patch
# patch -p0 $PATCH_PATH/container.py < Sequential.patch
