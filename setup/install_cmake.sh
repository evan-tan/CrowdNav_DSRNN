#!/usr/bin/env bash

. ./ENV_VARS.sh
printf "${RED}Start installing CMake${NC}\n"

sudo apt-get install wget -y

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null

sudo sh -c 'echo "deb https://apt.kitware.com/ubuntu/ bionic main" > /etc/apt/sources.list.d/apt_kitware_com_ubuntu.list'

sudo apt-get update && \
sudo apt-get install kitware-archive-keyring && \
sudo rm /etc/apt/trusted.gpg.d/kitware.gpg

sudo apt-get update && \
sudo apt-get install -y cmake
printf "${RED}Finished installing CMake${NC}\n"
