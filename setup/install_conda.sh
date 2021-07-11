#!/usr/bin/env bash

. ./ENV_VARS.sh
printf "${RED}Start installing Miniconmda${NC}\n"

# Python 3.6 specific
# file_name="Anaconda3-5.2.0-Linux-x86_64.sh"
file_name="Miniconda3-4.5.4-Linux-x86_64.sh"

sudo apt-get install wget -y

wget https://repo.anaconda.com/miniconda/$file_name && \
bash $file_name && \
rm $file_name

printf "${RED}Finished installing Miniconda${NC}\n"
