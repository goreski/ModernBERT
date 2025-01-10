#!/bin/bash
# Run: source setup_env.sh

# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run the installer
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda

# Initialize Miniconda
source $HOME/miniconda/bin/activate

# Add Miniconda to the PATH
export PATH="$HOME/miniconda/bin:$PATH"

# Verify the installation
conda --version

# Delete the Miniconda installer
rm Miniconda3-latest-Linux-x86_64*

# Create a new conda environment
conda env create -f environment.yaml

# Activate the new environment
conda activate bert24

# Install Flash attention 2
pip install "flash_attn==2.6.3" --no-build-isolation

# Git configur name and email
git config --global user.name "goreski"
git config --global user.email "goran.oreski@gmail.com"

