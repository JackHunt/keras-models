#!/usr/bin/env bash

# Check we have Python and Pip.
if [ ! command -v python3 &> /dev/null ]
then
    echo "python3 not found."
    exit 1
fi

if [ ! command -v pip3 &> /dev/null ]
then
    echo "pip3 not found."
    exit 1
fi

# Set the env name & location, defaults if not specified.
env_name=${1:-tf_env}
env_parent_dir=${2:-$(pwd)}

# Check that the parent directory for the env exists.
if [ ! -d $env_parent_dir ]
then
    echo "Specified parent directory does not exist."
    exit 1
fi

# If the env exists, prompt for it's deletion.
env_path="$env_parent_dir/$env_name"
if [ -d $env_path ]
then
    read -p "The specified environment already exists, remove it first? [Y/N]" yn
    case $yn in
        [Yy]* ) rm -rf $env_path;;
        [Nn]* ) echo "Reusing existing virtualenv.";;
    esac
fi

# Create the venv.
echo "Creating virtualenv."
python3 -m venv $env_path

# Enter the venv.
echo "Activating the virtualenv."
source $env_path/bin/activate

# Update pip.
echo "Upgrading pip."
pip3 install --upgrade pip

# Install TensorFlow.
read -p "Install TensorFlow with GPU support? [Y/N]" yn
case $yn in
    [Yy]* ) pip3 install --upgrade tensorflow-gpu;;
    [Nn]* ) pip3 install --upgrade tensorflow;;
esac