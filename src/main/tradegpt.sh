#!/bin/bash

# Find the virtual environment directory (tradegpt)
# Starting from the current directory and moving upwards if necessary

activate_script="tradegpt/bin/activate"

# Traverse directories upwards until the tradegpt virtual environment is found
current_dir=$(pwd)
while [[ "$current_dir" != "/" ]]; do
    if [[ -f "$current_dir/$activate_script" ]]; then
        source "$current_dir/$activate_script"
        echo "Activated virtual environment: tradegpt"
        exit 0
    fi
    current_dir=$(dirname "$current_dir")
done

# If the virtual environment was not found
echo "Error: Could not find tradegpt virtual environment."
exit 1
