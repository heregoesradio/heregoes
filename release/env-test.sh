#!/bin/bash

set -e

INIT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${INIT_DIR}/../"

source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"

while read -r yaml
do
    conda clean --all --yes --quiet

    env_name="$(basename -s '.yml' "$yaml")-$(uuidgen)"
    conda env create -f "$yaml" --name="$env_name"
    conda activate "$env_name"

    set +e
    find . -type d -name '__*cache*__' -exec rm -rf {} \; > /dev/null 2>&1
    set -e

    echo "Testing on $env_name"
    pytest tests/
    conda deactivate
    conda env remove --name="$env_name" --yes
    echo "Deleted $env_name"
done < <( find "${INIT_DIR}" -type f -name '*.yml' )