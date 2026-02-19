#!/bin/bash

set -e

INIT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${INIT_DIR}/../"

#format
black . --target-version=py313
isort --profile black .

#clear numba cache,
set +e
find . -type d -name '__*cache*__' -exec rm -rf {} \; > /dev/null 2>&1
set -e

#then do a "warm up"
pytest tests/

#full test with coverage
cat /dev/null > heregoes/core/.hg_parallel
export HEREGOES_ENV_NUM_CPUS=16
pytest -s --cov --cov-report html:coverage/html --cov-report xml:coverage/coverage.xml tests/
genbadge coverage -v -i coverage/coverage.xml -o coverage/coverage-badge.svg

#run demo and handle output
cd demo
./run.sh