#!/bin/bash

INIT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

output="${INIT_DIR}/docs"

rm -rf "$output"

pdoc "${INIT_DIR}/../heregoes" \
'!heregoes.' \
heregoes.goesr \
'!heregoes.goesr.suvi' \
heregoes.image \
heregoes.load \
heregoes.navigation \
heregoes.projection \
-o "$output" \
-t "${INIT_DIR}/template" \
--no-show-source \
--no-include-undocumented
