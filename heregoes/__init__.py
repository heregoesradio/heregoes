# Copyright (c) 2020-2023.

# Author(s):

#   Harry Dove-Robinson <admin@wx-star.com>
#   for Here GOES Radiotelescope (Harry Dove-Robinson & Heidi Neilson)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import os
import time
from pathlib import Path

import cv2
import numba
from threadpoolctl import threadpool_limits

SCRIPT_PATH = Path(__file__).parent.resolve()


def parallel_state_difference(runtime_state):
    """
    Checks whether the HEREGOES_ENV_PARALLEL mode has changed from the last run of the program
    """
    state_file = SCRIPT_PATH.joinpath(".hg_parallel")

    if state_file.exists():
        with open(state_file, "r") as f:
            line = f.readline()

            if line.lower() == "true":
                file_state = True

            else:
                file_state = False

        # if the runtime state matches the file state, return False (nothing to do)
        if file_state == runtime_state:
            return False

    # if the states do not match, or the state file did not exist, write the new state and return True
    with open(state_file, "w") as f:
        file_state = runtime_state
        f.write(str(file_state))

    return True


NUMBA_PARALLEL = False
GDAL_PARALLEL = False
NUM_CPUS = 1

if os.getenv("HEREGOES_ENV_PARALLEL", "False").lower() == "true":
    NUMBA_PARALLEL = True
    GDAL_PARALLEL = True

    if os.getenv("HEREGOES_ENV_NUM_CPUS"):
        NUM_CPUS = int(os.getenv("HEREGOES_ENV_NUM_CPUS"))

    else:
        NUM_CPUS = len(os.sched_getaffinity(0))

cv2.setNumThreads(NUM_CPUS)
numba.set_num_threads(NUM_CPUS)
threadpool_limits(limits=NUM_CPUS, user_api=None)


IREMIS_DIR = os.getenv("HEREGOES_ENV_IREMIS_DIR")
if IREMIS_DIR:
    IREMIS_DIR = Path(IREMIS_DIR)


NUMBA_CACHE_DIR = SCRIPT_PATH.joinpath("__numba_cache__")
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# if the parallel mode has changed since the last initialization, clear the Numba cache
if parallel_state_difference(NUMBA_PARALLEL):
    for child in NUMBA_CACHE_DIR.glob("*/*"):
        if child.is_file():
            if child.suffix == ".nbc" or child.suffix == ".nbi":
                child.unlink()

numba.config.CACHE_DIR = NUMBA_CACHE_DIR


heregoes_njit = numba.njit(cache=True, fastmath=False, parallel=NUMBA_PARALLEL)
heregoes_njit_noparallel = numba.njit(cache=True, fastmath=False, parallel=False)

logger = logging.getLogger("heregoes-logger")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s.%(msecs)03dZ | %(levelname)s | %(module)s | %(funcName)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    style="%",
)
formatter.converter = time.gmtime
handler.setFormatter(formatter)
logger.addHandler(handler)
