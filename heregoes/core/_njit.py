# Copyright (c) 2023-2025.

# Author(s):

#   R. Dove <admin@wx-star.com>
#   An early version of heregoes runs at Here GOES Radiotelescope
#   (Dove & Neilson, 2020) <heregoesradio.com>

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

"""Stores preferred definitions for Numba njit compiler decorators and handles Numba parallelism"""

import warnings
from pathlib import Path

import numba
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from heregoes.core import NUM_CPUS, PARALLEL_MODE

# warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
# warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

SCRIPT_PATH = Path(__file__).parent.resolve()


def _parallel_state_difference(runtime_state):
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


numba.set_num_threads(NUM_CPUS)

NUMBA_CACHE_DIR = SCRIPT_PATH.joinpath("__numba_cache__")
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# if the parallel mode has changed since the last initialization, clear the Numba cache
if _parallel_state_difference(PARALLEL_MODE):
    for child in NUMBA_CACHE_DIR.glob("*/*"):
        if child.is_file():
            if child.suffix == ".nbc" or child.suffix == ".nbi":
                child.unlink()

numba.config.CACHE_DIR = NUMBA_CACHE_DIR

heregoes_njit = numba.njit(cache=True, fastmath=False, parallel=PARALLEL_MODE)
heregoes_njit_noparallel = numba.njit(cache=True, fastmath=False, parallel=False)
