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

import os
import re

import cv2
from threadpoolctl import threadpool_limits

from heregoes import exceptions
from heregoes.goesr import ABIObject, SUVIObject


def load(nc_file):
    if bool(re.search("abi-l1b", str(nc_file).lower())):
        return ABIObject(nc_file)

    elif bool(re.search("suvi-l1b", str(nc_file).lower())):
        return SUVIObject(nc_file)

    else:
        raise exceptions.HereGOESUnsupportedProductException(
            caller=__name__, filepath=nc_file
        )


NUM_CPUS = 1
if os.getenv("HEREGOES_ENV_PARALLEL", "False").lower() == "true":
    if os.getenv("HEREGOES_ENV_NUM_CPUS"):
        NUM_CPUS = int(os.getenv("HEREGOES_ENV_NUM_CPUS"))

    else:
        NUM_CPUS = len(os.sched_getaffinity(0))

cv2.setNumThreads(NUM_CPUS)
threadpool_limits(limits=NUM_CPUS, user_api=None)
