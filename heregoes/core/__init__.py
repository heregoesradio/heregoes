"""
Low-level parts of the heregoes library
"""

import os

import cv2
from threadpoolctl import threadpool_limits

# default to parallel mode and all logical CPUs
PARALLEL_MODE = True
NUM_CPUS = len(os.sched_getaffinity(0))
if os.getenv("HEREGOES_ENV_PARALLEL", "true").lower() == "false":
    PARALLEL_MODE = False
    NUM_CPUS = 1

elif os.getenv("HEREGOES_ENV_NUM_CPUS"):
    NUM_CPUS = int(os.getenv("HEREGOES_ENV_NUM_CPUS"))

cv2.setNumThreads(NUM_CPUS)
threadpool_limits(limits=NUM_CPUS, user_api=None)

from ._ncinterface import NCInterface
from ._njit import heregoes_njit, heregoes_njit_noparallel
