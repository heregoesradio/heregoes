"""Low-level array methods optimized with Numba where possible"""

from . import _funcs
from ._funcs import *

__all__ = _funcs.__all__.copy()
