# Copyright (c) 2020-2025.

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

"""NumPy and OpenCV helper functions optimized with Numba where possible"""

import cv2
import numpy as np
from numba import prange
from numba.np.unsafe.ndarray import to_fixed_tuple

from heregoes.core import heregoes_njit, heregoes_njit_noparallel

__all__ = [
    "align_idx",
    "centered_slice",
    "linear_interp",
    "linear_norm",
    "linear_scale",
    "make_8bit",
    "minmax",
    "nearest_1d_indices",
    "nearest_2d_indices",
    "nearest_2d_search",
    "nearest_scale",
    "scale_arr",
    "scale_idx",
    "unravel_index",
    "x2",
    "x4",
]


@heregoes_njit_noparallel
def linear_interp(x1, x2, y1, y2, x):  # pragma: no cover
    """
    Linearly interpolates y at x for x between x1, x2 and y between y1, y2
    """
    return (y1 * (x2 - x) + y2 * (x - x1)) / (x2 - x1)


@heregoes_njit
def linear_norm(
    arr, old_min, old_max, new_min=0.0, new_max=1.0, copy=True
):  # pragma: no cover
    """
    Linearly normalizes `arr` with `old_min` and `old_max` to be between `new_min` and `new_max`
    """
    if copy:
        return np.clip(
            (arr - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min,
            new_min,
            new_max,
        )

    else:
        arr = np.clip(
            (arr - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min,
            new_min,
            new_max,
        )
        return arr


def minmax(arr):
    """
    Convenience function for the common case of linear normalization where `old_min` and `old_max` are the nanmin and nanmax of `arr`.
    """
    return linear_norm(arr, old_min=np.nanmin(arr), old_max=np.nanmax(arr))


def scale_arr(arr, k, interpolation):
    """
    Scale `arr` by a factor of `k` using cv2
    """
    if k == 1:
        return arr

    original_type = arr.dtype
    if arr.dtype == bool:
        cast_type = np.uint8
    else:
        cast_type = original_type
    return cv2.resize(
        arr.astype(cast_type), None, fx=k, fy=k, interpolation=interpolation
    ).astype(original_type)


def nearest_scale(arr, k):
    return scale_arr(arr, k, interpolation=cv2.INTER_NEAREST)


def linear_scale(arr, k):
    return scale_arr(arr, k, interpolation=cv2.INTER_LINEAR)


def x2(arr):
    return nearest_scale(arr, k=2)


def x4(arr):
    return nearest_scale(arr, k=4)


def align_idx(idx, modulus):
    """
    Aligns an array index or continuous slice `idx` to the closest index modulo `modulus`.
    Useful for aligning indices between different spatial resolutions,
    e.g. aligning a 500 m index to the 1 km ABI Fixed Grid with a `modulus` of 2,
    or to the 2 km Fixed Grid with a `modulus` of 4.
    """

    def safe_align(value, modulus):
        if value is None or value is Ellipsis:
            return value

        else:
            remainder = value % modulus
            i = value - remainder
            j = (value + modulus) - remainder

            return np.where(j - value < value - i, j, i).astype(int)

    def slice_align(slc, modulus):
        start = safe_align(slc.start, modulus)
        stop = safe_align(slc.stop, modulus)

        return np.s_[start:stop:None]

    aligned_idx = []
    for i in tuple(idx):
        if isinstance(i, slice):
            slc = slice_align(i, modulus)

        else:
            slc = safe_align(i, modulus)

        aligned_idx.append(slc)

    return tuple(aligned_idx)


def scale_idx(idx, scale_factor):
    """
    Adjusts an array index or continous slice `idx` by `scale_factor`.
    Useful for referencing the same point between different spatial resolutions,
    e.g. converting a 500 m index for use with a 2 km product using a `scale_factor` of 0.25.
    """

    def safe_floor(value):
        if value is None or value is Ellipsis:
            return value

        else:
            return np.floor(value * scale_factor).astype(int)

    def slice_floor(slc):
        start = safe_floor(slc.start)
        stop = safe_floor(slc.stop)

        return np.s_[start:stop:None]

    idx = np.s_[idx]

    scaled_idx = []
    for i in tuple(idx):
        if isinstance(i, slice):
            slc = slice_floor(i)

        else:
            slc = safe_floor(i)

        scaled_idx.append(slc)

    return tuple(scaled_idx)


@heregoes_njit
def make_8bit(arr):  # pragma: no cover
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def centered_slice(center_idx, shape):
    """
    Returns a slice object centered on `center_idx` with shape `shape`.
    """

    center_y, center_x = center_idx

    y1 = max(center_y - shape[0] // 2, 0)
    y2 = y1 + shape[0]

    x1 = max(center_x - shape[1] // 2, 0)
    x2 = x1 + shape[1]

    return np.s_[y1:y2, x1:x2]


@heregoes_njit
def unravel_index(index, shape):  # pragma: no cover
    # copied from: https://github.com/liberTEM/LiberTEM-blobfinder (GPLv3)
    sizes = np.zeros(len(shape), dtype=np.int64)
    result = np.zeros(len(shape), dtype=np.int64)
    sizes[-1] = 1
    for i in range(len(shape) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * shape[i + 1]
    remainder = index
    for i in range(len(shape)):
        result[i] = remainder // sizes[i]
        remainder %= sizes[i]
    return to_fixed_tuple(result, len(shape))


@heregoes_njit
def nearest_2d_indices(y_arr, x_arr, target_y, target_x):  # pragma: no cover
    # inspired by https://github.com/blaylockbk/pyBKB_v3/blob/master/demo/Nearest_lat-lon_Grid.ipynb

    if np.isnan(y_arr).any() | np.isnan(x_arr).any():
        y_arr = y_arr.copy()
        x_arr = x_arr.copy()

        y_arr.ravel()[np.nonzero(np.isnan(y_arr.ravel()))] = np.inf
        x_arr.ravel()[np.nonzero(np.isnan(x_arr.ravel()))] = np.inf

    idx_shape = (y_arr.shape[0], x_arr.shape[-1])

    target_y_raveled = target_y.ravel()
    target_x_raveled = target_x.ravel()

    nearest_ys = np.zeros(target_y.size, dtype=np.int64)
    nearest_xs = np.zeros(target_x.size, dtype=np.int64)

    for idx in prange(target_y.size):
        max_of_abs_differences = np.fmax(
            np.abs(y_arr - target_y_raveled[idx]),
            np.abs(x_arr - target_x_raveled[idx]),
        )
        argmin_of_maximum = np.int64(np.argmin(max_of_abs_differences))

        nearest_ys[idx], nearest_xs[idx] = unravel_index(argmin_of_maximum, idx_shape)

    return (nearest_ys.reshape(target_y.shape), nearest_xs.reshape(target_x.shape))


def nearest_1d_indices(a, v):
    sorted_idxs = np.argsort(a)

    sorted = a[sorted_idxs]

    insertion_idxs_sorted = np.searchsorted(sorted, v)
    insertion_idxs_sorted = np.clip(insertion_idxs_sorted, 0, a.size - 1)

    nearest_idxs_sorted = np.where(
        (insertion_idxs_sorted > 0)
        & (
            np.abs(v - sorted[insertion_idxs_sorted - 1])
            < np.abs(v - sorted[insertion_idxs_sorted])
        ),
        insertion_idxs_sorted - 1,
        insertion_idxs_sorted,
    )

    nearest_idxs = sorted_idxs[nearest_idxs_sorted]

    return nearest_idxs


def nearest_2d_search(y_arr, x_arr, target_y, target_x):
    """
    Search in (1D or 2D) coordinate arrays `y_arr`, `x_arr` for `target_y`, `target_x`.

    Returns tuple of the nearest indices or slices matching `target_y`, `target_x` in `y_arr`, `x_arr`.

    Notes:
    - Two target coordinates in `target_y`, `target_x` will resolve to a continous 2D slice
    - 2D `y_arr`, `x_arr` is slower for large `target_y`, `target_x`
    """
    if target_y.shape != target_x.shape:
        raise ValueError("`target_y` and `target_x` must be the same shape.")

    if y_arr.ndim == x_arr.ndim == 1:
        nearest_ys = nearest_1d_indices(y_arr, target_y)
        nearest_xs = nearest_1d_indices(x_arr, target_x)

        nearest_indices = (nearest_ys, nearest_xs)

    elif y_arr.ndim == x_arr.ndim == 2:
        nearest_indices = nearest_2d_indices(y_arr, x_arr, target_y, target_x)

    else:
        raise ValueError(
            "`y_arr` and `x_arr` must have the same number of dimensions between 1 and 2."
        )

    # form a single tuple index
    if len(nearest_indices[0]) == 1:
        nearest_indices = (nearest_indices[0].item(), nearest_indices[1].item())

    # or make a continous slice between 2 indices OR a monotonic increasing range
    elif len(nearest_indices[0]) == 2 or (
        all(
            [
                (np.diff(nearest_indices[i], axis=i) == 1).all()
                for i in range(nearest_indices[0].ndim)
            ]
        )
    ):
        nearest_indices = tuple(
            slice(idx.min(), idx.max() + 1) for idx in nearest_indices
        )

    return nearest_indices
