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

"""NumPy and OpenCV helper functions optimized with Numba where possible"""

import cv2
import numpy as np
from numba.np.unsafe.ndarray import to_fixed_tuple

from heregoes.util import njit


@njit.heregoes_njit
def linear_interp(x1, x2, y1, y2, x):
    # Linearly interpolates y at x for x between x1, x2 and y between y1, y2
    return (y1 * (x2 - x) + y2 * (x - x1)) / (x2 - x1)


@njit.heregoes_njit
def linear_norm(arr, old_min, old_max, new_min=0.0, new_max=1.0, copy=True):
    # Linearly normalizes arr to be between new_min and new_max
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
    return linear_norm(arr, old_min=np.nanmin(arr), old_max=np.nanmax(arr))


def nearest_scale(arr, k):
    # Let cv2 resize boolean matrices by casting to uint8, then back to bool
    original_type = arr.dtype
    if arr.dtype == bool:
        cast_type = np.uint8
    else:
        cast_type = original_type
    return cv2.resize(
        arr.astype(cast_type), None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST
    ).astype(original_type)


def x2(arr):
    return nearest_scale(arr, k=2)


def x4(arr):
    return nearest_scale(arr, k=4)


def align_idx(idx, modulus):
    """
    Aligns an array index or continuous slice `idx` to the closest index modulo `modulus`.
    Useful for aligning indices between different spatial resolutions, e.g. aligning a 500 m index to the 1 km ABI Fixed Grid with a `modulus` of 2, or to the 2 km Fixed Grid with a `modulus` of 4.
    """

    def safe_align(value, modulus):
        if value is None or value is Ellipsis:
            return value

        else:
            i = value - (value % modulus)
            j = (value + modulus) - (value % modulus)

            if j - value < value - i:
                return int(j)

            else:
                return int(i)

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
    Useful for referencing the same point between different spatial resolutions, e.g. converting a 500 m index for use with a 2 km product using a `scale_factor` of 0.25.
    """

    def safe_floor(value):
        if value is None or value is Ellipsis:
            return value

        else:
            return int(np.floor(value * scale_factor))

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


@njit.heregoes_njit_noparallel
def window_slice(arr, center_index, outer_radius, inner_radius=0, replace_inner=True):
    """
    Returns the square window of array `arr` centered at `center_index` with radius `outer_radius` and diameter `outer_radius` * 2 + 1.
    The central value in the window is optionally replaced with np.nan. Additional inner values may be replaced with np.nan by setting `inner_radius`.
    """

    y, x = center_index
    window = arr[
        y - outer_radius : y + outer_radius + 1, x - outer_radius : x + outer_radius + 1
    ].copy()

    window_center_y = window.shape[0] // 2
    window_center_x = window.shape[1] // 2

    if replace_inner:
        window[
            window_center_y - inner_radius : window_center_y + 1 + inner_radius,
            window_center_x - inner_radius : window_center_x + 1 + inner_radius,
        ] = np.nan

    return window


@njit.heregoes_njit_noparallel
def window_deviation(arr, idx, outer_radius=15, inner_radius=0):
    window = window_slice(arr, idx, outer_radius, inner_radius)
    mean = np.nanmean(window)
    deviation = arr[idx] - mean

    return deviation


@njit.heregoes_njit
def fill_border(arr, width, fill=np.nan, copy=True):
    # fills a border `width` pixels wide on a 2D array with value `fill`
    if copy:
        arr = arr.copy()

    arr[0 : int(width) + 1, :] = fill  # top
    arr[:, 0 : int(width) + 1] = fill  # left
    arr[-int(width) :, :] = fill  # bottom
    arr[:, -int(width) :] = fill  # right

    return arr


@njit.heregoes_njit_noparallel
def round(arr, decimals, dtype):
    out = np.empty_like(arr)
    return np.around(arr, decimals, out).astype(dtype)


@njit.heregoes_njit_noparallel
def rad2deg(arr):
    out = np.empty_like(arr)
    out = np.rad2deg(arr)
    return out


@njit.heregoes_njit_noparallel
def deg2rad(arr):
    out = np.empty_like(arr)
    out = np.deg2rad(arr)
    return out


@njit.heregoes_njit
def make_8bit(arr):
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def resize2width(arr, new_width, interp=cv2.INTER_NEAREST):
    # resize 2D arr to a new width while maintaining aspect ratio
    old_width, old_height = arr.shape

    resize_scaler = new_width / old_width
    resize_width = int(np.floor(old_width * resize_scaler))
    resize_height = int(np.floor(old_height * resize_scaler))

    return cv2.resize(arr, (resize_width, resize_height), interpolation=interp)


def resize2height(arr, new_height, interp=cv2.INTER_NEAREST):
    # resize 2D arr to a new height while maintaining aspect ratio
    old_width, old_height = arr.shape

    resize_scaler = new_height / old_height
    resize_width = int(np.floor(old_width * resize_scaler))
    resize_height = int(np.floor(old_height * resize_scaler))

    return cv2.resize(arr, (resize_width, resize_height), interpolation=interp)


@njit.heregoes_njit_noparallel
def crop_center(arr, center_idx, crop_shape):
    """
    Returns a crop of `arr` on center coordinate `center_idx` with shape `crop_shape`.
    The crop never extends beyond the border of `arr` and always has shape `crop_shape`.
    """

    arr_shape = arr.shape
    center_y, center_x = center_idx

    center_y = max(0, min(center_y, arr_shape[0] - (crop_shape[0] // 2 + 1)))
    center_x = max(0, min(center_x, arr_shape[1] - (crop_shape[1] // 2 + 1)))

    y1 = max(center_y - crop_shape[0] // 2, 0)
    y2 = y1 + crop_shape[0]

    x1 = max(center_x - crop_shape[1] // 2, 0)
    x2 = x1 + crop_shape[1]

    return arr[y1:y2, x1:x2].copy()


@njit.heregoes_njit
def unravel_index(index, shape):
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


def nearest_2d(y_arr, x_arr, target_y, target_x):
    """
    Finds the nearest index of a value simultaneously for two arrays `y_arr` and `x_arr` of the same shape.
    For example, with a latitude array `y_arr` and longitude array `x_arr`, returns the index of the nearest (lat, lon) match to (`target_y`, `target_x`).
    Adapted from: https://github.com/blaylockbk/pyBKB_v3/blob/master/demo/Nearest_lat-lon_Grid.ipynb (MIT)
    """

    @njit.heregoes_njit_noparallel
    def _nearest_2d(y_arr, x_arr, target_y, target_x):
        if y_arr.shape != x_arr.shape:
            return

        if np.isnan(y_arr).any() | np.isnan(x_arr).any():
            y_arr = y_arr.copy()
            x_arr = x_arr.copy()

            y_arr.ravel()[np.nonzero(np.isnan(y_arr.ravel()))] = np.inf
            x_arr.ravel()[np.nonzero(np.isnan(x_arr.ravel()))] = np.inf

        nearest_ys = np.zeros(target_y.ravel().shape, dtype=np.int64)
        nearest_xs = np.zeros(target_x.ravel().shape, dtype=np.int64)

        flatidx = 0
        for idx, val in np.ndenumerate(target_y):
            nearest_y, nearest_x = unravel_index(
                int(
                    np.argmin(
                        np.maximum(
                            np.abs(y_arr - target_y[idx]), np.abs(x_arr - target_x[idx])
                        )
                    )
                ),
                y_arr.shape,
            )

            nearest_ys[flatidx] = nearest_y
            nearest_xs[flatidx] = nearest_x

            flatidx += 1

        return (nearest_ys, nearest_xs)

    nearest_indices = _nearest_2d(y_arr, x_arr, target_y, target_x)

    # form a tuple of slices to encompass a continuous range
    if len(nearest_indices[0]) > 1 or len(nearest_indices[1]) > 1:
        nearest_indices = tuple(
            slice(idx.T.min(), idx.T.max() + 1) for idx in nearest_indices
        )

    # or form a single tuple index
    else:
        nearest_indices = (nearest_indices[0].item(), nearest_indices[1].item())

    return nearest_indices
