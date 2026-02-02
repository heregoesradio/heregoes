# Copyright (c) 2022-2025.

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

import numpy as np

from heregoes import image, load
from tests import output_dir
from tests.resources_l1b import *


def test_ncinterface():
    abi_data = load(abi_cc07_nc)

    # test __str__ on variable and dims
    print(abi_data.variables.Rad)
    print(abi_data.variables.Rad.dimensions)

    # take a small masked slice to test on
    array_slice = abi_data.variables.Rad[10:15, 10:15]
    assert (abi_data.variables.Rad.mask == True).all()
    assert abi_data.variables.Rad.mask.shape == array_slice.shape == (5, 5)

    # test assignment
    array_slice[0, 0] = 1.0
    array_slice[1, 1] = 2.0

    # test compound assignment
    array_slice[2:4, 2:4] /= 0.05
    array_slice[:, -1] *= 10

    test_arr = np.array(
        [
            [1.0000e00, 1.6383e04, 1.6383e04, 1.6383e04, 1.6383e05],
            [1.6383e04, 2.0000e00, 1.6383e04, 1.6383e04, 1.6383e05],
            [1.6383e04, 1.6383e04, 3.2766e05, 3.2766e05, 1.6383e05],
            [1.6383e04, 1.6383e04, 3.2766e05, 3.2766e05, 1.6383e05],
            [1.6383e04, 1.6383e04, 1.6383e04, 1.6383e04, 1.6383e05],
        ],
        dtype=np.float32,
    )

    # test contents of slice against known test_arr
    assert (array_slice == test_arr).all()

    # change the index in memory and test the new shape
    assert abi_data.variables.Rad[...].shape == (1500, 2500)

    # while we're looking at the full array, test how much of it is masked
    assert abi_data.variables.Rad.pct_unmasked == 0.9874234666666667

    # the contents of the same slice should have changed when the inner index changed
    assert not (abi_data.variables.Rad[...][10:15, 10:15] == test_arr).all()
    assert not (abi_data.variables.Rad[10:15, 10:15] == test_arr).all()

    # test that the fill value can be changed
    test_filled_arr = np.array(
        [
            [99.0, 99.0, 99.0, 99.0, 99.0],
            [99.0, 99.0, 99.0, 99.0, 99.0],
            [99.0, 99.0, 99.0, 99.0, 99.0],
            [99.0, 99.0, 99.0, 99.0, 99.0],
            [99.0, 99.0, 99.0, 99.0, 99.0],
        ],
        dtype=np.float32,
    )
    abi_data.variables.Rad.set_fill_value(99)
    assert (abi_data.variables.Rad[10:15, 10:15] == test_filled_arr).all()


def test_exceptions():
    try:
        abi_data = load("not_a_real_netcdf.nc")

    except Exception as e:
        assert isinstance(e, ValueError)

    try:
        abi_data = load("fake_abi-l1b_netcdf.nc")

    except Exception as e:
        assert isinstance(e, FileNotFoundError)

    try:
        suvi_data = load("fake_suvi-l1b_netcdf.nc")

    except Exception as e:
        assert isinstance(e, FileNotFoundError)

    try:
        abi_image = image.ABIImage(abi_mc07_nc)
        abi_image.save(filepath=output_dir, ext=".not_a_file_extension")

    except Exception as e:
        assert isinstance(e, IOError)
