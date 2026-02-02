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

import cv2
import numpy as np

from heregoes.core import NCInterface
from heregoes.projection import ABIProjection
from heregoes.util import linear_interp, minmax
from tests import output_dir
from tests.resources_ancillary import iremis_locations_nc, iremis_nc
from tests.resources_l1b import *


def test_iremis():
    # test projecting an ancillary dataset to an ABI scene
    abi_scene_projection = ABIProjection(
        abi_cc07_nc, lat_bounds=[44.224377, 19.44], lon_bounds=[-113.55036, -71.337036]
    )

    iremis_data = NCInterface(iremis_nc)

    c07_land_emissivity = linear_interp(
        3.7,
        4.3,
        iremis_data["emis1"][...],
        iremis_data["emis2"][...],
        3.9,
    ).astype(np.float32)

    # ocean pixels have a negative value, we set them to have an emissivity of 1.0
    c07_land_emissivity[c07_land_emissivity < 0.0] = 1.0

    # rotate IREMIS to be N-S and E-W
    c07_land_emissivity = np.flipud(np.rot90(c07_land_emissivity, k=1))

    iremis_locations = NCInterface(iremis_locations_nc)
    iremis_ul_lat = iremis_locations["lat"][0, 0].item()
    iremis_ul_lon = iremis_locations["lon"][0, 0].item()
    iremis_lr_lat = iremis_locations["lat"][-1, -1].item()
    iremis_lr_lon = iremis_locations["lon"][-1, -1].item()

    c07_land_emissivity = abi_scene_projection.resample2abi(
        c07_land_emissivity,
        lat_bounds=[iremis_ul_lat, iremis_lr_lat],
        lon_bounds=[iremis_ul_lon, iremis_lr_lon],
        resample_algo="bilinear",
    )

    assert c07_land_emissivity.dtype == np.float32
    assert c07_land_emissivity.shape == (
        1000,
        1500,
    )

    cv2.imwrite(
        str(output_dir.joinpath("iremis_c07.png")),
        minmax(c07_land_emissivity) * 255,
    )
