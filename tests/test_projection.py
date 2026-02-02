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

from heregoes import image, load, projection
from tests import output_dir, resources_20190904, resources_l1b

epsilon_radians = 1e-7


def test_bounds():
    # in the L1b netCDF, we are given y_image_bounds and x_image_bounds,
    # which differ from the coordinates pixels y and x at the edges of the image.
    # the difference is half of the IFOV in radians (half a pixel)
    # because the pixelwise navigation is to the center of each pixel.
    for nc in [
        resources_20190904.abi_m1c02_T1700_nc,
        resources_20190904.abi_m2c02_T1700_nc,
        resources_20190904.abi_cc02_nc,
        resources_20190904.abi_fc02_nc,
    ]:
        loaded = load(nc)

        # validate that non-subsetted image has the same bounds as the provided y_image_bounds, x_image_bounds
        pad = loaded.resolution_ifov / np.float32(2.0)
        derived_y_bounds = np.array(
            [loaded.variables.y[0] + pad, loaded.variables.y[-1] - pad],
            dtype=np.float64,
        ).reshape(loaded.variables.y_image_bounds.shape)
        derived_x_bounds = np.array(
            [loaded.variables.x[0] - pad, loaded.variables.x[-1] + pad],
            dtype=np.float64,
        ).reshape(loaded.variables.x_image_bounds.shape)

        assert (
            np.abs(derived_y_bounds - loaded.variables.y_image_bounds[...])
            < epsilon_radians
        ).all()
        assert (
            np.abs(derived_x_bounds - loaded.variables.x_image_bounds[...])
            < epsilon_radians
        ).all()

        # validate that the implementation in ABIProjection works out the same
        proj = projection.ABIProjection(nc)
        assert (derived_y_bounds == proj.y_image_bounds).all()
        assert (derived_x_bounds == proj.x_image_bounds).all()

        h = loaded.variables["goes_imager_projection"].perspective_point_height
        assert (derived_y_bounds * h == proj.y_projected_bounds).all()
        assert (derived_x_bounds * h == proj.x_projected_bounds).all()


def test_projection():
    abi_rgb = image.ABINaturalRGB(
        resources_l1b.abi_mc02_nc,
        resources_l1b.abi_mc03_nc,
        resources_l1b.abi_mc01_nc,
        lat_bounds=[45.4524291240206, 44.567740562044825],
        lon_bounds=[-93.86441304735817, -92.69697639796475],
        upscale=True,
        gamma=1.0,
    )
    abi_rgb.resample2cog(
        source="bv", filepath=output_dir.joinpath(abi_rgb.default_filename + ".tiff")
    )

    slc = np.s_[0:3500, 1000:9000]
    abi_rgb = image.ABINaturalRGB(
        resources_l1b.abi_cc02_nc,
        resources_l1b.abi_cc03_nc,
        resources_l1b.abi_cc01_nc,
        index=slc,
        upscale=True,
        gamma=0.75,
    )
    abi_rgb.resample2cog(
        source="bv",
        filepath=output_dir.joinpath(abi_rgb.default_filename + ".tiff"),
        resample_algo="nearest",
    )


# def test_manyprojections():
#     gamma = 0.75
#     for resample_algo in ["bilinear", "cubicspline", "lanczos", "nearest"]:
#         for nc in [
#             resources_20190904.abi_cc02_nc,
#             resources_20190904.abi_fc02_nc,
#             resources_20190904.abi_m1c02_T1700_nc,
#             resources_20190904.abi_m1c02_T1701_nc,
#             resources_20190904.abi_m2c02_T1700_nc,
#             resources_20190904.abi_m2c02_T1701_nc,
#         ]:
#             img = image.ABIImage(nc, gamma=gamma)
#             img.resample2cog(
#                 source="bv",
#                 filepath=output_dir.joinpath(
#                     f"{img.default_filename}_{resample_algo}.tiff"
#                 ),
#                 resample_algo=resample_algo,
#             )

#         for upscale in [False, True]:
#             for rgb in [
#                 resources_20190904.conus_rgb_ncs,
#                 resources_20190904.fulldisk_rgb_ncs,
#                 resources_20190904.meso1_t1700_rgb_ncs,
#                 resources_20190904.meso1_t1701_rgb_ncs,
#                 resources_20190904.meso2_t1700_rgb_ncs,
#                 resources_20190904.meso2_t1701_rgb_ncs,
#             ]:
#                 img = image.ABINaturalRGB(*rgb, gamma=gamma, upscale=upscale)
#                 img.resample2cog(
#                     source="bv",
#                     filepath=output_dir.joinpath(
#                         f"{img.default_filename}_upscale{upscale}_{resample_algo}.tiff"
#                     ),
#                     resample_algo=resample_algo,
#                 )
