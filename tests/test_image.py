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

import gc

import numpy as np

from heregoes import image, load
from heregoes.util import scale_idx
from tests import output_dir, resources_l1b, resources_l2

epsilon_k = 1e-1
epsilon_rf = 1e-3


def test_l2_cmi():
    l1b_ncs = resources_l1b.meso_ncs
    l2_ncs = resources_l2.meso_ncs

    for i in range(len(l1b_ncs)):
        l1b_data = load(l1b_ncs[i])
        l2_data = load(l2_ncs[i])

        # TODO: should this be integrated into our implementation of CMI?
        # operational CMI seems to clip at 0, but it can be useful to see out of range (negative) values
        heregoes_img = image.ABIImage(l1b_data)
        heregoes_img.cmi[heregoes_img.cmi < 0] = 0.0

        operational_cmi = l2_data.variables.CMI[...]

        if 1 <= l1b_data.variables.band_id[...] <= 6:
            assert (np.abs(heregoes_img.cmi - operational_cmi) < epsilon_rf).all()

        elif 7 <= l1b_data.variables.band_id[...] <= 16:
            assert (np.abs(heregoes_img.cmi - operational_cmi) < epsilon_k).all()


def test_abi_image():
    # test single-channel images
    for abi_nc in resources_l1b.abi_ncs:
        abi_image = image.ABIImage(abi_nc, gamma=0.75, black_space=True)

        assert abi_image.rad.dtype == np.float32
        assert abi_image.cmi.dtype == np.float32
        assert abi_image.bv.dtype == np.uint8

        abi_image.save(filepath=output_dir, ext=".jpg")

    # test index alignment for subsetted RGB images
    lat_bounds_500m = (46.0225830078125, 43.89013671875)
    lon_bounds_500m = [-94.68467712402344, -91.75820922851562]
    lat_bounds_1km = [46.02677536010742, 43.90188217163086]
    lon_bounds_1km = (-94.6901626586914, -91.77256774902344)

    for scene in ["meso", "conus"]:
        if scene == "meso":
            slc_500m = np.s_[213:474, 11:307]
            r_nc = resources_l1b.abi_mc02_nc
            g_nc = resources_l1b.abi_mc03_nc
            b_nc = resources_l1b.abi_mc01_nc

        elif scene == "conus":
            slc_500m = np.s_[613:875, 4451:4747]
            r_nc = resources_l1b.abi_cc02_nc
            g_nc = resources_l1b.abi_cc03_nc
            b_nc = resources_l1b.abi_cc01_nc

        slc_1km = scale_idx(slc_500m, 0.5)

        for upscale in [True, False]:
            for upscale_algo in ["area", "cubic", "lanczos", "linear", "nearest"]:
                if upscale:
                    slc = slc_500m
                    lat_bounds = lat_bounds_500m
                    lon_bounds = lon_bounds_500m
                else:
                    slc = slc_1km
                    lat_bounds = lat_bounds_1km
                    lon_bounds = lon_bounds_1km

                # full RGB
                abi_rgb_full = image.ABINaturalRGB(
                    r_nc,
                    g_nc,
                    b_nc,
                    upscale=upscale,
                    upscale_algo=upscale_algo,
                    gamma=0.75,
                    black_space=True,
                )

                assert abi_rgb_full.bv.dtype == np.uint8

                abi_rgb_full.save(filepath=output_dir, ext=".jpeg")

                # indexed RGB
                abi_rgb_indexed_bounds = image.ABINaturalRGB(
                    r_nc,
                    g_nc,
                    b_nc,
                    index=slc,
                    upscale=upscale,
                    upscale_algo=upscale_algo,
                    gamma=0.75,
                    black_space=True,
                )

                # latlon RGB
                abi_rgb_latlon_bounds = image.ABINaturalRGB(
                    r_nc,
                    g_nc,
                    b_nc,
                    lat_bounds=lat_bounds,
                    lon_bounds=lon_bounds,
                    upscale=upscale,
                    upscale_algo=upscale_algo,
                    gamma=0.75,
                    black_space=True,
                )

                # get the original index of the brightest pixel within the slice
                brightest_idx_500m = np.unravel_index(
                    np.nanargmax(np.sum(abi_rgb_full.bv[slc], axis=2)),
                    abi_rgb_full.bv[slc].shape[0:2],
                )

                # if the RGB image is upscaled, then the 500m slice of the below subsetted images will have been aligned +1,+1 pixels to the 1 km FGF
                brightest_idx_500m_aligned = tuple(
                    [i + upscale for i in brightest_idx_500m]
                )
                assert (
                    brightest_idx_500m_aligned
                    == np.unravel_index(
                        np.nanargmax(np.sum(abi_rgb_indexed_bounds.bv, axis=2)),
                        abi_rgb_indexed_bounds.bv.shape[0:2],
                    )
                    == np.unravel_index(
                        np.nanargmax(np.sum(abi_rgb_latlon_bounds.bv, axis=2)),
                        abi_rgb_latlon_bounds.bv.shape[0:2],
                    )
                )

                del abi_rgb_full
                del abi_rgb_indexed_bounds
                del abi_rgb_latlon_bounds
                _ = gc.collect()


def test_suvi_image():
    for suvi_nc in resources_l1b.suvi_ncs:
        suvi_image = image.SUVIImage(suvi_nc)

        assert suvi_image.rad.dtype == np.float32
        assert suvi_image.bv.dtype == np.uint8

        suvi_image.save(filepath=output_dir, ext=".png")
