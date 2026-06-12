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
    gamma = 0.5

    # test single-channel images
    for div_sun_za in [True, False]:
        for normalize_rf in [True, False]:
            for abi_nc in resources_l1b.abi_ncs:
                abi_image = image.ABIImage(
                    abi_nc,
                    gamma=gamma,
                    black_space=True,
                    normalize_rf=normalize_rf,
                    div_sun_za=div_sun_za,
                )

                assert abi_image.rad.dtype == np.float32
                assert abi_image.cmi.dtype == np.float32
                assert abi_image.bv.dtype == np.uint8

                filename = abi_image.default_filename
                if normalize_rf:
                    filename += "_normalize_rf"

                if div_sun_za:
                    filename += "_div_sun_za"

                abi_image.save(filepath=output_dir.joinpath(filename + ".jpg"))

        # test RGB
        meso_r_nc = resources_l1b.abi_mc02_nc
        meso_g_nc = resources_l1b.abi_mc03_nc
        meso_b_nc = resources_l1b.abi_mc01_nc

        conus_r_nc = resources_l1b.abi_cc02_nc
        conus_g_nc = resources_l1b.abi_cc03_nc
        conus_b_nc = resources_l1b.abi_cc01_nc

        for div_sun_za in [True, False]:
            for normalize_rf in [True, False]:
                abi_rgb_full = image.ABINaturalRGB(
                    conus_r_nc,
                    conus_g_nc,
                    conus_b_nc,
                    gamma=gamma,
                    black_space=True,
                    normalize_rf=normalize_rf,
                    div_sun_za=div_sun_za,
                )

                filename = abi_rgb_full.default_filename
                if normalize_rf:
                    filename += "_normalize_rf"

                if div_sun_za:
                    filename += "_div_sun_za"

                abi_rgb_full.save(filepath=output_dir.joinpath(filename + ".jpg"))

        # test index alignment for subsetted RGB images
        lat_bounds_500m = (46.0225830078125, 43.89013671875)
        lon_bounds_500m = [-94.68467712402344, -91.75820922851562]
        lat_bounds_1km = [46.02677536010742, 43.90188217163086]
        lon_bounds_1km = (-94.6901626586914, -91.77256774902344)

        for scene in ["meso", "conus"]:
            if scene == "meso":
                slc_500m = np.s_[213:474, 11:307]
                r_nc = meso_r_nc
                g_nc = meso_g_nc
                b_nc = meso_b_nc

            elif scene == "conus":
                slc_500m = np.s_[613:875, 4451:4747]
                r_nc = conus_r_nc
                g_nc = conus_g_nc
                b_nc = conus_b_nc

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
                        gamma=gamma,
                        black_space=True,
                    )

                    assert abi_rgb_full.bv.dtype == np.uint8

                    filename = f"{abi_rgb_full.default_filename}_full_rgb"
                    if upscale:
                        filename += f"_upscale_{upscale_algo}"

                    filepath = output_dir.joinpath(filename + ".jpeg")
                    abi_rgb_full.save(filepath=filepath)

                    # indexed RGB
                    abi_rgb_indexed_bounds = image.ABINaturalRGB(
                        r_nc,
                        g_nc,
                        b_nc,
                        index=slc,
                        upscale=upscale,
                        upscale_algo=upscale_algo,
                        gamma=gamma,
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
                        gamma=gamma,
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
        for colorize in [True, False]:
            suvi_image = image.SUVIImage(suvi_nc, colorize=colorize)

            assert suvi_image.rad.dtype == np.float32
            assert suvi_image.bv.dtype == np.uint8

            filename = suvi_image.default_filename
            if colorize:
                filename += "_colorized"

            suvi_image.save(filepath=output_dir.joinpath(filename + ".jpg"))


def test_suvi_rgb():
    red = image.SUVIImage(
        resources_l1b.suvi_red_nc,
        input_range=(0.0, 15.0),
        asinh_a=0.0007,
        output_range=(0.0, 1.0),
    )

    green = image.SUVIImage(
        resources_l1b.suvi_green_nc,
        input_range=(0.0, 50.0),
        asinh_a=0.001,
        output_range=(0.0, 1.0),
    )

    blue = image.SUVIImage(
        resources_l1b.suvi_blue_nc,
        input_range=(0.0, 60.0),
        asinh_a=0.000075,
        output_range=(0.0, 1.0),
    )

    rgb = image.SUVIRGB(red, green, blue)

    assert (rgb.bv[:, :, 2] == red.bv).all()
    assert (rgb.bv[:, :, 1] == green.bv).all()
    assert (rgb.bv[:, :, 0] == blue.bv).all()

    assert (
        rgb.time == np.array([red.time, green.time, blue.time], dtype="datetime64")
    ).all()

    rgb.save(filepath=output_dir, ext=".jpeg")

    try:
        rgb.save("not-a-real-format.xyz")

    except Exception as e:
        assert isinstance(e, IOError)
