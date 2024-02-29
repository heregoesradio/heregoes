# Copyright (c) 2022-2023.

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

import gc

import cv2
import numpy as np

from heregoes import ancillary, exceptions, image, load, navigation, projection
from heregoes.util import minmax, scale_idx

from .test_resources import *

output_dir = SCRIPT_PATH.joinpath("output")
output_dir.mkdir(parents=True, exist_ok=True)

for output_file in output_dir.glob("*"):
    output_file.unlink()

epsilon_degrees = 1e-4
epsilon_meters = 1


def test_abi_image():
    # test single-channel images
    for abi_nc in abi_ncs:
        abi_image = image.ABIImage(abi_nc, gamma=0.75)

        assert abi_image.rad.dtype == np.float32
        assert abi_image.cmi.dtype == np.float32
        assert abi_image.bv.dtype == np.uint8

        abi_image.save(file_path=output_dir, file_ext=".jpg")

    # test index alignment for subsetted RGB images
    lat_bounds_500m = (46.0225830078125, 43.89013671875)
    lon_bounds_500m = [-94.68467712402344, -91.75820922851562]
    lat_bounds_1km = [46.02677536010742, 43.90188217163086]
    lon_bounds_1km = (-94.6901626586914, -91.77256774902344)

    for scene in ["meso", "conus"]:
        if scene == "meso":
            slc_500m = np.s_[213:474, 11:307]
            r_nc = abi_mc02_nc
            g_nc = abi_mc03_nc
            b_nc = abi_mc01_nc

        elif scene == "conus":
            slc_500m = np.s_[613:875, 4451:4747]
            r_nc = abi_cc02_nc
            g_nc = abi_cc03_nc
            b_nc = abi_cc01_nc

        slc_1km = scale_idx(slc_500m, 0.5)

        for upscale in [True, False]:
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
                gamma=0.75,
                black_space=True,
            )

            assert abi_rgb_full.bv.dtype == np.uint8

            abi_rgb_full.save(file_path=output_dir, file_ext=".jpeg")

            # indexed RGB
            abi_rgb_indexed_bounds = image.ABINaturalRGB(
                r_nc,
                g_nc,
                b_nc,
                index=slc,
                upscale=upscale,
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
    for suvi_nc in suvi_ncs:
        suvi_image = image.SUVIImage(suvi_nc)

        assert suvi_image.rad.dtype == np.float32
        assert suvi_image.bv.dtype == np.uint8

        suvi_image.save(file_path=output_dir, file_ext=".png")


def test_projection():
    abi_rgb = image.ABINaturalRGB(
        abi_mc02_nc,
        abi_mc03_nc,
        abi_mc01_nc,
        lat_bounds=[45.4524291240206, 44.567740562044825],
        lon_bounds=[-93.86441304735817, -92.69697639796475],
        upscale=True,
        gamma=1.0,
    )
    abi_projection = projection.ABIProjection(abi_rgb.abi_data, index=abi_rgb.index)
    abi_rgb = abi_projection.resample2cog(
        abi_rgb.bv, output_dir.joinpath(abi_rgb.default_filename + ".tiff")
    )

    slc = np.s_[0:3500, 1000:9000]
    abi_rgb = image.ABINaturalRGB(
        abi_cc02_nc, abi_cc03_nc, abi_cc01_nc, index=slc, upscale=True, gamma=0.75
    )
    abi_projection = projection.ABIProjection(abi_rgb.abi_data, index=abi_rgb.index)
    abi_rgb = abi_projection.resample2cog(
        abi_rgb.bv, output_dir.joinpath(abi_rgb.default_filename + ".tiff")
    )


def test_point_navigation():
    idx = (92, 42)

    meso_data = load(abi_mc07_nc)

    meso_nav = navigation.ABINavigation(meso_data, precise_sun=False)

    assert meso_nav.lat_deg.dtype == np.float32
    assert meso_nav.lon_deg.dtype == np.float32
    assert meso_nav.lat_deg[idx] == 44.73149490356445
    assert meso_nav.lon_deg[idx] == -93.01798248291016

    assert meso_nav.area_m.dtype == np.float32
    assert meso_nav.area_m[idx] == 4398247.5

    assert meso_nav.sat_za.dtype == np.float32
    assert meso_nav.sat_az.dtype == np.float32
    assert meso_nav.sat_za[idx] == 0.9509874582290649
    assert meso_nav.sat_az[idx] == 2.7129034996032715

    assert meso_nav.sun_za.dtype == np.float32
    assert meso_nav.sun_az.dtype == np.float32
    assert meso_nav.sun_za[idx] == 0.4886804521083832
    assert meso_nav.sun_az[idx] == 3.9760777950286865

    # test on astropy sun
    meso_nav = navigation.ABINavigation(meso_data, index=idx, precise_sun=True)

    assert meso_nav.sun_za.dtype == np.float32
    assert meso_nav.sun_az.dtype == np.float32
    assert meso_nav.sun_za[0] == 0.4887488782405853
    assert meso_nav.sun_az[0] == 3.976273775100708

    # test with height correction
    meso_nav = navigation.ABINavigation(meso_data, precise_sun=False, hae_m=1.2345678)

    assert meso_nav.lat_deg.dtype == np.float32
    assert meso_nav.lon_deg.dtype == np.float32
    assert meso_nav.lat_deg[idx] == 44.73153305053711
    assert meso_nav.lon_deg[idx] == -93.01801300048828

    assert meso_nav.area_m.dtype == np.float32
    assert meso_nav.area_m[idx] == 4398248.0

    assert meso_nav.sat_za.dtype == np.float32
    assert meso_nav.sat_az.dtype == np.float32
    assert meso_nav.sat_za[idx] == 0.9509882926940918
    assert meso_nav.sat_az[idx] == 2.7129032611846924

    assert meso_nav.sun_za.dtype == np.float32
    assert meso_nav.sun_az.dtype == np.float32
    assert meso_nav.sun_za[idx] == 0.4886806607246399
    assert meso_nav.sun_az[idx] == 3.976076126098633

    # test with astropy sun and height correction
    meso_nav = navigation.ABINavigation(
        meso_data, index=idx, precise_sun=True, hae_m=1.2345678
    )

    assert meso_nav.sun_za.dtype == np.float32
    assert meso_nav.sun_az.dtype == np.float32
    assert meso_nav.sun_za[0] == 0.48874911665916443
    assert meso_nav.sun_az[0] == 3.9762721061706543

    # test single-point indexing on a navigation dataset that does not contain NaNs (G16 meso)
    meso_nav = navigation.ABINavigation(
        meso_data, lat_bounds=44.72609499, lon_bounds=-93.02279070
    )
    assert meso_nav.index == idx

    # test single-point indexing on a navigation dataset that contains NaNs (G16 CONUS)
    conus_data = load(abi_cc07_nc)
    conus_nav = navigation.ABINavigation(
        conus_data, lat_bounds=44.72609499, lon_bounds=-93.02279070
    )
    assert conus_nav.index == (192, 1152)


def test_range_navigation():
    # test meso
    meso_data = load(abi_mc07_nc)

    # test retrieving points from lat/lon bounds
    lat_bounds = np.array(
        [
            44.731495,
            44.602596,
        ],
        dtype=np.float32,
    )
    lon_bounds = np.array(
        [
            -93.01798,
            -92.85648,
        ],
        dtype=np.float32,
    )
    meso_nav_bounded = navigation.ABINavigation(
        meso_data, lat_bounds=lat_bounds, lon_bounds=lon_bounds, degrees=True
    )

    # check index
    assert meso_nav_bounded.index == np.s_[92:97, 42:47]

    # check bounds
    assert meso_nav_bounded.lat_deg[0, 0], (
        meso_nav_bounded.lat_deg[-1, -1] == lat_bounds
    )
    assert meso_nav_bounded.lon_deg[0, 0], (
        meso_nav_bounded.lon_deg[-1, -1] == lon_bounds
    )

    # another nav object using the same derived index should have the same gridded data
    meso_nav_indexed = navigation.ABINavigation(
        meso_data, index=meso_nav_bounded.index, degrees=True
    )
    for i in ["lat_deg", "lon_deg", "sat_za", "sat_az", "sun_za", "sun_az", "area_m"]:
        assert (getattr(meso_nav_indexed, i) == getattr(meso_nav_bounded, i)).all()

    # test conus
    conus_data = load(abi_cc07_nc)

    # test retrieving points fron multiple lat/lon
    lat_bounds = np.array(
        [
            [44.7315, 44.7303, 44.729107, 44.727917, 44.726727],
            [44.700382, 44.69919, 44.697994, 44.696804, 44.695618],
            [44.669365, 44.668175, 44.666985, 44.6658, 44.66461],
            [44.638294, 44.637104, 44.635918, 44.63473, 44.63355],
            [44.60733, 44.606144, 44.60496, 44.60378, 44.602596],
        ],
        dtype=np.float32,
    )
    lon_bounds = np.array(
        [
            [-93.01798, -92.98884, -92.9597, -92.93057, -92.90145],
            [-93.00659, -92.97746, -92.94835, -92.91924, -92.890144],
            [-92.99527, -92.96617, -92.93707, -92.90799, -92.878914],
            [-92.98393, -92.95485, -92.92577, -92.89671, -92.86766],
            [-92.97267, -92.94361, -92.91456, -92.88552, -92.85648],
        ],
        dtype=np.float32,
    )
    conus_nav_bounded = navigation.ABINavigation(
        conus_data, lat_bounds=lat_bounds, lon_bounds=lon_bounds, degrees=True
    )

    # check index
    assert conus_nav_bounded.index == np.s_[192:197, 1152:1157]

    # check bounds
    assert (conus_nav_bounded.lat_deg == lat_bounds).all()
    assert (conus_nav_bounded.lon_deg == lon_bounds).all()

    # another nav object using the same derived index should have the same gridded data
    conus_nav_indexed = navigation.ABINavigation(
        conus_data, index=conus_nav_bounded.index, degrees=True
    )
    for i in ["lat_deg", "lon_deg", "sat_za", "sat_az", "sun_za", "sun_az", "area_m"]:
        assert (getattr(conus_nav_indexed, i) == getattr(conus_nav_bounded, i)).all()

    # meso and conus nav reference the same area, the time-invariant gridded data should match within tolerance
    assert (
        conus_nav_bounded.lat_deg - meso_nav_bounded.lat_deg < epsilon_degrees
    ).all()
    assert (
        conus_nav_bounded.lon_deg - meso_nav_bounded.lon_deg < epsilon_degrees
    ).all()
    assert (conus_nav_bounded.sat_za - meso_nav_bounded.sat_za < epsilon_degrees).all()
    assert (conus_nav_bounded.sat_az - meso_nav_bounded.sat_az < epsilon_degrees).all()
    assert (conus_nav_bounded.area_m - meso_nav_bounded.area_m < epsilon_meters).all()


def test_ancillary():
    slc = np.s_[250:1250, 500:2000]
    abi_data = load(abi_cc07_nc)

    # test water mask
    water_indexed = ancillary.WaterMask(abi_data, index=slc, rivers=True)
    water_bounded = ancillary.WaterMask(
        abi_data,
        lat_bounds=np.array([44.224377, 19.44], dtype=np.float32),
        lon_bounds=np.array([-113.55036, -71.337036], dtype=np.float32),
        rivers=True,
    )

    assert (
        water_indexed.data["water_mask"].dtype
        == water_bounded.data["water_mask"].dtype
        == bool
    )
    assert (
        water_indexed.data["water_mask"].shape
        == water_bounded.data["water_mask"].shape
        == (
            1000,
            1500,
        )
    )
    assert (water_indexed.data["water_mask"] == water_bounded.data["water_mask"]).all()

    water_bounded.save(save_dir=output_dir)
    cv2.imwrite(
        str(output_dir.joinpath("water.png")), water_bounded.data["water_mask"] * 255
    )

    # test IREMIS
    iremis_indexed = ancillary.IREMIS(abi_data, index=slc)
    iremis_bounded = ancillary.IREMIS(
        abi_data,
        lat_bounds=np.array([44.224377, 19.44], dtype=np.float32),
        lon_bounds=np.array([-113.55036, -71.337036], dtype=np.float32),
    )

    assert (
        iremis_indexed.data["c07_land_emissivity"].dtype
        == iremis_bounded.data["c07_land_emissivity"].dtype
        == np.float32
    )
    assert (
        iremis_indexed.data["c07_land_emissivity"].shape
        == iremis_bounded.data["c07_land_emissivity"].shape
        == (
            1000,
            1500,
        )
    )
    assert (
        iremis_indexed.data["c07_land_emissivity"]
        == iremis_bounded.data["c07_land_emissivity"]
    ).all()
    assert (
        iremis_indexed.data["c14_land_emissivity"]
        == iremis_bounded.data["c14_land_emissivity"]
    ).all()

    iremis_bounded.save(save_dir=output_dir)
    cv2.imwrite(
        str(output_dir.joinpath("iremis.png")),
        minmax(iremis_bounded.data["c07_land_emissivity"]) * 255,
    )


def test_ncinterface():
    abi_data = load(abi_cc07_nc)

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
        assert isinstance(e, exceptions.HereGOESUnsupportedProductException)

    try:
        abi_data = load("fake_abi-l1b_netcdf.nc")

    except Exception as e:
        assert isinstance(e, exceptions.HereGOESIOReadException)

    try:
        suvi_data = load("fake_suvi-l1b_netcdf.nc")

    except Exception as e:
        assert isinstance(e, exceptions.HereGOESIOReadException)

    try:
        abi_image = image.ABIImage(abi_mc07_nc)
        abi_image.save(file_path=output_dir, file_ext=".not_a_file_extension")

    except Exception as e:
        assert isinstance(e, exceptions.HereGOESIOWriteException)
