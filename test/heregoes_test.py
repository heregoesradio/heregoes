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

"""Test cases for checking imagery and navigation outputs"""

from pathlib import Path

import cv2
import numpy as np

from heregoes import ancillary, exceptions, image, load, navigation, projection
from heregoes.util import minmax

SCRIPT_PATH = Path(__file__).parent.resolve()

input_dir = SCRIPT_PATH.joinpath("input")
input_dir.mkdir(parents=True, exist_ok=True)
output_dir = SCRIPT_PATH.joinpath("output")
output_dir.mkdir(parents=True, exist_ok=True)

for output_file in output_dir.glob("*"):
    output_file.unlink()

# abi
abi_mc01_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C01_G16_s20211691942252_e20211691942310_c20211691942342.nc"
)
abi_mc02_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C02_G16_s20211691942252_e20211691942310_c20211691942334.nc"
)
abi_mc03_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C03_G16_s20211691942252_e20211691942310_c20211691942351.nc"
)
abi_mc04_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C04_G16_s20211691942252_e20211691942310_c20211691942340.nc"
)
abi_mc05_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C05_G16_s20211691942252_e20211691942310_c20211691942347.nc"
)
abi_mc06_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C06_G16_s20211691942252_e20211691942315_c20211691942345.nc"
)
abi_mc07_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C07_G16_s20211691942252_e20211691942321_c20211691942355.nc"
)
abi_mc08_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C08_G16_s20211691942252_e20211691942310_c20211691942357.nc"
)
abi_mc09_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C09_G16_s20211691942252_e20211691942315_c20211691942368.nc"
)
abi_mc10_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C10_G16_s20211691942252_e20211691942322_c20211691942353.nc"
)
abi_mc11_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C11_G16_s20211691942252_e20211691942310_c20211691942348.nc"
)
abi_mc12_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C12_G16_s20211691942252_e20211691942316_c20211691942356.nc"
)
abi_mc13_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C13_G16_s20211691942252_e20211691942322_c20211691942361.nc"
)
abi_mc14_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C14_G16_s20211691942252_e20211691942310_c20211691942364.nc"
)
abi_mc15_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C15_G16_s20211691942252_e20211691942316_c20211691942358.nc"
)
abi_mc16_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadM1-M6C16_G16_s20211691942252_e20211691942322_c20211691942366.nc"
)

# add a few g16 conuses to test off-earth pixels
abi_cc01_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadC-M6C01_G16_s20211691941174_e20211691943547_c20211691943589.nc"
)
abi_cc02_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadC-M6C02_G16_s20211691941174_e20211691943547_c20211691943571.nc"
)
abi_cc03_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadC-M6C03_G16_s20211691941174_e20211691943547_c20211691943587.nc"
)
abi_cc07_nc = input_dir.joinpath(
    "abi/OR_ABI-L1b-RadC-M6C07_G16_s20211691941174_e20211691943558_c20211691944002.nc"
)

abi_ncs = [
    abi_cc01_nc,
    abi_cc02_nc,
    abi_cc03_nc,
    abi_cc07_nc,
    abi_mc01_nc,
    abi_mc02_nc,
    abi_mc03_nc,
    abi_mc04_nc,
    abi_mc05_nc,
    abi_mc06_nc,
    abi_mc07_nc,
    abi_mc08_nc,
    abi_mc09_nc,
    abi_mc10_nc,
    abi_mc11_nc,
    abi_mc12_nc,
    abi_mc13_nc,
    abi_mc14_nc,
    abi_mc15_nc,
    abi_mc16_nc,
]

# suvi
suvi_094_nc = input_dir.joinpath(
    "suvi/OR_SUVI-L1b-Fe093_G16_s20203160623501_e20203160623511_c20203160624091.nc"
)
suvi_131_nc = input_dir.joinpath(
    "suvi/OR_SUVI-L1b-Fe131_G16_s20203160623001_e20203160623011_c20203160623196.nc"
)
suvi_171_nc = input_dir.joinpath(
    "suvi/OR_SUVI-L1b-Fe171_G16_s20203160624201_e20203160624211_c20203160624396.nc"
)
suvi_195_nc = input_dir.joinpath(
    "suvi/OR_SUVI-L1b-Fe195_G16_s20203160623301_e20203160623311_c20203160623491.nc"
)
suvi_284_nc = input_dir.joinpath(
    "suvi/OR_SUVI-L1b-Fe284_G16_s20203160624501_e20203160624511_c20203160625090.nc"
)
suvi_304_nc = input_dir.joinpath(
    "suvi/OR_SUVI-L1b-He303_G16_s20203160622501_e20203160622511_c20203160623090.nc"
)
suvi_ncs = [
    suvi_094_nc,
    suvi_131_nc,
    suvi_171_nc,
    suvi_195_nc,
    suvi_284_nc,
    suvi_304_nc,
]


def test_abi_image():
    for abi_nc in abi_ncs:
        abi_image = image.ABIImage(abi_nc, gamma=0.75)

        assert abi_image.rad.dtype == np.float32
        assert abi_image.cmi.dtype == np.float32
        assert abi_image.bv.dtype == np.uint8

        abi_image.save(file_path=output_dir, file_ext=".jpg")

    abi_rgb = image.ABINaturalRGB(
        abi_mc02_nc,
        abi_mc03_nc,
        abi_mc01_nc,
        upscale=True,
        gamma=0.75,
    )

    assert abi_rgb.bv.dtype == np.uint8

    abi_rgb.save(file_path=output_dir, file_ext=".jpeg")

    abi_rgb = image.ABINaturalRGB(abi_cc02_nc, abi_cc03_nc, abi_cc01_nc, gamma=0.75)

    assert abi_rgb.bv.dtype == np.uint8

    abi_rgb.save(file_path=output_dir, file_ext=".jpeg")


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
        upscale=True,
        gamma=1.0,
    )
    abi_projection = projection.ABIProjection(abi_rgb.abi_data)
    abi_rgb = abi_projection.resample2cog(
        abi_rgb.bv, output_dir.joinpath(abi_rgb.default_filename + ".tiff")
    )

    abi_rgb = image.ABINaturalRGB(abi_cc02_nc, abi_cc03_nc, abi_cc01_nc, gamma=0.75)
    abi_projection = projection.ABIProjection(abi_rgb.abi_data)
    abi_rgb = abi_projection.resample2cog(
        abi_rgb.bv, output_dir.joinpath(abi_rgb.default_filename + ".tiff")
    )


def test_navigation():
    idx = (92, 42)

    abi_data = load(abi_mc07_nc)

    abi_nav = navigation.ABINavigation(abi_data, precise_sun=False)

    assert abi_nav.lat_deg.dtype == np.float32
    assert abi_nav.lon_deg.dtype == np.float32
    assert abi_nav.lat_deg[idx] == 44.73149490356445
    assert abi_nav.lon_deg[idx] == -93.01798248291016

    assert abi_nav.area_m.dtype == np.float32
    assert abi_nav.area_m[idx] == 4398247.5

    assert abi_nav.sat_za.dtype == np.float32
    assert abi_nav.sat_az.dtype == np.float32
    assert abi_nav.sat_za[idx] == 0.9509874582290649
    assert abi_nav.sat_az[idx] == 2.7129034996032715

    assert abi_nav.sun_za.dtype == np.float32
    assert abi_nav.sun_az.dtype == np.float32
    assert abi_nav.sun_za[idx] == 0.4886804521083832
    assert abi_nav.sun_az[idx] == 3.9760777950286865

    # test on astropy sun
    abi_nav = navigation.ABINavigation(abi_data, index=idx, precise_sun=True)

    assert abi_nav.sun_za.dtype == np.float32
    assert abi_nav.sun_az.dtype == np.float32
    assert abi_nav.sun_za[0] == 0.4887488782405853
    assert abi_nav.sun_az[0] == 3.976273775100708

    # test with height correction
    abi_nav = navigation.ABINavigation(abi_data, precise_sun=False, hae_m=1.2345678)

    assert abi_nav.lat_deg.dtype == np.float32
    assert abi_nav.lon_deg.dtype == np.float32
    assert abi_nav.lat_deg[idx] == 44.73153305053711
    assert abi_nav.lon_deg[idx] == -93.01801300048828

    assert abi_nav.area_m.dtype == np.float32
    assert abi_nav.area_m[idx] == 4398248.0

    assert abi_nav.sat_za.dtype == np.float32
    assert abi_nav.sat_az.dtype == np.float32
    assert abi_nav.sat_za[idx] == 0.9509882926940918
    assert abi_nav.sat_az[idx] == 2.7129032611846924

    assert abi_nav.sun_za.dtype == np.float32
    assert abi_nav.sun_az.dtype == np.float32
    assert abi_nav.sun_za[idx] == 0.4886806607246399
    assert abi_nav.sun_az[idx] == 3.976076126098633

    # test with astropy sun and height correction
    abi_nav = navigation.ABINavigation(
        abi_data, index=idx, precise_sun=True, hae_m=1.2345678
    )

    assert abi_nav.sun_za.dtype == np.float32
    assert abi_nav.sun_az.dtype == np.float32
    assert abi_nav.sun_za[0] == 0.48874911665916443
    assert abi_nav.sun_az[0] == 3.9762721061706543

    # test on a navigation dataset that does not contain NaNs (G16 meso)
    abi_nav = navigation.ABINavigation(
        abi_data, lat_deg=44.72609499, lon_deg=-93.02279070
    )
    assert abi_nav.index == idx

    # test on a navigation dataset that contains NaNs (G16 CONUS)
    abi_data = load(abi_cc07_nc)
    abi_nav = navigation.ABINavigation(
        abi_data, lat_deg=44.72609499, lon_deg=-93.02279070
    )
    assert abi_nav.index == (192, 1152)


def test_ancillary():
    abi_data = load(abi_cc07_nc)
    water = ancillary.WaterMask(abi_data, rivers=True)

    assert water.data["water_mask"].dtype == bool
    assert water.data["water_mask"].shape == (
        abi_data.dimensions["y"].size,
        abi_data.dimensions["x"].size,
    )

    water.save(save_dir=output_dir)
    cv2.imwrite(str(output_dir.joinpath("water.png")), water.data["water_mask"] * 255)

    iremis = ancillary.IREMIS(abi_data)

    assert iremis.data["c07_land_emissivity"].dtype == np.float32
    assert iremis.data["c07_land_emissivity"].shape == (
        abi_data.dimensions["y"].size,
        abi_data.dimensions["x"].size,
    )

    iremis.save(save_dir=output_dir)
    cv2.imwrite(
        str(output_dir.joinpath("iremis.png")),
        minmax(iremis.data["c07_land_emissivity"]) * 255,
    )


def test_exceptions():
    try:
        abi_data = load("not_a_real_netcdf.nc")

    except Exception as e:
        assert type(e) == exceptions.HereGOESUnsupportedProductException

    try:
        abi_data = load("fake_abi-l1b_netcdf.nc")

    except Exception as e:
        assert type(e) == exceptions.HereGOESIOReadException

    try:
        suvi_data = load("fake_suvi-l1b_netcdf.nc")

    except Exception as e:
        assert type(e) == exceptions.HereGOESIOReadException

    try:
        abi_image = image.ABIImage(abi_mc07_nc)
        abi_image.save(file_path=output_dir, file_ext=".not_a_file_extension")

    except Exception as e:
        assert type(e) == exceptions.HereGOESIOWriteException
