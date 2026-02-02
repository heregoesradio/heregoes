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

from heregoes import load
from heregoes.navigation import ABINavigation
from heregoes.navigation._funcs import inverse_navigate, navigate
from tests.resources_l1b import abi_cc07_nc, abi_mc07_nc

epsilon_float32 = np.finfo(np.float32).eps
epsilon_degrees = 1e-5
epsilon_radians = 1e-6
epsilon_meters = 1


def test_pugvol4_example():
    # https://www.goes-r.gov/users/docs/PUG-GRB-vol4.pdf
    # sections 7.1.2.8.1-7.1.2.8.2 give the following example coordinates on the 2 km CONUS fixed grid,
    # but they don't seem to be real coordinates in the operational 2 km L1b radiance netCDF.
    # so we're just testing our implementation of the equations using the example coordinates as written
    y_rad = 0.095340
    x_rad = -0.024052
    lat_deg = 33.846162
    lon_deg = -84.690932

    loaded = load(abi_cc07_nc)

    def _navigate(y_rad, x_rad):
        return navigate(
            y_rad=y_rad,
            x_rad=x_rad,
            lon_origin=loaded.variables.goes_imager_projection.longitude_of_projection_origin,
            r_eq=loaded.variables.goes_imager_projection.semi_major_axis,
            r_pol=loaded.variables.goes_imager_projection.semi_minor_axis,
            sat_height=loaded.variables.goes_imager_projection.perspective_point_height,
        )

    def _inverse_navigate(lat_deg, lon_deg):
        return inverse_navigate(
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            lon_origin=loaded.variables.goes_imager_projection.longitude_of_projection_origin,
            r_eq=loaded.variables.goes_imager_projection.semi_major_axis,
            r_pol=loaded.variables.goes_imager_projection.semi_minor_axis,
            sat_height=loaded.variables.goes_imager_projection.perspective_point_height,
            feature_height=np.atleast_1d(0.0).astype(np.float32),
        )

    # 7.1.2.8.1 Navigation example
    navigated_lat_deg, navigated_lon_deg = _navigate(
        y_rad=np.atleast_1d(y_rad), x_rad=np.atleast_1d(x_rad)
    )

    assert navigated_lat_deg == lat_deg
    assert navigated_lon_deg == lon_deg

    # 7.1.2.8.2 Inverse navigation example
    navigated_y_rad, navigated_x_rad = _inverse_navigate(
        lat_deg=np.atleast_1d(lat_deg), lon_deg=np.atleast_1d(lon_deg)
    )

    assert navigated_y_rad == y_rad
    assert navigated_x_rad == x_rad


def test_point_navigation():
    idx = (92, 42)

    meso_data = load(abi_mc07_nc)

    meso_nav = ABINavigation(meso_data, precise_sun=False)

    assert meso_nav.lat_deg.dtype == np.float32
    assert meso_nav.lon_deg.dtype == np.float32
    assert meso_nav.lat_deg[idx] == 44.73149490356445
    assert meso_nav.lon_deg[idx] == -93.01798248291016

    assert meso_nav.area_m2.dtype == np.float32
    assert meso_nav.area_m2[idx] == 4398247.5

    assert meso_nav.sat_za.dtype == np.float32
    assert meso_nav.sat_az.dtype == np.float32
    assert meso_nav.sat_za[idx] == 0.9509874582290649
    assert meso_nav.sat_az[idx] == 2.7129034996032715

    assert meso_nav.sun_za.dtype == np.float32
    assert meso_nav.sun_az.dtype == np.float32
    assert meso_nav.sun_za[idx] == 0.4886804521083832
    assert meso_nav.sun_az[idx] == 3.9760777950286865

    # test on astropy sun
    meso_nav = ABINavigation(meso_data, index=idx, precise_sun=True)

    assert meso_nav.sun_za.dtype == np.float32
    assert meso_nav.sun_az.dtype == np.float32
    assert meso_nav.sun_za[0] == 0.4887488782405853
    assert meso_nav.sun_az[0] == 3.976273775100708

    # test with height correction
    meso_nav = ABINavigation(meso_data, precise_sun=False, height_m=1.2345678)

    assert meso_nav.lat_deg.dtype == np.float32
    assert meso_nav.lon_deg.dtype == np.float32
    assert meso_nav.lat_deg[idx] == 44.73153305053711
    assert meso_nav.lon_deg[idx] == -93.01801300048828

    assert meso_nav.area_m2.dtype == np.float32
    assert meso_nav.area_m2[idx] == 4398247.5

    assert meso_nav.sat_za.dtype == np.float32
    assert meso_nav.sat_az.dtype == np.float32
    assert meso_nav.sat_za[idx] == 0.9509882926940918
    assert meso_nav.sat_az[idx] == 2.7129032611846924

    assert meso_nav.sun_za.dtype == np.float32
    assert meso_nav.sun_az.dtype == np.float32
    assert meso_nav.sun_za[idx] == 0.4886806607246399
    assert meso_nav.sun_az[idx] == 3.976076126098633

    # test with astropy sun and height correction
    meso_nav = ABINavigation(meso_data, index=idx, precise_sun=True, height_m=1.2345678)

    assert meso_nav.sun_za.dtype == np.float32
    assert meso_nav.sun_az.dtype == np.float32
    assert meso_nav.sun_za[0] == 0.48874911665916443
    assert meso_nav.sun_az[0] == 3.9762721061706543

    # test single-point indexing on a navigation dataset that does not contain NaNs (G16 meso)
    meso_nav = ABINavigation(meso_data, lat_bounds=44.72609499, lon_bounds=-93.02279070)
    assert meso_nav.index == idx

    # test single-point indexing on a navigation dataset that contains NaNs (G16 CONUS)
    conus_data = load(abi_cc07_nc)
    conus_nav = ABINavigation(
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
    meso_nav_bounded = ABINavigation(
        meso_data, lat_bounds=lat_bounds, lon_bounds=lon_bounds
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
    meso_nav_indexed = ABINavigation(meso_data, index=meso_nav_bounded.index)
    for i in [
        "y_rad",
        "x_rad",
        "lat_deg",
        "lon_deg",
        "sat_za",
        "sat_az",
        "sun_za",
        "sun_az",
        "area_m2",
    ]:
        assert (getattr(meso_nav_indexed, i) == getattr(meso_nav_bounded, i)).all()

    # test conus
    # 2025: set the conus nav times to that of the meso so sun vector can be tested,
    # and return sat, sun vectors in radians
    conus_data = load(abi_cc07_nc)

    # test retrieving points fron multiple lat/lon
    # these are the center lats,lons of Fixed Grid pixels
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
    conus_nav_bounded = ABINavigation(
        conus_data,
        lat_bounds=[lat_bounds[0, 0], lat_bounds[-1, -1]],
        lon_bounds=[lon_bounds[0, 0], lon_bounds[-1, -1]],
        time=meso_nav_bounded.time,
    )

    # check index
    assert conus_nav_bounded.index == np.s_[192:197, 1152:1157]

    # check bounds
    assert (conus_nav_bounded.lat_deg == lat_bounds).all()
    assert (conus_nav_bounded.lon_deg == lon_bounds).all()

    # another nav object using the same derived index should have the same gridded data
    conus_nav_indexed = ABINavigation(
        conus_data, index=conus_nav_bounded.index, time=meso_nav_indexed.time
    )
    for i in [
        "y_rad",
        "x_rad",
        "lat_deg",
        "lon_deg",
        "sat_za",
        "sat_az",
        "sun_za",
        "sun_az",
        "area_m2",
    ]:
        assert (getattr(conus_nav_indexed, i) == getattr(conus_nav_bounded, i)).all()

    # meso and conus nav reference the same area, the time-invariant gridded data should match within tolerance
    assert (
        np.abs(conus_nav_bounded.lat_deg - meso_nav_bounded.lat_deg) < epsilon_degrees
    ).all()
    assert (
        np.abs(conus_nav_bounded.lon_deg - meso_nav_bounded.lon_deg) < epsilon_degrees
    ).all()
    assert (
        np.abs(conus_nav_bounded.sat_za - meso_nav_bounded.sat_za) < epsilon_radians
    ).all()
    assert (
        np.abs(conus_nav_bounded.sat_az - meso_nav_bounded.sat_az) < epsilon_radians
    ).all()
    assert (
        np.abs(conus_nav_bounded.sun_za - meso_nav_bounded.sun_za) < epsilon_radians
    ).all()
    assert (
        np.abs(conus_nav_bounded.sun_az - meso_nav_bounded.sun_az) < epsilon_radians
    ).all()
    assert (
        np.abs(conus_nav_bounded.area_m2 - meso_nav_bounded.area_m2) < epsilon_meters
    ).all()
