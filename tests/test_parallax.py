# Copyright (c) 2025.

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

from heregoes.core import heregoes_njit
from heregoes.navigation import ABINavigation
from heregoes.navigation._funcs import norm_az
from tests import input_dir

# TODO: We just check that the distances between true and different parallax-corrected points stay the same.
# I would like to test more irregular points around these to demonstrate the behavior around nearest_2d;
# the accuracy of the correction should always be within 1 px in each of y and x (or 1 diagonally)
cases = {
    "mtrainier": {  # a point on Mt. Rainier near the peak
        "nc": input_dir.joinpath(
            "abi-l1b/cases/2019-09-04/OR_ABI-L1b-RadC-M6C02_G16_s20192471701118_e20192471703491_c20192471703540.nc"
        ),
        "feature_true_lat": 46.847876,
        "feature_true_lon": -121.7604712,
        "feature_height": 4206.1700449017335,
        "feature_index": (749, 1483),
    },
    "alamosa": {  # the Alamosa Solar Generating Project in Colorado
        "nc": input_dir.joinpath(
            "abi-l1b/cases/2021-10-07/OR_ABI-L1b-RadC-M6C02_G17_s20212802106176_e20212802108549_c20212802108568.nc"
        ),
        "feature_true_lat": 37.597958411,
        "feature_true_lon": -105.948782805,
        "feature_height": 2295.038888835219,
        "feature_index": (1874, 9896),
    },  # feature_heights from SRTM15+V2 (EGM96) and transformed to a height above WGS84
}


class ParallaxTestCase:
    def __init__(self, this: str):
        self.this = this
        self.this_nc = cases[self.this]["nc"]
        self.this_feature_true_lat = cases[self.this]["feature_true_lat"]
        self.this_feature_true_lon = cases[self.this]["feature_true_lon"]
        self.this_feature_height = cases[self.this]["feature_height"]
        self.this_feature_index = cases[self.this]["feature_index"]

        # not corrected for parallax
        self.uncorrected_nav_from_index = ABINavigation(
            self.this_nc, index=self.this_feature_index
        )

        # when the index and feature height are known, we should be able to get to true lat/lon:
        self.corrected_nav_from_index = ABINavigation(
            self.this_nc,
            index=self.this_feature_index,
            height_m=self.this_feature_height,
        )

        # not corrected for parallax
        self.uncorrected_nav_from_latlon = ABINavigation(
            self.this_nc,
            lat_bounds=self.this_feature_true_lat,
            lon_bounds=self.this_feature_true_lon,
        )

        # when the true lat/lon and feature height are known, we should be able to get to the index:
        self.corrected_nav_from_latlon = ABINavigation(
            self.this_nc,
            lat_bounds=self.this_feature_true_lat,
            lon_bounds=self.this_feature_true_lon,
            height_m=self.this_feature_height,
        )

        # corrected on the sphere
        # when the wrong lat/lon and feature height are known, we should be able to get to a true lat/lon
        self.spherical_lat, self.spherical_lon = self.spherical_parallax_correction(
            lat_deg=self.uncorrected_nav_from_index.lat_deg,
            lon_deg=self.uncorrected_nav_from_index.lon_deg,
            sat_za=self.uncorrected_nav_from_index.sat_za,
            sat_az=self.uncorrected_nav_from_index.sat_az,
            sat_height=self.uncorrected_nav_from_index.abi_data[
                "goes_imager_projection"
            ].perspective_point_height,
            feature_height=self.this_feature_height,
        )

        (
            self.parallax_corrected_dist_to_true_point,
            self.parallax_corrected_az_to_true_point,
            _,
        ) = self.ellipsoidal_distance(
            self.corrected_nav_from_latlon.lat_deg,
            self.corrected_nav_from_latlon.lon_deg,
        )
        self.spherical_dist_to_true_point, self.spherical_az_to_true_point, _ = (
            self.ellipsoidal_distance(self.spherical_lat, self.spherical_lon)
        )

    @staticmethod
    @heregoes_njit
    def spherical_parallax_correction(
        lat_deg, lon_deg, sat_za, sat_az, sat_height, feature_height
    ):
        # corrects lat/lon for parallax on the sphere when the true lat/lon on the ground are unknown.
        # feature_height is typically the derived height of a cloud in a pixel
        # lat_deg/lon_deg are naively given from the satellite navigation, as are sat_za and sat_az
        # https://doi.org/10.1109/LGRS.2013.2283573 eqs. 1-3
        # see also:
        # - http://nwafiles.nwas.org/jom/articles/2023/2023-JOM2/2023-JOM2.pdf
        # - https://www.star.nesdis.noaa.gov/goesr/documents/ATBDs/Enterprise/ATBD_Enterprise_Cloud_Height_v3.4_2020-09.pdf

        H = sat_height
        h = feature_height
        theta = sat_za
        displacement_vector = (h * H * np.tan(theta)) / (H - h)

        y_d = np.deg2rad(lat_deg)
        x_d = np.deg2rad(lon_deg)
        phi = sat_az

        earth_radius_m = 6371.0 * 1000.0

        y_e = y_d + (displacement_vector * np.cos(phi)) / earth_radius_m
        x_e = x_d + (displacement_vector * np.sin(phi)) / (earth_radius_m * np.cos(y_d))

        return (
            np.atleast_1d(np.rad2deg(y_e)).astype(np.float32),
            np.atleast_1d(np.rad2deg(x_e)).astype(np.float32),
        )

    def ellipsoidal_distance(self, lat2, long2):
        # Vincenty's Formulae
        # Adapted from: https://www.johndcook.com/blog/2018/11/24/spheroid-distance/

        lat1 = np.deg2rad(self.this_feature_true_lat)
        long1 = np.deg2rad(self.this_feature_true_lon)
        lat2 = np.deg2rad(lat2)
        long2 = np.deg2rad(long2)

        a = 6378137.0  # equatorial radius in meters
        f = 1 / 298.257223563  # ellipsoid flattening
        b = (1 - f) * a
        tolerance = 1e-11  # to stop iteration

        phi1, phi2 = lat1, lat2
        U1 = np.arctan((1 - f) * np.tan(phi1))
        U2 = np.arctan((1 - f) * np.tan(phi2))
        L1, L2 = long1, long2
        L = L2 - L1

        lambda_old = L + 0

        while True:

            t = (np.cos(U2) * np.sin(lambda_old)) ** 2
            t += (
                np.cos(U1) * np.sin(U2) - np.sin(U1) * np.cos(U2) * np.cos(lambda_old)
            ) ** 2
            sin_sigma = t**0.5
            cos_sigma = np.sin(U1) * np.sin(U2) + np.cos(U1) * np.cos(U2) * np.cos(
                lambda_old
            )
            sigma = np.arctan2(sin_sigma, cos_sigma)

            sin_alpha = np.cos(U1) * np.cos(U2) * np.sin(lambda_old) / sin_sigma
            cos_sq_alpha = 1 - sin_alpha**2
            cos_2sigma_m = cos_sigma - 2 * np.sin(U1) * np.sin(U2) / cos_sq_alpha
            C = f * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha)) / 16

            t = sigma + C * sin_sigma * (
                cos_2sigma_m + C * cos_sigma * (-1 + 2 * cos_2sigma_m**2)
            )
            lambda_new = L + (1 - C) * f * sin_alpha * t
            if abs(lambda_new - lambda_old) <= tolerance:
                break
            else:
                lambda_old = lambda_new

        u2 = cos_sq_alpha * ((a**2 - b**2) / b**2)
        A = 1 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
        B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
        t = cos_2sigma_m + 0.25 * B * (cos_sigma * (-1 + 2 * cos_2sigma_m**2))
        t -= (
            (B / 6)
            * cos_2sigma_m
            * (-3 + 4 * sin_sigma**2)
            * (-3 + 4 * cos_2sigma_m**2)
        )
        delta_sigma = B * sin_sigma * t
        s = b * A * (sigma - delta_sigma)

        alpha1 = np.arctan2(
            np.cos(U2) * np.sin(lambda_new),
            np.cos(U1) * np.sin(U2) - np.sin(U1) * np.cos(U2) * np.cos(lambda_new),
        )
        alpha2 = np.arctan2(
            np.cos(U1) * np.sin(lambda_new),
            -np.sin(U1) * np.cos(U2) + np.cos(U1) * np.sin(U2) * np.cos(lambda_new),
        )

        return s, np.rad2deg(norm_az(alpha1)), np.rad2deg(norm_az(alpha2))


def test_parallax_cases():
    mtrainier_test = ParallaxTestCase("mtrainier")
    # assert(mtrainier_test.parallax_corrected_dist_to_true_point == 145.81930029)
    # assert(mtrainier_test.parallax_corrected_az_to_true_point == 28.610382)
    # assert(mtrainier_test.spherical_dist_to_true_point == 191.88328552246094)
    # assert(mtrainier_test.spherical_az_to_true_point == 359.37774658203125)

    assert mtrainier_test.uncorrected_nav_from_index.lat_deg == 46.90920639038086
    assert mtrainier_test.uncorrected_nav_from_index.lon_deg == -121.88736724853516
    assert mtrainier_test.uncorrected_nav_from_latlon.lat_deg == 46.849021911621094
    assert mtrainier_test.uncorrected_nav_from_latlon.lon_deg == -121.75955200195312
    assert mtrainier_test.corrected_nav_from_index.lat_deg == 46.84967803955078
    assert mtrainier_test.corrected_nav_from_index.lon_deg == -121.76020050048828
    assert mtrainier_test.corrected_nav_from_latlon.lat_deg == 46.849021911621094
    assert mtrainier_test.corrected_nav_from_latlon.lon_deg == -121.75955200195312
    assert mtrainier_test.spherical_lat == 46.84960174560547
    assert mtrainier_test.spherical_lon == -121.76050567626953

    assert (
        mtrainier_test.corrected_nav_from_latlon.index
        == mtrainier_test.this_feature_index
    )

    alamosa_test = ParallaxTestCase("alamosa")
    # assert(alamosa_test.parallax_corrected_dist_to_true_point == 176.79632568359375)
    # assert(alamosa_test.parallax_corrected_az_to_true_point == 2.7376346588134766)
    # assert(alamosa_test.spherical_dist_to_true_point == 260.0396728515625)
    # assert(alamosa_test.spherical_az_to_true_point == 13.685140609741211)

    assert alamosa_test.uncorrected_nav_from_index.lat_deg == 37.62069320678711
    assert alamosa_test.uncorrected_nav_from_index.lon_deg == -105.92235565185547
    assert alamosa_test.uncorrected_nav_from_latlon.lat_deg == 37.59954833984375
    assert alamosa_test.uncorrected_nav_from_latlon.lon_deg == -105.94868469238281
    assert alamosa_test.corrected_nav_from_index.lat_deg == 37.6002311706543
    assert alamosa_test.corrected_nav_from_index.lon_deg == -105.9478530883789
    assert alamosa_test.corrected_nav_from_latlon.lat_deg == 37.59954833984375
    assert alamosa_test.corrected_nav_from_latlon.lon_deg == -105.94868469238281
    assert alamosa_test.spherical_lat == 37.60023498535156
    assert alamosa_test.spherical_lon == -105.94808197021484

    assert (
        alamosa_test.corrected_nav_from_latlon.index == alamosa_test.this_feature_index
    )
