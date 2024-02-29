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

import logging

import astropy.units as u
import numpy as np
from astropy import coordinates
from astropy.time import Time

from heregoes import exceptions
from heregoes.util import nearest_2d, njit, orbital, rad2deg

logger = logging.getLogger("heregoes-logger")


class ABINavigation:
    """
    This is a class for GOES-R ABI navigation routines using an ABIObject to access netCDF data.
    The routines may be constrained to a single index or continuous slice of the ABI Fixed Grid with `index`, or to a single area by providing `lat_bounds` and `lon_bounds` (degrees).
    All calculations return 32-bit floating point NumPy arrays which should be accurate enough for most applications at ABI scale.

    The following quantities are always generated upon instantiation:
        - Instrument scanning angle (`y_rad`, `x_rad`)
        - Latitude and longitude of pixels from instrument scanning angle (`lat_deg`, `lon_deg`)
            OR the index or continuous slice of an area on the ABI Fixed Grid (`index`) from latitude and longitude (degrees)

    These quantities may be generated upon access after instantiation:
        - Angles of the satellite vector at the pixel (`sat_za`, `sat_az`)
        - Angles of the Sun vector at the pixel (`sun_za`, `sun_az`)
        - The effective projected area of the pixel in square meters (`area_m`)

    Arguments:
        - `abi_data`: The ABIObject formed on a GOES-R ABI L1b Radiance netCDF file as returned by `heregoes.load()`
        - `index`: Optionally constrains the navigation routines to an array index or continuous slice on the ABI Fixed Grid
        - `lat_bounds`, `lon_bounds`: Optionally constrains the navigation routines to a latitude and longitude bounding box defined by the upper left and lower right points, e.g. `lat_bounds=[ul_lat, lr_lat]`, `lon_bounds=[ul_lon, lr_lon]`
        - `hae_m`: The Height Above Ellipsoid (HAE) in meters of the ABI array to correct for terrain height. Default 0.0 (no correction)
        - `time`: The time for which the Sun position is valid. The product midpoint time is used if not provided
        - `precise_sun`: Whether to calculate solar position using Equation of Time with Pyorbital (`False`, default) or real ephemeris with Astropy (`True`)
        - `degrees`: Whether to return calculated Sun/satellite vector angles in radians or degrees. Default `False`
    """

    def __init__(
        self,
        abi_data,
        index=None,
        lat_bounds=None,
        lon_bounds=None,
        hae_m=0.0,
        time=None,
        precise_sun=False,
        degrees=False,
    ):
        self.abi_data = abi_data
        self.index = index
        self.hae_m = np.atleast_1d(hae_m).astype(np.float32)
        self.time = time
        self.precise_sun = precise_sun
        self.degrees = degrees

        self._sat_za = None
        self._sat_az = None
        self._sun_za = None
        self._sun_az = None
        self._area_m = None

        if self.index is None:
            self.index = np.s_[:, :]

        if self.time is None:
            self.time = self.abi_data.midpoint_time

        self.x_rad, self.y_rad = np.meshgrid(
            self.abi_data["x"][self.index[1]],
            self.abi_data["y"][self.index[0]],
        )

        if lat_bounds is None or lon_bounds is None:
            self.lat_deg, self.lon_deg = self.navigate(
                self.y_rad,
                self.x_rad,
                lon_origin=self.abi_data[
                    "goes_imager_projection"
                ].longitude_of_projection_origin,
                r_eq=self.abi_data["goes_imager_projection"].semi_major_axis,
                r_pol=self.abi_data["goes_imager_projection"].semi_minor_axis,
                sat_height=self.abi_data[
                    "goes_imager_projection"
                ].perspective_point_height,
            )

            # correct for terrain parallax if HAE is provided
            if (self.hae_m != 0.0).any():
                self.y_rad, self.x_rad = self.reverse_navigate(
                    self.lat_deg,
                    self.lon_deg,
                    lon_origin=self.abi_data[
                        "goes_imager_projection"
                    ].longitude_of_projection_origin,
                    r_eq=self.abi_data["goes_imager_projection"].semi_major_axis,
                    r_pol=self.abi_data["goes_imager_projection"].semi_minor_axis,
                    sat_height=self.abi_data[
                        "goes_imager_projection"
                    ].perspective_point_height,
                    feature_height=np.broadcast_to(self.hae_m, self.lat_deg.shape),
                )
                self.lat_deg, self.lon_deg = self.navigate(
                    self.y_rad,
                    self.x_rad,
                    lon_origin=self.abi_data[
                        "goes_imager_projection"
                    ].longitude_of_projection_origin,
                    r_eq=self.abi_data["goes_imager_projection"].semi_major_axis,
                    r_pol=self.abi_data["goes_imager_projection"].semi_minor_axis,
                    sat_height=self.abi_data[
                        "goes_imager_projection"
                    ].perspective_point_height,
                )

        else:
            lat_bounds = np.atleast_1d(lat_bounds).astype(np.float32)
            lon_bounds = np.atleast_1d(lon_bounds).astype(np.float32)

            if lat_bounds.shape != lon_bounds.shape:
                raise exceptions.HereGOESValueError(
                    msg="`lat_bounds` and `lon_bounds` must be the same shape.",
                    caller=f"{__name__}.{self.__class__.__name__}",
                )

            if np.isnan(lat_bounds).any() | np.isnan(lon_bounds).any():
                logger.warning(
                    "Provided lat/lon bounds contain NaN; check bounds for off-Earth pixels.",
                    extra={"caller": f"{__name__}.{self.__class__.__name__}"},
                )

            # this is tricky to deal with because of Numba broadcasting issues, and the desired shape of hae_m being unknown at this step when using lat/lon bounds
            # related: https://github.com/numba/numba/issues/4632
            if (self.hae_m != 0.0).any():
                logger.warning(
                    "Terrain height correction is not currently supported when `lat_bounds` and `lon_bounds` are specified; ignoring argument `hae_m`.",
                    extra={"caller": f"{__name__}.{self.__class__.__name__}"},
                )

            derived_y_rad, derived_x_rad = self.reverse_navigate(
                lat_bounds,
                lon_bounds,
                lon_origin=self.abi_data[
                    "goes_imager_projection"
                ].longitude_of_projection_origin,
                r_eq=self.abi_data["goes_imager_projection"].semi_major_axis,
                r_pol=self.abi_data["goes_imager_projection"].semi_minor_axis,
                sat_height=self.abi_data[
                    "goes_imager_projection"
                ].perspective_point_height,
                feature_height=np.broadcast_to(
                    np.atleast_1d(0.0).astype(np.float32), lat_bounds.shape
                ),
            )

            self.index = nearest_2d(
                self.y_rad, self.x_rad, derived_y_rad, derived_x_rad
            )

            self.y_rad = np.atleast_1d(self.y_rad[self.index]).astype(np.float32)
            self.x_rad = np.atleast_1d(self.x_rad[self.index]).astype(np.float32)

            self.lat_deg, self.lon_deg = self.navigate(
                self.y_rad,
                self.x_rad,
                lon_origin=self.abi_data[
                    "goes_imager_projection"
                ].longitude_of_projection_origin,
                r_eq=self.abi_data["goes_imager_projection"].semi_major_axis,
                r_pol=self.abi_data["goes_imager_projection"].semi_minor_axis,
                sat_height=self.abi_data[
                    "goes_imager_projection"
                ].perspective_point_height,
            )

            if self.hae_m.shape != self.lat_deg.shape:
                self.hae_m = np.broadcast_to(self.hae_m, self.lat_deg.shape)

    @property
    def sat_za(self):
        if self._sat_za is None:
            self._calc_sat()

        return self._sat_za

    @sat_za.setter
    def sat_za(self, value):
        self._sat_za = value

    @property
    def sat_az(self):
        if self._sat_az is None:
            self._calc_sat()

        return self._sat_az

    @sat_az.setter
    def sat_az(self, value):
        self._sat_az = value

    @property
    def sun_za(self):
        if self._sun_za is None:
            self._calc_sun()

        return self._sun_za

    @sun_za.setter
    def sun_za(self, value):
        self._sun_za = value

    @property
    def sun_az(self):
        if self._sun_az is None:
            self._calc_sun()

        return self._sun_az

    @sun_az.setter
    def sun_az(self, value):
        self._sun_az = value

    @property
    def area_m(self):
        if self._area_m is None:
            self._area_m = self.pixel_area(
                self.y_rad,
                self.x_rad,
                self.abi_data["goes_imager_projection"].semi_major_axis,
                self.abi_data["goes_imager_projection"].perspective_point_height,
                self.abi_data.resolution_ifov,
            )

        return self._area_m

    @area_m.setter
    def area_m(self, value):
        self._area_m = value

    def _calc_sat(self):
        self._sat_az, self._sat_za = orbital.get_observer_look(
            sat_lon=np.atleast_1d(self.abi_data["nominal_satellite_subpoint_lon"][...]),
            sat_lat=np.atleast_1d(self.abi_data["nominal_satellite_subpoint_lat"][...]),
            sat_alt=np.atleast_1d(self.abi_data["nominal_satellite_height"][...]),
            jdays2000=orbital.jdays2000(self.time),
            lon=self.lon_deg,
            lat=self.lat_deg,
            alt=self.hae_m / 1000.0,
        )

        # normalize azimuth to North-clockwise convention between 0 and 2pi
        self._sat_az = orbital.norm_az(self._sat_az)
        # pyorbital functions output elevation/altitude, convert to zenith angle
        self._sat_za = orbital.el2za(self._sat_za)

        if self.degrees:
            self._sat_az = rad2deg(self._sat_az)
            self._sat_za = rad2deg(self._sat_za)

    def _calc_sun(self):
        if self.precise_sun:
            earth_position = coordinates.EarthLocation.from_geodetic(
                lat=self.lat_deg * u.deg,
                lon=self.lon_deg * u.deg,
                height=self.hae_m * u.m,
                ellipsoid="GRS80",
            )
            sun_position = coordinates.get_sun(Time(self.time)).transform_to(
                coordinates.AltAz(obstime=Time(self.time), location=earth_position)
            )
            self._sun_az = np.atleast_1d(sun_position.az.rad).astype(np.float32)
            self._sun_za = np.atleast_1d(sun_position.alt.rad).astype(np.float32)

        else:
            self._sun_za, self._sun_az = orbital.get_alt_az(
                jdays2000=orbital.jdays2000(self.time),
                lon=self.lon_deg,
                lat=self.lat_deg,
            )

        # normalize azimuth to North-clockwise convention between 0 and 2pi
        self._sun_az = orbital.norm_az(self._sun_az)
        # pyorbital functions output elevation/altitude, convert to zenith angle
        self._sun_za = orbital.el2za(self._sun_za)

        if self.degrees:
            self._sun_az = rad2deg(self._sun_az)
            self._sun_za = rad2deg(self._sun_za)

    @staticmethod
    @njit.heregoes_njit
    def navigate(y_rad, x_rad, lon_origin, r_eq, r_pol, sat_height):
        # navigates instrument scanning angle to latitude and longitude
        # following 7.1.2.8.1 in the PUG Volume 4: https://www.goes-r.gov/users/docs/PUG-GRB-vol4.pdf

        H = sat_height + r_eq

        lambda_0 = np.deg2rad(lon_origin)

        a = np.square(np.sin(x_rad)) + (
            np.square(np.cos(x_rad))
            * (
                np.square(np.cos(y_rad))
                + (((np.square(r_eq)) / (np.square(r_pol))) * np.square(np.sin(y_rad)))
            )
        )
        b = -2.0 * H * np.cos(x_rad) * np.cos(y_rad)
        c = np.square(H) - np.square(r_eq)

        r_s = (-b - np.sqrt((np.square(b)) - (4.0 * a * c))) / (2.0 * a)

        s_x = r_s * np.cos(x_rad) * np.cos(y_rad)
        s_y = -r_s * np.sin(x_rad)
        s_z = r_s * np.cos(x_rad) * np.sin(y_rad)

        lat_deg = np.rad2deg(
            np.arctan(
                ((np.square(r_eq)) / (np.square(r_pol)))
                * ((s_z / np.sqrt((np.square((H - s_x))) + (np.square(s_y)))))
            )
        )
        lon_deg = np.rad2deg(lambda_0 - np.arctan(s_y / (H - s_x)))

        return (
            np.atleast_1d(lat_deg).astype(np.float32),
            np.atleast_1d(lon_deg).astype(np.float32),
        )

    @staticmethod
    @njit.heregoes_njit_noparallel
    def reverse_navigate(
        lat_deg, lon_deg, lon_origin, r_eq, r_pol, sat_height, feature_height
    ):
        # navigates latitude and longitude to instrument scanning angle
        # following 7.1.2.8.2 in the PUG Volume 4: https://www.goes-r.gov/users/docs/PUG-GRB-vol4.pdf

        phi = np.deg2rad(lat_deg)
        lambda_ = np.deg2rad(lon_deg)

        e = 0.0818191910435

        H = sat_height + r_eq

        lambda_0 = np.deg2rad(lon_origin)

        phi_c = np.arctan((np.square(r_pol) / np.square(r_eq)) * np.tan(phi))

        r_c = r_pol / np.sqrt(1 - np.square(e) * np.square(np.cos(phi_c)))
        r_c -= feature_height

        s_x = H - r_c * np.cos(phi_c) * np.cos(lambda_ - lambda_0)
        s_y = -r_c * np.cos(phi_c) * np.sin(lambda_ - lambda_0)
        s_z = r_c * np.sin(phi_c)

        y_rad = np.arctan(s_z / s_x)
        x_rad = np.arcsin(
            -s_y / np.sqrt(np.square(s_x) + np.square(s_y) + np.square(s_z))
        )

        return (
            np.atleast_1d(y_rad).astype(np.float32),
            np.atleast_1d(x_rad).astype(np.float32),
        )

    @staticmethod
    @njit.heregoes_njit
    def pixel_area(y_rad, x_rad, semi_major_axis, perspective_point_height, ifov):
        # returns the effective area of a pixel in meters

        r = semi_major_axis
        sh = perspective_point_height
        beta = ifov

        # https://doi.org/10.1017/CBO9781139029346.005 eqs. 3.10 a-b
        # cross-track (N-S)
        alpha_c = y_rad
        delta = np.arcsin(((sh + r) / r) * np.sin(alpha_c)) - alpha_c
        Lc = 2 * (((r * np.sin(delta)) / np.sin(alpha_c)) * np.tan(beta / 2.0))

        # https://doi.org/10.1017/CBO9781139029346.005 eqs. 3.7 - 3.9
        # along-track (W-E)
        alpha_a = x_rad
        alpha_a1 = alpha_a - beta / 2.0
        alpha_a2 = alpha_a + beta / 2.0
        L1 = r * (np.arcsin(((sh + r) / r) * np.sin(alpha_a1)) - alpha_a1)
        L2 = r * (np.arcsin(((sh + r) / r) * np.sin(alpha_a2)) - alpha_a2)
        La = L2 - L1

        return np.atleast_1d(Lc * La).astype(np.float32)
