# Copyright (c) 2020-2025.

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

import datetime
from typing import Optional

import astropy.units as u
import numpy as np
from astropy import coordinates
from astropy.time import Time

from heregoes import load
from heregoes.core.types import ABIInputType, FixedGridDataType, FixedGridIndexType
from heregoes.navigation._funcs import (
    el2za,
    fractional_jd,
    inverse_navigate,
    navigate,
    norm_az,
    pixel_area,
    pixel_height,
    pixel_width,
)
from heregoes.navigation._orbital import get_alt_az, get_observer_look
from heregoes.util import nearest_2d_search


class ABINavigation:
    """
    ### Navigation and indexing with parallax correction on the ABI Fixed Grid

    ### Parameters:
        - `abi_data`:
            - Either a str or Path referencing an ABI L1b/L2+ netCDF file,
            - or the `ABIL1bData` or `ABIL2Data` object formed by `heregoes.load()` on the path

        - `index` (optional): 2D array index or slice to select a subset of the ABI Fixed Grid, e.g.:
            - `(y, x)`
            - `np.s_[y1:y2, x1:x2]`
            - `(slice(y1, y2, None), slice(x1, x2, None))`

        - `lat_bounds`, `lon_bounds` (optional): Instead of `index`, use geodetic latitude and longitude (degrees) to select a point or slice of the Fixed Grid, e.g.:
            - `lat_bounds=point_lat`, `lon_bounds=point_lon`
            - `lat_bounds=[ul_lat, lr_lat]`, `lon_bounds=[ul_lon, lr_lon]`

        - `height_m` (optional): If subsetting the navigation by `index` or `lat_bounds` and `lon_bounds`, provide the height in meters relative to the GRS80 at the bounding points. Default 0.0 (no correction)
            - `height_m=1234.0`
            - `height_m=[ul_m, lr_m]`

        - `precise_sun` (optional): Whether to calculate solar position using Equation of Time with Pyorbital (`False`, default) or real ephemeris with Astropy (`True`, slower)

        - `time` (optional): UTC time for which the Sun position is valid. The product midpoint time is used if not provided

        - `degrees` (optional): Whether to return calculated Sun/satellite vector angles in radians or degrees. Default `False`

        - `resample_nav` (optional): Determines whether to resample the navigation to fit the ABI image (`True`), or resample the ABI image to fit the navigation (`False`, default)
            - *Only takes effect when `lat_bounds` and `lon_bounds` are provided with `height_m` for parallax correction*

    ### Attributes:
        - `y_rad`, `x_rad`:
            - Fixed Grid (y, x) coordinates in radians

        - `index`:
            - The currently selected index or slice of the Fixed Grid. May be derived from geodetic coordinates if `lat_bounds` and `lon_bounds` are provided

        - `lat_deg`, `lon_deg`:
            - Geodetic coordinates navigated from Fixed Grid instrument scan angles

        - `sat_za`, `sat_az`:
            - Satellite zenith angle and azimuth at the center of each pixel. Returned in radians if the `degrees` argument was `False` (default)

        - `sun_za`, `sun_az`:
            - Sun zenith angle and azimuth at the center of each pixel. Returned in radians if the `degrees` argument was `False` (default)

        - `area_m2`:
            - The effective ground area of each Fixed Grid pixel in meters squared, increasing from nadir

        - `along_track_m`, `cross_track_m`:
            - The effective along-track width and cross-track height of each pixel in meters
    """

    # TODO: implement pixelwise acquisition time given in the L1b reprocessed (RP) product

    def __init__(
        self,
        abi_data: ABIInputType,
        index: Optional[FixedGridIndexType] = None,
        lat_bounds: Optional[FixedGridDataType] = None,
        lon_bounds: Optional[FixedGridDataType] = None,
        height_m: Optional[FixedGridDataType] = 0.0,
        time: Optional[datetime.datetime] = None,
        precise_sun: Optional[bool] = False,
        degrees: Optional[bool] = False,
        resample_nav: Optional[bool] = False,
    ):
        self.abi_data = load(abi_data)
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.height_m = np.atleast_1d(height_m).astype(np.float32)
        self.time = time
        self.precise_sun = precise_sun
        self.degrees = degrees
        self.resample_nav = resample_nav

        self._y_rad = None
        self._x_rad = None
        self._index = None
        self._lat_deg = None
        self._lon_deg = None
        self._cross_track_m = None
        self._along_track_m = None
        self._area_m2 = None
        self._sat_za = None
        self._sat_az = None
        self._sun_za = None
        self._sun_az = None

        if self.time is None:
            self.time = self.abi_data.midpoint_time

        self._index_mode = self.lat_bounds is None or self.lon_bounds is None
        if self._index_mode:
            if index is None:
                self._index = np.s_[:, :]
            else:
                self._index = index

            self._y_rad = self.abi_data["y"][self._index[0]]
            self._x_rad = self.abi_data["x"][self._index[1]]

            self._setup = self._from_index

        else:
            self.lat_bounds = np.atleast_1d(self.lat_bounds).astype(np.float32)
            self.lon_bounds = np.atleast_1d(self.lon_bounds).astype(np.float32)

            if self.lat_bounds.shape != self.lon_bounds.shape:
                raise ValueError(
                    "`self.lat_bounds` and `self.lon_bounds` must be the same shape."
                )

            if np.isnan(self.lat_bounds).any() | np.isnan(self.lon_bounds).any():
                raise ValueError(
                    "`self.lat_bounds` and `self.lon_bounds` cannot contain NaN."
                )

            if index is not None:
                raise ValueError(
                    "`Cannot use a Fixed Grid `index` when also using `lat_bounds` and `lon_bounds`."
                )

            self._setup = self._from_latlon

    def _from_index(self):
        x_rad_mesh, y_rad_mesh = np.meshgrid(self._x_rad, self._y_rad, copy=False)

        # silences the numpy warning about future broadcasted arrays not being writeable
        y_rad_mesh.flags.writeable = False
        x_rad_mesh.flags.writeable = False

        # get points on the ellipsoid
        self._lat_deg, self._lon_deg = self._navigate(
            y_rad=y_rad_mesh, x_rad=x_rad_mesh
        )

        # correct for parallax if HAE is provided
        if (self.height_m != 0.0).any():
            # what would the scanning angles y_rad, x_rad be if the feature were pushed back down onto the ellipsoid?
            corrected_y_rad_mesh, corrected_x_rad_mesh = self._inverse_navigate(
                lat_deg=self._lat_deg,
                lon_deg=self._lon_deg,
                feature_height=np.broadcast_to(-self.height_m, self._lat_deg.shape),
            )

            # get corrected lat/lon from the parallax-corrected instrument scanning angles
            self._lat_deg, self._lon_deg = self._navigate(
                y_rad=corrected_y_rad_mesh, x_rad=corrected_x_rad_mesh
            )

            # and set the scanning angles to the corrected ones
            self._y_rad = corrected_y_rad_mesh[:, 0]
            self._x_rad = corrected_x_rad_mesh[0, :]

    def _from_latlon(self):
        self._y_rad = self.abi_data["y"][...]
        self._x_rad = self.abi_data["x"][...]

        # get the scanning angles that correspond to the inputted lat/lon (not considering parallax)
        uncorrected_y_rad, uncorrected_x_rad = self._inverse_navigate(
            lat_deg=self.lat_bounds,
            lon_deg=self.lon_bounds,
            feature_height=np.broadcast_to(np.float32(0.0), self.lat_bounds.shape),
        )

        # and find their closest index slice in the full set of scan angles
        uncorrected_index = nearest_2d_search(
            self._y_rad, self._x_rad, uncorrected_y_rad, uncorrected_x_rad
        )

        # don't correct for parallax if no HAE is provided
        if (self.height_m == 0.0).all():
            image_index = uncorrected_index
            nav_index = uncorrected_index

        # otherwise, either resample the navigation to fit the displaced image pixels,
        elif self.resample_nav:
            image_index = uncorrected_index

            negative_corrected_y_rad, negative_corrected_x_rad = self._inverse_navigate(
                lat_deg=self.lat_bounds,
                lon_deg=self.lon_bounds,
                feature_height=np.broadcast_to(
                    -self.height_m,  # what would scanning angles be if the feature were pushed down onto the ellipsoid?
                    self.lat_bounds.shape,
                ).astype(np.float32),
            )
            nav_index = nearest_2d_search(
                self._y_rad,
                self._x_rad,
                negative_corrected_y_rad,
                negative_corrected_x_rad,
            )

        # or resample the index to fit the provided lat, lon, and height
        else:
            positive_corrected_y_rad, positive_corrected_x_rad = self._inverse_navigate(
                lat_deg=self.lat_bounds,
                lon_deg=self.lon_bounds,
                feature_height=np.broadcast_to(
                    self.height_m,  # push the feature up off the ellipsoid
                    self.lat_bounds.shape,
                ).astype(np.float32),
            )
            image_index = nearest_2d_search(
                self._y_rad,
                self._x_rad,
                positive_corrected_y_rad,
                positive_corrected_x_rad,
            )

            nav_index = uncorrected_index

        # the index that gets exposed is the image_index
        self._index = image_index

        # get the closest-fit Fixed Grid coordinate mesh using the nav_index
        nav_y_idx = nav_index[0]
        nav_x_idx = nav_index[1]

        # try to avoid making a huge meshgrid
        if self._y_rad[nav_y_idx].ndim == self._x_rad[nav_x_idx].ndim == 1:
            x_rad_mesh, y_rad_mesh = np.meshgrid(
                self._x_rad[nav_x_idx],
                self._y_rad[nav_y_idx],
                copy=False,
            )

        else:
            x_rad_mesh, y_rad_mesh = np.meshgrid(
                self._x_rad,
                self._y_rad,
                copy=False,
            )
            y_rad_mesh = y_rad_mesh[nav_index]
            x_rad_mesh = x_rad_mesh[nav_index]

        self._lat_deg, self._lon_deg = self._navigate(
            y_rad=np.atleast_1d(y_rad_mesh).astype(np.float32),
            x_rad=np.atleast_1d(x_rad_mesh).astype(np.float32),
        )

        self._y_rad = self._y_rad[nav_y_idx]
        self._x_rad = self._x_rad[nav_x_idx]

    @property
    def y_rad(self):
        if self._y_rad is None:
            self._setup()

        return self._y_rad

    @property
    def x_rad(self):
        if self._x_rad is None:
            self._setup()

        return self._x_rad

    @property
    def index(self):
        if self._index is None and not self._index_mode:
            self._setup()

        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def lat_deg(self):
        if self._lat_deg is None:
            self._setup()

        return self._lat_deg

    @property
    def lon_deg(self):
        if self._lon_deg is None:
            self._setup()

        return self._lon_deg

    @property
    def cross_track_m(self):
        if self._cross_track_m is None:
            self._cross_track_m = pixel_height(
                y_rad=self.y_rad,
                r_eq=self.abi_data["goes_imager_projection"].semi_major_axis,
                sat_height=self.abi_data[
                    "goes_imager_projection"
                ].perspective_point_height,
                ifov=self.abi_data.resolution_ifov,
            )

        return self._cross_track_m

    @property
    def along_track_m(self):
        if self._along_track_m is None:
            self._along_track_m = pixel_width(
                x_rad=self.x_rad,
                r_eq=self.abi_data["goes_imager_projection"].semi_major_axis,
                sat_height=self.abi_data[
                    "goes_imager_projection"
                ].perspective_point_height,
                ifov=self.abi_data.resolution_ifov,
            )

        return self._along_track_m

    @property
    def area_m2(self):
        if self._area_m2 is None:
            self._area_m2 = pixel_area(
                cross_track=self.cross_track_m,
                along_track=self.along_track_m,
            )

        return self._area_m2

    @property
    def sat_za(self):
        if self._sat_za is None:
            self._calc_sat()

        return self._sat_za

    @property
    def sat_az(self):
        if self._sat_az is None:
            self._calc_sat()

        return self._sat_az

    @property
    def sun_za(self):
        if self._sun_za is None:
            self._calc_sun()

        return self._sun_za

    @property
    def sun_az(self):
        if self._sun_az is None:
            self._calc_sun()

        return self._sun_az

    def _navigate(self, y_rad, x_rad):
        return navigate(
            y_rad=y_rad,
            x_rad=x_rad,
            lon_origin=self.abi_data[
                "goes_imager_projection"
            ].longitude_of_projection_origin,
            r_eq=self.abi_data["goes_imager_projection"].semi_major_axis,
            r_pol=self.abi_data["goes_imager_projection"].semi_minor_axis,
            sat_height=self.abi_data["goes_imager_projection"].perspective_point_height,
        )

    def _inverse_navigate(self, lat_deg, lon_deg, feature_height):
        return inverse_navigate(
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            lon_origin=self.abi_data[
                "goes_imager_projection"
            ].longitude_of_projection_origin,
            r_eq=self.abi_data["goes_imager_projection"].semi_major_axis,
            r_pol=self.abi_data["goes_imager_projection"].semi_minor_axis,
            sat_height=self.abi_data["goes_imager_projection"].perspective_point_height,
            feature_height=feature_height,
        )

    def _calc_sat(self):
        # calculate satellite look vector
        self._sat_az, self._sat_za = get_observer_look(
            sat_lon=np.broadcast_to(
                self.abi_data["nominal_satellite_subpoint_lon"][...], self.lat_deg.shape
            ),
            sat_lat=np.broadcast_to(
                self.abi_data["nominal_satellite_subpoint_lat"][...], self.lat_deg.shape
            ),
            sat_alt=np.broadcast_to(
                self.abi_data["nominal_satellite_height"][...], self.lat_deg.shape
            ),
            jdays2000=np.broadcast_to(fractional_jd(self.time), self.lat_deg.shape),
            lon=self.lon_deg,
            lat=self.lat_deg,
            alt=np.broadcast_to(self.height_m / 1000.0, self.lat_deg.shape),
        )

        # normalize azimuth to North-clockwise convention between 0 and 2pi
        self._sat_az = norm_az(self._sat_az, degrees=self.degrees)
        # pyorbital functions output elevation/altitude, convert to zenith angle
        self._sat_za = el2za(self._sat_za, degrees=self.degrees)

    def _calc_sun(self):
        # calculate Sun vector
        if self.precise_sun:
            earth_position = coordinates.EarthLocation.from_geodetic(
                lat=self.lat_deg * u.deg,
                lon=self.lon_deg * u.deg,
                height=self.height_m * u.m,
                ellipsoid="GRS80",
            )
            sun_position = coordinates.get_sun(Time(self.time)).transform_to(
                coordinates.AltAz(obstime=Time(self.time), location=earth_position)
            )
            self._sun_az = np.atleast_1d(sun_position.az.rad).astype(np.float32)
            self._sun_za = np.atleast_1d(sun_position.alt.rad).astype(np.float32)

        else:
            self._sun_za, self._sun_az = get_alt_az(
                jdays2000=fractional_jd(self.time),
                lon=self.lon_deg,
                lat=self.lat_deg,
            )

        # normalize azimuth to North-clockwise convention between 0 and 2pi
        self._sun_az = norm_az(self._sun_az, degrees=self.degrees)
        # pyorbital functions output elevation/altitude, convert to zenith angle
        self._sun_za = el2za(self._sun_za, degrees=self.degrees)
