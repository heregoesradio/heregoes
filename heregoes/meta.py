# Copyright (c) 2020, 2021, 2022.

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

"""Storage and utilities for frequently-referenced product metadata"""

import datetime
import os

import netCDF4
import numpy as np

from heregoes import coefficients

noaa_time_format = "%Y-%m-%dT%H:%M:%S.%fZ"
cspp_time_format = "%Y-%m-%d %H:%M:%S.%f"
safe_time_format = "%Y-%m-%dT%H%M%SZ"


class NCMeta:
    """
    Stores general fields for GOES-R netCDF metadata.

    Arguments:
        -  `nc`: String or Path object pointing to a netCDF file for either:
            - GOES-R ABI L1b Radiance, or
            - GOES-R L1b SUVI Imagery
    """

    def __init__(self, nc):
        with netCDF4.Dataset(nc, "r") as loaded_nc:
            self.platform_ID = loaded_nc.platform_ID
            self.platform_ID_safe = "-".join(("GOES", self.platform_ID[-2:]))
            self.instrument_type = loaded_nc.instrument_type
            self.dataset_name = loaded_nc.dataset_name
            self.product_name = "-".join(
                self.dataset_name.split("_")[1].split("-")[0:3]
            )
            self.instrument_type_safe = self.product_name.split("-")[0]
            self.time_coverage_start = norm_date(
                loaded_nc.time_coverage_start, noaa_time_format
            )
            self.time_coverage_end = norm_date(
                loaded_nc.time_coverage_end, noaa_time_format
            )
            self.date_created = norm_date(loaded_nc.date_created, noaa_time_format)

            # this is a nonstandard field added by CSPP GEO. If we can't get it, use the actual file mtime
            try:
                self.local_file_time = norm_date(
                    loaded_nc.cspp_geo_grb_reconstruction_end_time, cspp_time_format
                )

            except:
                self.local_file_time = datetime.datetime.utcfromtimestamp(
                    os.stat(nc).st_mtime
                ).replace(tzinfo=datetime.timezone.utc)

            if self.instrument_type_safe == "ABI":
                self.instrument_meta = _ABIMeta(loaded_nc)
                self.instrument_meta.coefficients = coefficients.ABICoeff(
                    self.platform_ID, self.instrument_meta.band_id.item()
                )

            elif self.instrument_type_safe == "SUVI":
                self.instrument_meta = _SUVIMeta(loaded_nc)
                self.instrument_meta.coefficients = coefficients.SUVICoeff(
                    self.platform_ID, self.instrument_meta.wavelength.item()
                )


class _ABIMeta:
    """Stores ABI-specific fields for GOES-R netCDF metadata"""

    def __init__(self, loaded_nc):
        self.scene_id = loaded_nc.scene_id
        if "Meso" in self.scene_id:
            self.mesoscale_id = loaded_nc.dataset_name.split("-")[2].split("RadM")[1]
            self.scene_id = "Mesoscale " + self.mesoscale_id
            self.scene_id_safe = "Meso" + self.mesoscale_id
        elif self.scene_id == "Full Disk":
            self.scene_id_safe = "FullDisk"
        elif self.scene_id == "CONUS":
            self.scene_id_safe = "CONUS"

        resolution = loaded_nc["Rad"].resolution
        if resolution == "y: 0.000014 rad x: 0.000014 rad":
            self.ifov = 14.0e-6
        elif resolution == "y: 0.000028 rad x: 0.000028 rad":
            self.ifov = 28.0e-6
        elif resolution == "y: 0.000056 rad x: 0.000056 rad":
            self.ifov = 56.0e-6

        self.y = loaded_nc.dimensions["y"].size
        self.x = loaded_nc.dimensions["x"].size
        self.timeline_id = loaded_nc.timeline_id
        self.band_id = loaded_nc["band_id"][0]
        self.band_id_safe = "C" + str(self.band_id).zfill(2)
        self.band_wavelength = round(float(loaded_nc["band_wavelength"][0]), 2)
        self.esd = loaded_nc["earth_sun_distance_anomaly_in_AU"][0]
        self.esun = loaded_nc["esun"][0]
        self.planck_fk1 = loaded_nc["planck_fk1"][0]
        self.planck_fk2 = loaded_nc["planck_fk2"][0]
        self.planck_bc1 = loaded_nc["planck_bc1"][0]
        self.planck_bc2 = loaded_nc["planck_bc2"][0]
        self.midpoint_time = epoch2timestamp(seconds=float(loaded_nc["t"][:].item()))
        self.time_bounds = loaded_nc["time_bounds"][:]
        self.projection_y_coordinate = loaded_nc["y"][:]
        self.y_image_bounds = loaded_nc["y_image_bounds"][:]
        self.projection_x_coordinate = loaded_nc["x"][:]
        self.x_image_bounds = loaded_nc["x_image_bounds"][:]
        self.perspective_point_height = loaded_nc[
            "goes_imager_projection"
        ].perspective_point_height
        self.semi_major_axis = loaded_nc["goes_imager_projection"].semi_major_axis
        self.semi_minor_axis = loaded_nc["goes_imager_projection"].semi_minor_axis
        self.inverse_flattening = loaded_nc["goes_imager_projection"].inverse_flattening
        self.latitude_of_projection_origin = loaded_nc[
            "goes_imager_projection"
        ].latitude_of_projection_origin
        self.longitude_of_projection_origin = loaded_nc[
            "goes_imager_projection"
        ].longitude_of_projection_origin
        self.sweep_angle_axis = loaded_nc["goes_imager_projection"].sweep_angle_axis
        self.y_image = loaded_nc["y_image"][:]
        self.x_image = loaded_nc["x_image"][:]
        self.nominal_satellite_subpoint_lat = loaded_nc[
            "nominal_satellite_subpoint_lat"
        ][0]
        self.nominal_satellite_subpoint_lon = loaded_nc[
            "nominal_satellite_subpoint_lon"
        ][0]
        self.nominal_satellite_height = loaded_nc["nominal_satellite_height"][0]


class _SUVIMeta:
    """Stores SUVI-specific fields for GOES-R netCDF metadata"""

    def __init__(self, loaded_nc):
        # The short exposure has a masked exposure time in netCDF
        self.exposure = loaded_nc["CMD_EXP"][0].data
        self.wavelength = loaded_nc["WAVELNTH"][0].data
        self.wavelength_safe = str(int(loaded_nc["WAVELNTH"][0].data)).zfill(3)
        # Wavelength for SUVI 304 is masked in netCDF
        if self.wavelength_safe == "000":
            self.wavelength_safe = "304"
            self.wavelength = np.array([304], dtype=np.float32)


def norm_date(datestring, fmtstring):
    return datetime.datetime.strptime(datestring, fmtstring).replace(
        tzinfo=datetime.timezone.utc
    )


def epoch2timestamp(seconds):
    # J200 epoch converter
    epoch = datetime.datetime(2000, 1, 1, 12, 0, 0, 0, tzinfo=datetime.timezone.utc)
    timestamp = epoch + datetime.timedelta(seconds=seconds)

    return timestamp


def nc_reader(nc, variable, index=slice(None), attribute=None):
    # generic netCDF inspector
    with netCDF4.Dataset(nc, "r") as loaded_nc:
        if attribute is None:
            return loaded_nc[variable][index]
        else:
            return getattr(loaded_nc[variable], attribute)


def image_filename(image):
    # returns a safe filename using image metadata e.g. "g16_abi_conus_c02" or "g16_suvi_094"
    if image.meta.instrument_type_safe == "ABI":
        filename = (
            "_".join(
                (
                    image.meta.platform_ID,
                    image.meta.instrument_type_safe,
                    image.meta.instrument_meta.scene_id_safe,
                    image.meta.instrument_meta.band_id_safe,
                )
            )
        ).lower()

    elif image.meta.instrument_type_safe == "SUVI":
        filename = (
            "_".join(
                (
                    image.meta.platform_ID,
                    image.meta.instrument_type_safe,
                    image.meta.instrument_meta.wavelength_safe,
                )
            )
        ).lower()

    return filename
