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

"""GOES-R specific conversions and corrections for netCDF data on top of an NCInterface"""

import datetime

from heregoes.goesr import coefficients
from heregoes.util.ncinterface import NCInterface

noaa_time_format = "%Y-%m-%dT%H:%M:%S.%fZ"
cspp_time_format = "%Y-%m-%d %H:%M:%S.%f"


class GOESRObject(NCInterface):
    """Custom generic attributes on top of GOES-R netCDF"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.platform_ID_safe = "-".join(("GOES", self.platform_ID[-2:]))
        self.product_name = "-".join(self.dataset_name.split("_")[1].split("-")[0:3])
        self.instrument_type_safe = self.product_name.split("-")[0]
        self.time_coverage_start = self.norm_date(
            self.time_coverage_start, noaa_time_format
        )
        self.time_coverage_end = self.norm_date(
            self.time_coverage_end, noaa_time_format
        )
        self.date_created = self.norm_date(self.date_created, noaa_time_format)

        # this is a nonstandard field added by CSPP GEO. If we can't get it, use the actual file mtime
        try:
            self.local_file_time = self.norm_date(
                self.cspp_geo_grb_reconstruction_end_time, cspp_time_format
            )

        except:
            self.local_file_time = datetime.datetime.utcfromtimestamp(
                self._nc_file.stat().st_mtime
            ).replace(tzinfo=datetime.timezone.utc)

    def norm_date(self, datestring, fmtstring):
        return datetime.datetime.strptime(datestring, fmtstring).replace(
            tzinfo=datetime.timezone.utc
        )

    def epoch2timestamp(self, seconds):
        epoch = datetime.datetime(2000, 1, 1, 12, 0, 0, 0, tzinfo=datetime.timezone.utc)
        timestamp = epoch + datetime.timedelta(seconds=seconds)

        return timestamp


class ABIObject(GOESRObject):
    """Custom ABI attributes on top of GOES-R ABI netCDF"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "Meso" in self.scene_id:
            self.mesoscale_id = self.dataset_name.split("-")[2].split("RadM")[1]
            self.scene_id = "Mesoscale " + self.mesoscale_id
            self.scene_id_safe = "Meso" + self.mesoscale_id
        elif self.scene_id == "Full Disk":
            self.scene_id_safe = "FullDisk"
        elif self.scene_id == "CONUS":
            self.scene_id_safe = "CONUS"

        if self["Rad"].resolution == "y: 0.000014 rad x: 0.000014 rad":
            self.resolution_ifov = 14.0e-6
            self.resolution_km = 0.5
        elif self["Rad"].resolution == "y: 0.000028 rad x: 0.000028 rad":
            self.resolution_ifov = 28.0e-6
            self.resolution_km = 1.0
        elif self["Rad"].resolution == "y: 0.000056 rad x: 0.000056 rad":
            self.resolution_ifov = 56.0e-6
            self.resolution_km = 2.0

        self.band_id_safe = "C" + str(self["band_id"][...].item()).zfill(2)

        self.midpoint_time = self.epoch2timestamp(seconds=float(self["t"][...].item()))

        self.instrument_coefficients = coefficients.ABICoeff(
            self.platform_ID, self["band_id"][...].item()
        )


class SUVIObject(GOESRObject):
    """Custom SUVI attributes on top of GOES-R SUVI netCDF"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Wavelength for SUVI 304 is masked in netCDF
        self["WAVELNTH"].set_fill_value(0)
        self.wavelength_safe = str(int(self["WAVELNTH"][...].item())).zfill(3)
        if self.wavelength_safe == "000":
            self.wavelength_safe = "304"
            self["WAVELNTH"][...] = 304

        self.instrument_coefficients = coefficients.SUVICoeff(
            self.platform_ID, self["WAVELNTH"][...].item()
        )
