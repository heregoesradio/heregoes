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

"""GOES-R specific conversions and corrections for netCDF data on top of an NCInterface"""

import datetime
import re

import numpy as np

from heregoes.core import NCInterface
from heregoes.goesr import coefficients

noaa_time_format = "%Y-%m-%dT%H:%M:%S.%fZ"
cspp_time_format = "%Y-%m-%d %H:%M:%S.%f"


class GOESRData(NCInterface):
    """Custom generic attributes on top of GOES-R netCDF"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._platform_ID_str = "-".join(("GOES", self.platform_ID[-2:]))
        self.product_name = "-".join(self.dataset_name.split("_")[1].split("-")[0:3])
        self._instrument_type_str = self.product_name.split("-")[0]
        self.time_coverage_start = self._norm_date(
            self.time_coverage_start, noaa_time_format
        )
        self.time_coverage_end = self._norm_date(
            self.time_coverage_end, noaa_time_format
        )
        self.date_created = self._norm_date(self.date_created, noaa_time_format)

        # this is a nonstandard field added by CSPP GEO. If we can't get it, use the actual file mtime
        try:
            self.local_file_time = self._norm_date(
                self.cspp_geo_grb_reconstruction_end_time, cspp_time_format
            )

        except:
            self.local_file_time = datetime.datetime.fromtimestamp(
                self._nc_file.stat().st_mtime, datetime.UTC
            )

    def _norm_date(self, datestring, fmtstring):
        return datetime.datetime.strptime(datestring, fmtstring).replace(
            tzinfo=datetime.timezone.utc
        )

    def epoch2timestamp(self, seconds):
        epoch = datetime.datetime(2000, 1, 1, 12, 0, 0, 0, tzinfo=datetime.timezone.utc)
        timestamp = epoch + datetime.timedelta(seconds=seconds)

        return timestamp

    def timestamp2epoch(self, timestamp):
        epoch = datetime.datetime(2000, 1, 1, 12, 0, 0, 0, tzinfo=datetime.timezone.utc)
        seconds = (timestamp - epoch).total_seconds()

        return seconds


class ABIData(GOESRData):
    """Custom ABI attributes on top of GOES-R ABI netCDF"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "Meso" in self.scene_id:
            mesoscale_id_search = re.search(
                "[A-Za-z]+([0-9])", self.dataset_name.split("-")[2]
            )

            if bool(mesoscale_id_search):
                self.mesoscale_id = mesoscale_id_search.group(1)

            if not (bool(mesoscale_id_search)) or self.mesoscale_id not in ["1", "2"]:
                raise ValueError(
                    f"Failed to determine mesoscale sector in {self._nc_file}."
                )

            self.scene_id = "Mesoscale " + self.mesoscale_id
            self._scene_id_str = "Meso" + self.mesoscale_id

        elif self.scene_id == "Full Disk":
            self._scene_id_str = "FullDisk"

        elif self.scene_id == "CONUS":
            self._scene_id_str = "CONUS"

        self.midpoint_time = self.epoch2timestamp(seconds=float(self["t"][...].item()))

        # https://www.goes-r.gov/users/docs/PUG-GRB-vol4.pdf Table 7.1.2.6, Table 7.1.2.7-1
        self.resolution_ifov = (
            self.variables.x.scale_factor
        )  # horizontal spatial resolution in radians
        resolution_ifov_1km = np.array(28.0e-6, dtype=np.float32)
        self.resolution_km = self.resolution_ifov / resolution_ifov_1km


class ABIL1bData(ABIData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._band_id_str = "C" + str(self["band_id"][...].item()).zfill(2)

        self.instrument_coefficients = coefficients.ABICoeff(
            self.platform_ID, self["band_id"][...].item()
        )


class ABIL2Data(ABIData):
    pass


class SUVIL1bData(GOESRData):
    """Custom SUVI attributes on top of GOES-R SUVI netCDF"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Wavelength for SUVI 304 is masked in netCDF
        self["WAVELNTH"].set_fill_value(0)
        self._wavelength_str = str(int(self["WAVELNTH"][...].item())).zfill(3)
        if self._wavelength_str == "000":
            self._wavelength_str = "304"
            self["WAVELNTH"][...] = 304

        self.instrument_coefficients = coefficients.SUVICoeff(
            self["WAVELNTH"][...].item()
        )
