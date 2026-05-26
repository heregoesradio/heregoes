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

"""GOES-R specific conversions and corrections for netCDF data in a walkable interface"""

import datetime
import re

import numpy as np

from heregoes.core import NCInterface
from heregoes.goesr import _coefficients

noaa_time_format = "%Y-%m-%dT%H:%M:%S.%fZ"
cspp_time_format = "%Y-%m-%d %H:%M:%S.%f"


class GOESRData(NCInterface):
    """
    #### Walkable interface for netCDF

    Access netCDF4 variables under `.variables`, and dimensions under `.dimensions`.
    Masked variables are always filled, and scalar variables are always 1D arrays.

    You should never have to invoke `GOESRData` directly, as the appropriate data object is returned by running `heregoes.load()` on a supported netCDF file.

    ##### Load netCDF file
    ```python
    from heregoes import load

    loaded = load("my_goes-r_netcdf.nc")
    ```

    ##### Access variables with an Ellipsis, index, or slice
    ```python
    loaded.variables.MyVariable[...]
    loaded["MyVariable"][...]
    ```

    ##### Override the fill value for a variable
    ```python
    import numpy as np

    loaded.variables.MyVariable.set_fill_value(np.nan)
    ```

    *Note: While the shape of the variable array can change when indexed, the `dimensions` attribute of the variable remains the same.*
    """

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


class _ABIData(GOESRData):
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


class ABIL1bData(_ABIData):
    """Returned by `heregoes.load()` when called on an ABI L1b netCDF file"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._band_id_str = "C" + str(self["band_id"][...].item()).zfill(2)

        self.instrument_coefficients = _coefficients.ABICoeff(
            self.platform_ID, self["band_id"][...].item()
        )


class ABIL2Data(_ABIData):
    """Returned by `heregoes.load()` when called on an ABI L2 netCDF file"""

    pass


class SUVIL1bData(GOESRData):
    """Returned by `heregoes.load()` when called on a SUVI L1b netCDF file"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Wavelength for SUVI 304 is masked in netCDF
        self["WAVELNTH"].set_fill_value(0)
        self._wavelength_str = str(int(self["WAVELNTH"][...].item())).zfill(3)
        if self._wavelength_str == "000":
            self._wavelength_str = "304"
            self["WAVELNTH"][...] = 304

        self.instrument_coefficients = _coefficients.SUVICoeff(
            self["WAVELNTH"][...].item()
        )
