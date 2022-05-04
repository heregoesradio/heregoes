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

"""Classes for working with ancillary datasets in the ABI fixed grid projection"""

import io

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib

matplotlib.use("Agg")
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np

from heregoes import IREMIS_DIR, SCRIPT_PATH, logger, projection, util


class AncillaryDataset:
    """This is the base class for ancillary datasets, and can be used directly to save and load compressed npz NumPy objects"""

    def __init__(self):
        self.data = {}
        self.dataset_name = None

    def save(self, save_dir=SCRIPT_PATH.joinpath("ancillary")):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        file_name = "_".join(
            (
                self.abi_meta.platform_ID,
                self.abi_meta.instrument_meta.scene_id_safe,
                self.dataset_name,
            )
        ).lower()
        file_path = save_dir.joinpath(file_name).with_suffix(".npz")

        try:
            np.savez_compressed(file_path, **self.data)

        except:
            logger.critical("Could not save dataset at %s", file_path)
            print(traceback.format_exc())

            return

    def load(self, npz_path):
        npz_path = Path(npz_path)
        try:
            npz = np.load(npz_path)

        except:
            logger.critical("Could not load dataset at %s", npz_path)
            print(traceback.format_exc())

            return

        self.dataset_name = npz_path.name
        for key in npz.keys():
            self.data[key] = npz[key]


class IREMIS(AncillaryDataset):
    """
    Loads in UW CIMSS' Baseline Fit Infrared Emissivity Database a.k.a IREMIS (Download and info: https://cimss.ssec.wisc.edu/iremis/)
    and projects to the ABI fixed grid. Requres IREMIS netCDFs to be present in a directory set by the `iremis_dir` argument or
    by the HEREGOES_ENV_IREMIS_DIR environmental variable.

    Provides a linear interpolation of land surface emissivity for:
        - ABI channel 7 (3.9 μm): `data['c07_land_emissivity']`
        - ABI channel 14 (11.2 μm): `data['c14_land_emissivity']`

    Arguments:
        - `abi_meta`: The NCMeta object formed on a GOES-R ABI L1b Radiance netCDF file
        - `iremis_dir`: Location of IREMIS netCDF files. Defaults to the directory set by the HEREGOES_ENV_IREMIS_DIR environmental variable
    """

    def __init__(self, abi_meta, iremis_dir=IREMIS_DIR):
        super(IREMIS, self).__init__()

        self.abi_meta = abi_meta
        month = self.abi_meta.time_coverage_start.month
        self.dataset_name = "iremis_abi_month" + str(month).zfill(2)

        try:
            iremis_dir = Path(iremis_dir)

        except:
            logger.critical("iremis_dir is not set or is not a real path")
            print(traceback.format_exc())

            return

        iremis_locations = iremis_dir.joinpath("global_emis_inf10_location.nc")
        iremis_months = [
            "global_emis_inf10_monthFilled_MYD11C3.A2016001.041.nc",
            "global_emis_inf10_monthFilled_MYD11C3.A2016032.041.nc",
            "global_emis_inf10_monthFilled_MYD11C3.A2016061.041.nc",
            "global_emis_inf10_monthFilled_MYD11C3.A2016092.041.nc",
            "global_emis_inf10_monthFilled_MYD11C3.A2016122.041.nc",
            "global_emis_inf10_monthFilled_MYD11C3.A2016153.041.nc",
            "global_emis_inf10_monthFilled_MYD11C3.A2016183.041.nc",
            "global_emis_inf10_monthFilled_MYD11C3.A2016214.041.nc",
            "global_emis_inf10_monthFilled_MYD11C3.A2016245.041.nc",
            "global_emis_inf10_monthFilled_MYD11C3.A2016275.041.nc",
            "global_emis_inf10_monthFilled_MYD11C3.A2016306.041.nc",
            "global_emis_inf10_monthFilled_MYD11C3.A2016336.041.nc",
        ]
        self.iremis_nc = iremis_dir.joinpath(iremis_months[month - 1])

        if not self.iremis_nc.exists():
            logger.critical("Could not locate IREMIS netCDF: %s", self.iremis_nc)
            print(traceback.format_exc())

            return

        with netCDF4.Dataset(self.iremis_nc, "r") as loaded_iremis_nc:
            # UW Baseline Fit IREMIS may be linearly interpolated for moderate resolution spectral emissivity: https://doi.org/10.1175/2007JAMC1590.1
            self.data["c07_land_emissivity"] = util.linear_interp(
                3.7,
                4.3,
                loaded_iremis_nc["emis1"][:],
                loaded_iremis_nc["emis2"][:],
                3.9,
            ).astype(np.float32)
            self.data["c14_land_emissivity"] = util.linear_interp(
                10.8,
                12.1,
                loaded_iremis_nc["emis8"][:],
                loaded_iremis_nc["emis9"][:],
                11.2,
            ).astype(np.float32)

        # ocean pixels have a negative fill value, we set them to have an emissivity of 1.0
        self.data["c07_land_emissivity"][self.data["c07_land_emissivity"] < 0.0] = 1.0
        self.data["c14_land_emissivity"][self.data["c14_land_emissivity"] < 0.0] = 1.0

        # rotate IREMIS to be N-S and E-W
        self.data["c07_land_emissivity"] = np.flipud(
            np.rot90(self.data["c07_land_emissivity"], k=1)
        )
        self.data["c14_land_emissivity"] = np.flipud(
            np.rot90(self.data["c14_land_emissivity"], k=1)
        )

        with netCDF4.Dataset(iremis_locations, "r") as iremis_locations_nc:
            iremis_ul_lat = iremis_locations_nc["lat"][0, 0]
            iremis_ul_lon = iremis_locations_nc["lon"][0, 0]
            iremis_lr_lat = iremis_locations_nc["lat"][-1, -1]
            iremis_lr_lon = iremis_locations_nc["lon"][-1, -1]

        abi_projection = projection.ABIProjection(self.abi_meta)
        self.data["c07_land_emissivity"] = abi_projection.resample2abi(
            self.data["c07_land_emissivity"],
            latlon_bounds=[iremis_ul_lon, iremis_ul_lat, iremis_lr_lon, iremis_lr_lat],
            interpolation="bilinear",
        )
        self.data["c14_land_emissivity"] = abi_projection.resample2abi(
            self.data["c14_land_emissivity"],
            latlon_bounds=[iremis_ul_lon, iremis_ul_lat, iremis_lr_lon, iremis_lr_lat],
            interpolation="bilinear",
        )


class WaterMask(AncillaryDataset):
    """
    Loads in Global Self-consistent, Hierarchical, High-resolution Shorelines (GSHHS a.k.a GSHHG) (http://www.soest.hawaii.edu/pwessel/gshhg/)
    and projects to the ABI fixed grid. Natural Earth rivers (https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-rivers-lake-centerlines/)
    are optionally added on top of GSHHS when `rivers` is `True`. Both the GSHHS and Natural Earth datasets are automatically downloaded by Cartopy.

    Provides a boolean land/water mask in `data['water_mask']` where water is `False` and land is `True`.

    Arguments:
        - `abi_meta`: The NCMeta object formed on a GOES-R ABI L1b Radiance netCDF file
        - `gshhs_scale`: 'auto', 'coarse', 'low', 'intermediate', 'high, or 'full' (https://scitools.org.uk/cartopy/docs/latest/reference/generated/cartopy.feature.GSHHSFeature.html)
        - `rivers`: Default `False`
    """

    def __init__(self, abi_meta, gshhs_scale="intermediate", rivers=False):
        super(WaterMask, self).__init__()

        self.abi_meta = abi_meta
        self.dataset_name = "gshhs_abi_" + gshhs_scale

        # https://scitools.org.uk/cartopy/docs/latest/crs/index.html#cartopy.crs.Globe
        goes_globe = ccrs.Globe(
            datum=None,
            ellipse="GRS80",
            semimajor_axis=self.abi_meta.instrument_meta.semi_major_axis,
            semiminor_axis=self.abi_meta.instrument_meta.semi_minor_axis,
            flattening=None,
            inverse_flattening=self.abi_meta.instrument_meta.inverse_flattening,
            towgs84=None,
            nadgrids=None,
        )
        # https://scitools.org.uk/cartopy/docs/latest/crs/projections.html#goes
        goes_projection = ccrs.Geostationary(
            central_longitude=self.abi_meta.instrument_meta.longitude_of_projection_origin,
            satellite_height=self.abi_meta.instrument_meta.perspective_point_height,
            false_easting=0,
            false_northing=0,
            globe=goes_globe,
            sweep_axis=self.abi_meta.instrument_meta.sweep_angle_axis,
        )

        dpi = 1000
        plt.figure(
            figsize=(
                self.abi_meta.instrument_meta.x / dpi,
                self.abi_meta.instrument_meta.y / dpi,
            ),
            dpi=dpi,
        )
        ax = plt.axes(projection=goes_projection)

        # cartopy errors on extents the size of the ABI Full Disk
        if self.abi_meta.instrument_meta.scene_id != "Full Disk":
            ul_x = (
                self.abi_meta.instrument_meta.x_image_bounds[0]
                * self.abi_meta.instrument_meta.perspective_point_height
            )
            ul_y = (
                self.abi_meta.instrument_meta.y_image_bounds[0]
                * self.abi_meta.instrument_meta.perspective_point_height
            )
            lr_x = (
                self.abi_meta.instrument_meta.x_image_bounds[1]
                * self.abi_meta.instrument_meta.perspective_point_height
            )
            lr_y = (
                self.abi_meta.instrument_meta.y_image_bounds[1]
                * self.abi_meta.instrument_meta.perspective_point_height
            )
            ax.set_extent([ul_x, lr_x, ul_y, lr_y], crs=goes_projection)

        # https://scitools.org.uk/cartopy/docs/v0.14/matplotlib/feature_interface.html#cartopy.feature.GSHHSFeature
        gshhs_coastline = cf.GSHHSFeature(
            scale=gshhs_scale,
            levels=[1],
            edgecolor="black",
            facecolor="black",
            zorder=-1,
        )
        gshhs_lakes = cf.GSHHSFeature(
            scale=gshhs_scale,
            levels=[2],
            edgecolor="white",
            facecolor="white",
            zorder=-1,
        )
        gshhs_lake_islands = cf.GSHHSFeature(
            scale=gshhs_scale,
            levels=[3],
            edgecolor="black",
            facecolor="black",
            zorder=-1,
        )
        gshhs_island_ponds = cf.GSHHSFeature(
            scale=gshhs_scale,
            levels=[4],
            edgecolor="white",
            facecolor="white",
            zorder=-1,
        )

        linewidth = 1 / dpi
        ax.add_feature(gshhs_coastline, linewidth=linewidth)
        ax.add_feature(gshhs_lakes, linewidth=linewidth)
        ax.add_feature(gshhs_lake_islands, linewidth=linewidth)
        ax.add_feature(gshhs_island_ponds, linewidth=linewidth)

        if rivers:
            self.dataset_name += "_rivers"
            ne_rivers = cf.NaturalEarthFeature(
                "physical",
                "rivers_lake_centerlines",
                "10m",
                edgecolor="white",
                facecolor="none",
                zorder=-1,
            )
            ax.add_feature(ne_rivers, linewidth=linewidth)

        matplotlib.rc("axes", edgecolor="white")
        matplotlib.rc("lines", antialiased=False)
        matplotlib.rc("patch", antialiased=False)

        # remove margin space: https://stackoverflow.com/a/27227718
        ax.set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        io_buf = io.BytesIO()
        plt.savefig(io_buf, format="raw", dpi=dpi)
        plt.close()
        io_buf.seek(0)
        self.data["water_mask"] = np.reshape(
            np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
            newshape=(
                self.abi_meta.instrument_meta.y,
                self.abi_meta.instrument_meta.x,
                -1,
            ),
        )
        io_buf.close()
        # invert the mask such that water is False and land is True
        self.data["water_mask"] = ~(
            (self.data["water_mask"][:, :, 0] / 255).astype(dtype=bool)
        )
