import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer, crs

SCRIPT_PATH = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_PATH.parent.resolve()))

from heregoes.core import NCInterface


def egm96_to_grs80(lat, lon, height_above_geoid):
    # converts heights above the EGM96 geoid to heights above the GRS80/WGS84 ellipsoid

    wgs84_egm96 = crs.CompoundCRS(
        name="WGS 84 + EGM96 height", components=["EPSG:4326", "EPSG:5773"]
    )
    grs80 = crs.CRS("+proj=lonlat +ellps=GRS80 +units=m +vunits=m +no_defs")
    # needs egm96_15.gtx installed in $ENV/share/proj/egm96_15.gtx. equivalent to the pipeline string:
    # "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=vgridshift +grids=egm96_15.gtx +multiplier=1 +step +proj=unitconvert +xy_in=rad +xy_out=deg"
    transformer = Transformer.from_crs(crs_from=wgs84_egm96, crs_to=grs80)

    return transformer.transform(lat, lon, height_above_geoid)[2]


def get_ellipsoidal_srtm(srtm_nc_path, lat_bounds, lon_bounds):
    srtm = NCInterface(srtm_nc_path)

    # load in SRTM coordinates,
    srtm_lat = srtm.variables.lat[...]
    srtm_lon = srtm.variables.lon[...]

    # and find the bounding indices of the ABI scene
    srtm_y1, srtm_y2 = np.searchsorted(srtm_lat, lat_bounds)
    srtm_x1, srtm_x2 = np.searchsorted(srtm_lon, lon_bounds)
    srtm_slice = np.s_[srtm_y2:srtm_y1, srtm_x1:srtm_x2]

    srtm_mesh_lon, srtm_mesh_lat = np.meshgrid(
        srtm_lon[srtm_slice[1]],
        srtm_lat[srtm_slice[0]][::-1],
    )

    srtm_height_egm96 = srtm.variables.z[srtm_slice][::-1]
    srtm_height_grs80 = egm96_to_grs80(srtm_mesh_lat, srtm_mesh_lon, srtm_height_egm96)

    return srtm_mesh_lat, srtm_mesh_lon, srtm_height_grs80
