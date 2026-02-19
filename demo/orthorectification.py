from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from srtm import get_ellipsoidal_srtm

from heregoes.image import ABIImage
from heregoes.projection import ABIProjection
from heregoes.util import (
    make_8bit,
    minmax,
    nearest_2d_search,
    nearest_scale,
    scale_idx,
    x2,
)
from tests import input_dir
from tests.resources_l1b import abi_cc02_nc

"""Orthorectification of the ABI Fixed Grid with heregoes (netCDF4, Numpy, GDAL, cv2)"""

# in a ~250x250 px region of the Cascade Range in the GOES-East CONUS,
lat_bounds = 47.48457, 44.93645
lon_bounds = -125.14742, -118.879265

# we will track displacement by parallax of these mountain summit coordinates taken from SRTM15
target_latlon = [
    (46.852085, -121.760414),  # Mt. Rainier  4317.157 m
    (46.189583, -122.18958),  # Mt. St. Helens  2396.208 m
    (46.202084, -121.49375),  # Mt. Adams  3650.2834 m
    (45.372917, -121.697914),  # Mt. Hood  3246.899 m
]

# Download GOES-{16...19} ABI C02 L1b netCDF from NOAA CLASS or AWS S3:
# https://noaa-goes16.s3.amazonaws.com/index.html
abi_nc_path = abi_cc02_nc
abi_gamma = 2 / 3

# Download SRTM15 netCDF from:
# https://topex.ucsd.edu/WWW_html/srtm15_plus.html
srtm_nc_path = "/home/wx-star/geo/SRTM/SRTM15_V2.7.nc"


# TODO: add more parallax test cases
# abi_nc_path = input_dir.joinpath("abi-l1b/cases/2021-10-07/OR_ABI-L1b-RadC-M6C02_G17_s20212802106176_e20212802108549_c20212802108568.nc")
# lat_bounds = 38.254807, 36.9999
# lon_bounds = -106.32204, -105.53823
# target_latlon = [(37.59823, -105.955956)]


def demo():
    # get 15-arcsecond SRTM height data for the ABI region
    srtm_lat, srtm_lon, srtm_height = get_ellipsoidal_srtm(
        srtm_nc_path=srtm_nc_path,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
    )
    srtm_lat_bounds = srtm_lat[0, 0], srtm_lat[-1, -1]
    srtm_lon_bounds = srtm_lon[0, 0], srtm_lon[-1, -1]

    # SRTM15 includes bathymetry, so only look down to 0 m
    srtm_height = np.where(srtm_height > 0, srtm_height, 0)

    # form ABIProjection object subsetted on the ABI bounds using SRTM heights to parallax-correct
    abi_projection = ABIProjection(
        abi_nc_path,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        height_m=[srtm_height[0, 0], srtm_height[-1, -1]],
    )

    # warp SRTM height data to the projection of the ABI scene
    warped_heights = abi_projection.resample2abi(
        srtm_height,
        lat_bounds=srtm_lat_bounds,
        lon_bounds=srtm_lon_bounds,
        resample_algo="cubic",
    )

    # make new images using ABI Fixed Grid lat/lon and the warped SRTM heights

    # if resample_nav=False (default), the ABI image is orthorectified to correct for terrain parallax
    img_resampled_to_nav = ABIImage(
        abi_nc_path,
        gamma=abi_gamma,
        lat_bounds=abi_projection.lat_deg,
        lon_bounds=abi_projection.lon_deg,
        height_m=warped_heights,
        resample_nav=False,
    )

    # if resample_nav=True, the underlying navigation is orthorectified and image pixels are untouched
    nav_resampled_to_img = ABIImage(
        abi_nc_path,
        gamma=abi_gamma,
        lat_bounds=abi_projection.lat_deg,
        lon_bounds=abi_projection.lon_deg,
        height_m=warped_heights,
        resample_nav=True,
    )

    # render the uncorrected ABI image for comparison
    original_abi_img = ABIImage(
        abi_nc_path,
        gamma=abi_gamma,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
    )

    # save images with our target coordinates indicated in green
    save_path = Path("img")
    save_path.mkdir(exist_ok=True)
    cv2.imwrite(
        save_path.joinpath("original.png"),
        latlon_target(
            img=original_abi_img.bv,
            search_lat=original_abi_img.lat_deg,
            search_lon=original_abi_img.lon_deg,
            target_latlon=target_latlon,
        ),
    )

    cv2.imwrite(
        save_path.joinpath("resampled-image.png"),
        latlon_target(
            img=img_resampled_to_nav.bv,
            search_lat=img_resampled_to_nav.lat_deg,
            search_lon=img_resampled_to_nav.lon_deg,
            target_latlon=target_latlon,
        ),
    )
    cv2.imwrite(
        save_path.joinpath("resampled-nav.png"),
        latlon_target(
            img=nav_resampled_to_img.bv,
            search_lat=nav_resampled_to_img.lat_deg,
            search_lon=nav_resampled_to_img.lon_deg,
            target_latlon=target_latlon,
        ),
    )

    index_delta_figure(
        nav_resampled_to_img.index,
        nav_resampled_to_img.nav_index,
        filepath=save_path.joinpath(f"index-delta.png"),
    )

    # zero the inverse-orthorectified nav index,
    y, x = nav_resampled_to_img.nav_index
    norm_y = y - y.min()
    norm_x = x - x.min()
    norm_inverse_ortho_idx = np.clip(norm_y, 0, norm_y.shape[0] - 1), np.clip(
        norm_x, 0, norm_x.shape[1] - 1
    )

    # and use it for visualization
    warped_heights_img = make_8bit(minmax(warped_heights) * 255)
    warped_orthorectified_heights = make_8bit(
        minmax(warped_heights[norm_inverse_ortho_idx]) * 255
    )
    cv2.imwrite(
        save_path.joinpath("warped-heights.png"),
        x2(warped_heights_img),
        # latlon_target(img=warped_heights_img, search_lat=img_resampled_to_nav.lat_deg, search_lon=img_resampled_to_nav.lon_deg, target_latlon=target_latlon),
    )
    cv2.imwrite(
        save_path.joinpath("warped-inverse-orthorectified-heights.png"),
        x2(warped_orthorectified_heights),
        # latlon_target(img=warped_orthorectified_heights, search_lat=nav_resampled_to_img.lat_deg, search_lon=nav_resampled_to_img.lon_deg, target_latlon=target_latlon),
    )


def latlon_target(
    img: NDArray,
    search_lat: NDArray,
    search_lon: NDArray,
    target_latlon: tuple[float, float],
    color_bgr: tuple[int, int, int] = (0, 255, 0),
) -> NDArray:
    upscale_factor = 2
    upscaled_img = nearest_scale(img, upscale_factor)

    marked_img = np.stack((upscaled_img,) * 3, axis=-1).astype(np.uint8)
    for target_lat, target_lon in target_latlon:
        target_idx = nearest_2d_search(
            y_arr=search_lat,
            x_arr=search_lon,
            target_y=np.atleast_1d(target_lat),
            target_x=np.atleast_1d(target_lon),
        )

        target_idx = scale_idx(target_idx, upscale_factor)
        target_y, target_x = target_idx

        marked_img[target_y : target_y + 2, target_x : target_x + 2, :] = color_bgr
        # cv2.circle(marked_img, center=target_idx[::-1], radius=3, thickness=1, color=color_bgr)
        cv2.circle(
            marked_img, center=target_idx[::-1], radius=25, thickness=1, color=color_bgr
        )

    return marked_img


def slice2idx(slc):
    if not (
        isinstance(slc, tuple)
        and len(slc) == 2
        and isinstance(slc[0], slice)
        and isinstance(slc[1], slice)
    ):
        return slc

    y_slice, x_slice = slc

    y1, y2 = y_slice.start, y_slice.stop
    x1, x2 = x_slice.start, x_slice.stop

    y_indices = np.arange(y1, y2)
    x_indices = np.arange(x1, x2)

    xx, yy = np.meshgrid(x_indices, y_indices)

    return yy, xx


def index_delta_figure(idx1, idx2, filepath):
    import plotly.express as px

    # euclidean
    index_delta = np.linalg.norm(
        np.asarray(slice2idx(idx1)) - np.asarray(slice2idx(idx2)), axis=0
    )

    # chebyshev
    # index_delta = np.fmax(np.abs(idx1[0] - idx2[0]), np.abs(idx1[1] - idx2[1]))

    fig = px.imshow(
        index_delta,
        height=500,
        labels={"color": "Nav error (px)"},
        aspect="equal",
        template="plotly_dark",
    )

    fig.update_layout(margin=dict(t=0, b=0, l=0))
    fig.update_xaxes(
        showticklabels=False,
        ticks="",
    )
    # fig.update_yaxes(
    #     showticklabels=False,
    #     ticks="",
    # )
    pad = 5
    max = index_delta.shape[0]
    mid = max // 2
    fig.update_yaxes(
        tickvals=(0 + pad, mid, max - pad),
        ticktext=("0", str(mid), str(max)),
    )

    fig.write_image(filepath)


if __name__ == "__main__":
    demo()
