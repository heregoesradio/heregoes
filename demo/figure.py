import cv2
import numpy as np
from numpy.typing import NDArray

from heregoes.util import (
    nearest_2d_search,
    nearest_scale,
    scale_idx,
)


def latlon_target(
    img: NDArray,
    search_lat: NDArray,
    search_lon: NDArray,
    target_latlon: tuple[float, float],
    color_bgr: tuple[int, int, int] = (0, 255, 0),
    upscale_factor: float = 2,
) -> NDArray:
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

        marked_img[
            target_y : target_y + upscale_factor,
            target_x : target_x + upscale_factor,
            :,
        ] = color_bgr

        # cv2.circle(marked_img, center=target_idx[::-1], radius=3, thickness=1, color=color_bgr)
        cv2.circle(
            marked_img,
            center=(target_x + upscale_factor // 2, target_y + upscale_factor // 2),
            radius=25,
            thickness=1,
            color=color_bgr,
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
