## Navigation and indexing with parallax correction on the ABI Fixed Grid


```python
from heregoes.navigation import ABINavigation

nav = ABINavigation("OR_ABI-L2-FDCC-M6[...].nc")
```


### Navigation
ABI L1b and L2+ products are delivered with Fixed Grid coordinates (y, x) corresponding to N-S and E-W instrument scan angles.
Starting from these coordinates, we expose the following navigation elements as attributes of the `ABINavigation` class:
- Geodetic latitude and longitude of Earth pixels (`lat_deg`, `lon_deg`)[^1]
- Local zenith and azimuth angles for Sun (`sun_za`, `sun_az`) and satellite (`sat_za`, `sat_az`) look vectors[^2]
- Along-track and cross-track distance (`along_track_m`, `cross_track_m`), and effective ground area `area_m2`[^3]


### Indexing
Initialize `ABINavigation` on a subset of the ABI scene.

#### With a Fixed Grid index or slice:
```python
import numpy as np

from heregoes.navigation import ABINavigation

#2d index:
index = (y, x)

#or continuous slice:
index = (slice(y1, y2, None), slice(x1, x2, None))
index = np.s_[y1:y2, x1:x2]

nav = ABINavigation("OR_ABI-L1b-RadC-M6C07[...].nc", index=index)
```

#### Or with geodetic Earth coordinates:
```python
from heregoes.navigation import ABINavigation

#lat/lon point:
lat_bounds = point_latitude
lon_bounds = point_longitude

#or lat/lon bounding box:
lat_bounds = [upper_left_lat, lower_right_lat]
lon_bounds = [upper_left_lon, lower_right_lon]

nav = ABINavigation("OR_ABI-L1b-RadC-M6C07[...].nc", lat_bounds=lat_bounds, lon_bounds=lon_bounds)
```


### Parallax correction
Navigated geodetic coordinates can be displaced for image features above the GRS80 ellipsoid, such as high terrain or cloud. `ABINavigation` corrects for this parallax effect when given the ellipsoidal height of the feature in `height_m`. Cloud height is typically estimated from brightness temperatures within the ABI scene[^5], whereas terrain height can be given as individual scalars or a matrix taken from a DEM.

- #### For cloud height:
    ```python
    ABINavigation(
        "OR_ABI[...].nc",
        index=index_or_slice_containing_cloud,
        height_m=cloud_height_meters,
    )
    ```
    Ellipsoidal height `height_m` is valid for the indexed Fixed Grid point(s); if no `index` argument is provided, then `height_m` is considered for all pixels in the ABI scene.

- #### For terrain height with `lat_bounds` and `lon_bounds`:
    ```python
    ABINavigation(
        "OR_ABI[...].nc",
        lat_bounds=terrain_latitudes,
        lon_bounds=terrain_longitudes,
        height_m=terrain_height_meters,
    )
    ```
    Ellipsoidal height `height_m` is valid for all Earth points provided in `lat_bounds` and `lon_bounds`.

See the [terrain correction demo](../../demo/README.md) and [orthorectification.py](../../demo/orthorectification.py) for more advanced usage of the parallax correction feature.

### References
[^1]: https://www.goes-r.gov/users/docs/PUG-GRB-vol4.pdf
[^2]: https://doi.org/10.5281/zenodo.6078954
[^3]: https://doi.org/10.1017/CBO9781139029346.005
[^4]: https://doi.org/10.1109/LGRS.2013.2283573
[^5]: https://www.star.nesdis.noaa.gov/goesr/documents/ATBDs/Baseline/ATBD_GOES-R_Cloud_Height_v3.0_Jul2012.pdf