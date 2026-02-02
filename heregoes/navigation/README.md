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
from heregoes.navigation import ABINavigation

#2d index:
index = (y, x)

#or continuous slice:
index = (slice(y1, y2, None), slice(x1, x2, None))
index = np.s_[y1:y2, x1:x2]

nav = ABINavigation("OR_ABI-L1b-RadC-M6C07[...].nc", index=index)
```

#### Or a geodetic Earth point or bounding box:
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
Navigated geodetic coordinates can be displaced for image features above the GRS80 ellipsoid, such as high terrain or cloud. The parallax displacement vector is described on the sphere by[^4] [^5]:

```
(h * H * tan(θ)) / (H - h)
```

where,

```
H = satellite ellipsoidal height
h = ellipsoidal height of cloud or high terrain
θ = satellite zenith angle
```

Subsetted navigation elements are corrected for parallax to the nearest Fixed Grid pixel if `height_m` is provided as an argument to `ABINavigation`. Typically, the height is either estimated for cloud pixels within the scene[^6] or taken from a DEM for the terrain.

- #### For cloud height with the `index` argument:
    The `height_m` argument is valid for the indexed Fixed Grid point(s). The calculated latitude / longitude and derived navigation elements are corrected for parallax by adding ellipsoidal height to the `r_c` term of equations 7.1.2.8.2 in [^1].

- #### For terrain height with `lat_bounds` and `lon_bounds`:
    The `height_m` argument is valid for the bounding geodetic point(s). All navigation elements *except* for latitude / longitude are then shifted toward `lat_bounds` and `lon_bounds` to the nearest Fixed Grid pixel.


### References
[^1]: https://www.goes-r.gov/users/docs/PUG-GRB-vol4.pdf
[^2]: https://doi.org/10.5281/zenodo.6078954
[^3]: https://doi.org/10.1017/CBO9781139029346.005
[^4]: https://doi.org/10.1109/LGRS.2013.2283573
[^5]: http://nwafiles.nwas.org/jom/articles/2023/2023-JOM2/2023-JOM2.pdf
[^6]: https://www.star.nesdis.noaa.gov/goesr/documents/ATBDs/Baseline/ATBD_GOES-R_Cloud_Height_v3.0_Jul2012.pdf