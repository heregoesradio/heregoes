## Terrain correction of the ABI Fixed Grid using heregoes

Here we orthorectify (parallax-correct) an ABI scene in its native Fixed Grid projection using the `heregoes` library (`netCDF4`, `Numpy`, `GDAL`, `cv2`). See the [navigation README](../heregoes/navigation/README.md) for more background on navigation and parallax correction with `heregoes`.


### Motivation
At off-nadir viewing angles, the apparent positions of Earth features like cloud or high terrain are displaced by parallax effects in satellite imagery. Cloud height can be estimated from radiance and used to derive the true geodetic coordinates of cloud pixels[^1]. Orthorectification is the inverse problem, where terrain height and coordinates are known, but the pixel locations are not[^2].

In 2022, Pestana and Lundquist[^3] demonstrated that rapid orthorectification of ABI imagery can be achieved in Python by interpolating ABI pixels onto a digital elevation model (DEM). However, as the authors note, ABI L1b radiance is already resampled from detector counts through ground system processing[^4]. Further resampling by orthorectification complicates sub-pixel retrievals, particularly for high-intensity targets such as fires[^5]. If we instead orthorectify the underlying coordinate system, the ABI Fixed Grid[^6], then we can accurately navigate elevated pixels *in situ* and preserve the original sampling of the L1b image (Figure 1).


### Demo
We consider a region of the GOES-East CONUS covering the Pacific Northwest. Summit locations of four mountains in the Cascade Range are targeted to track displacement by parallax: Mt. Rainier, Mt. St. Helens, Mt. Adams, and Mt. Hood.

![Before and after parallax-corrected navigation](./img/resampled-nav.gif "Before and after parallax-corrected navigation")
* **Figure 1.** *Mountaintop coordinates (circled in green) are navigated in the ABI image. When the navigation is parallax-corrected, the coordinates align with the displaced summit pixels in the original image.*

<br>

By simply reversing the direction of parallax correction, `heregoes` can also orthorectify the ABI image itself (Figure 2) while maintaining the same geostationary projection.

![Before and after orthorectifying the image](./img/resampled-image.gif "Before and after orthorectifying the image")
* **Figure 2.** *Mountaintop coordinates (circled in green) are navigated in the GOES-East ABI image. After orthorectification of the image, the summit pixels align with the displaced coordinates.*

<br>

### Implementation
For this demo, a 15 arcsecond (~500 m) rendition of SRTM[^7] was used to complement the nominal ~500 m resolution of ABI channel 2 (0.64 µm) imagery taken from GOES-16. SRTM orthometric heights were converted to heights above the GRS80 ellipsoid with `pyproj` and warped to the ABI scene with `heregoes` using `GDAL`'s cubic spline interpolation. The orthorectification was performed using navigation and nearest neighbor methods built into `heregoes`; see [orthorectification.py](orthorectification.py) and [`ABINavigation`](../heregoes/navigation/_navigation.py).


### References
[^1]: https://www.star.nesdis.noaa.gov/goesr/documents/ATBDs/Baseline/ATBD_GOES-R_Cloud_Height_v3.0_Jul2012.pdf
[^2]: https://doi.org/10.3390/rs15092403
[^3]: https://doi.org/10.1016/j.rse.2022.113221
[^4]: https://doi.org/10.3390/rs10020177
[^5]: https://doi.org/10.1016/B978-0-12-814327-8.00013-5
[^6]: https://www.goes-r.gov/users/docs/PUG-GRB-vol4.pdf
[^7]: https://doi.org/10.1029/2019EA000658
