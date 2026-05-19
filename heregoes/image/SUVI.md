## SUVI Examples

### SUVI Radiance Imagery
Check the exposure time of a SUVI L1b netCDF:
```python
from heregoes import load

suvi_data = load("OR_SUVI-L1b-[...].nc")

#print the commanded exposure time variable (seconds)
print(suvi_data.variables.CMD_EXP[...])
```

Render the long-exposure (1-second) SUVI radiance image; shorter exposures for flares are not officially supported and will cause a warning:
```python
from heregoes.image import SUVIImage

img = SUVIImage("OR_SUVI-L1b-[...].nc")
img.save("suvi.png")
```

<a href="https://static.heregoesradio.com/suvi/g16_suvi_171_2024-10-03T122712Z.jpg">
<img src="https://static.heregoesradio.com/suvi/g16_suvi_171_2024-10-03T122712Z_thumbnail.jpg">
</a>

<br>

### SUVI Custom RGB
Create red, green, and blue SUVIImage channels with custom scaling coefficients, then combine as an RGB image:

```python
from heregoes.image import SUVIImage, SUVIRGB
from heregoes.util import max_time_delta

#SUVI RGB recipe for Here GOES Radiotelescope
red = SUVIImage(
    "OR_SUVI-L1b-Fe171[...].nc",
    input_range=(0.0, 15.0),
    asinh_a=0.0007,
    output_range=(0.0, 1.0),
)

green = SUVIImage(
    "OR_SUVI-L1b-Fe195[...].nc",
    input_range=(0.0, 50.0),
    asinh_a=0.001,
    output_range=(0.0, 1.0),
)

blue = SUVIImage(
    "OR_SUVI-L1b-Fe284[...].nc",
    input_range=(0.0, 60.0),
    asinh_a=0.000075,
    output_range=(0.0, 1.0),
)

#check the max time delta between input images (blurring occurs beyond ~20 minutes)
print(max_time_delta([red.time, green.time, blue.time]))

rgb = SUVIRGB(red, green, blue)

rgb.save(filepath="output_dir", ext=".jpeg")

#PosixPath('output_dir/g16_suvi_color_2024-10-03T122712Z.jpeg')
```

<br>

<a href="https://static.heregoesradio.com/suvi/g16_suvi_color_2024-10-03T122712Z.jpeg">
<img src="https://static.heregoesradio.com/suvi/g16_suvi_color_2024-10-03T122712Z_thumbnail.jpeg">
</a>
