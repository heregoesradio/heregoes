#!/bin/bash

label_font=Helvetica
label_pointsize=15
label_position=southeast
label_offset=+20+15

gif_fps=0.5
gif_quality=100

#run demo and handle output
python orthorectification.py || exit 1
cd img
convert original.png -gravity $label_position -font $label_font -pointsize $label_pointsize -fill white -annotate $label_offset 'Uncorrected' original.png

convert resampled-nav.png -gravity $label_position -font $label_font -pointsize $label_pointsize -fill white -annotate $label_offset 'Nav corrected' resampled-nav.png
convert resampled-image.png -gravity $label_position -font $label_font -pointsize $label_pointsize -fill white -annotate $label_offset 'Image corrected' resampled-image.png

gifski --fps $gif_fps --quality $gif_quality --output resampled-image.gif original.png resampled-image.png
gifski --fps $gif_fps --quality $gif_quality --output resampled-nav.gif original.png resampled-nav.png


convert warped-inverse-orthorectified-heights.png -gravity $label_position -font $label_font -pointsize $label_pointsize -fill white -annotate $label_offset 'Inverse orthorectified' warped-inverse-orthorectified-heights.png
convert warped-heights.png -gravity $label_position -font $label_font -pointsize $label_pointsize -fill white -annotate $label_offset 'Projected SRTM' warped-heights.png

gifski --fps $gif_fps --quality $gif_quality --output warped-srtm.gif warped-heights.png warped-inverse-orthorectified-heights.png
