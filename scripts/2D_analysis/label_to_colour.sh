cp -r LabelsPredicted ColouredLabels
cd ColouredLabels

mogrify -fill "#b3cccc" -opaque "#000000" *.tif #background
mogrify -fill "#00ff00" -opaque "#010101" *.tif #endosperm (green)
mogrify -fill "#ccbb00" -opaque "#020202" *.tif #aleurone layer (yellow)
mogrify -fill "#0015ff" -opaque "#030303" *.tif #germ (blue)