for dir in Om2 Omat Beloat_OB_forML Om_1_7_test OB6_test; do
    cd $dir
    cp -r Labels NewLabels
    chmod -R +w NewLabels
    cd ..
done

cd Om2/NewLabels
mogrify -fuzz 05% -fill "#000000" -opaque "#b3cccc" *.tif #background
mogrify -fuzz 05% -fill "#010101" -opaque "#00ffff" *.tif #transfer bundle (cyan) -> endosperm
mogrify -fuzz 05% -fill "#010101" -opaque "#00ff00" *.tif #endosperm (green)
mogrify -fuzz 05% -fill "#020202" -opaque "#f3d918" *.tif #aleurone layer (yellow)
mogrify -fuzz 05% -fill "#030303" -opaque "#3b3bf8" *.tif #germ (blue)

cd ../../Omat/NewLabels
mogrify -fuzz 05% -fill "#000000" -opaque "#b3cccc" *.tif #background
mogrify -fuzz 05% -fill "#010101" -opaque "#00ffff" *.tif #transfer bundle (cyan) -> endosperm
mogrify -fuzz 05% -fill "#010101" -opaque "#00ff00" *.tif #endosperm (green)
mogrify -fuzz 05% -fill "#020202" -opaque "#f3d918" *.tif #aleurone layer (yellow)
mogrify -fuzz 05% -fill "#030303" -opaque "#1500ff" *.tif #germ (blue)

cd ../../Beloat_OB_forML/NewLabels
mogrify -fuzz 05% -fill "#000000" -opaque "#b3cccc" *.tif #background
mogrify -fuzz 05% -fill "#000000" -opaque "#a300cc" *.tif #hair -> background
mogrify -fuzz 05% -fill "#010101" -opaque "#00eaff" *.tif #transfer bundle (cyan) -> endosperm
mogrify -fuzz 05% -fill "#010101" -opaque "#00d62a" *.tif #endosperm (green)
mogrify -fuzz 05% -fill "#020202" -opaque "#ffff00" *.tif #aleurone layer (yellow)
mogrify -fuzz 05% -fill "#030303" -opaque "#3b3bf8" *.tif #germ (blue)

cd ../../Om_1_7_test/NewLabels
mogrify -fuzz 05% -fill "#000000" -opaque "#b3cccc" *.tif #background
mogrify -fuzz 05% -fill "#010101" -opaque "#00ff00" *.tif #endosperm (green)
mogrify -fuzz 05% -fill "#020202" -opaque "#c4cc00" *.tif #aleurone layer (dark yellow)
mogrify -fuzz 05% -fill "#030303" -opaque "#000aff" *.tif #germ (blue)

cd ../../OB6_test/NewLabels
mogrify -fuzz 05% -fill "#000000" -opaque "#b3cccc" *.tif #background
mogrify -fuzz 05% -fill "#010101" -opaque "#00ff00" *.tif #endosperm (green)
mogrify -fuzz 05% -fill "#020202" -opaque "#ccbb00" *.tif #aleurone layer (yellow)
mogrify -fuzz 05% -fill "#030303" -opaque "#0015ff" *.tif #germ (blue)
