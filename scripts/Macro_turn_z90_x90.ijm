title=getTitle();
run("Image Sequence...", "open=/home/suze/Documents/Thesis/seed_images/Data_for_ML_Test/BM1/Images/ + title file=BM1_ sort");
run("TransformJ Turn", "z-angle=90 y-angle=0 x-angle=90");
run("Image Sequence... ", "format=TIFF name=BM1_zx_ save=/home/suze/Documents/Thesis/seed_images/Data_for_ML_Test/BM1/Images_rotated/BM1_zx_0000.tif");
