# Oat_Thesis

Project overview:

In this thesis annotated oat seed images (2D stack forming 3D seed)  created with x-ray micro CT were used to create a classifier that can annotate new oat seed images.
Five oat seeds were available for training, validation, and testing. Image stacks were rotated to create additional views. 
The software used for this project and the project steps are described below, along with the files that belong to each section.


Author: Suze Julia Roostee

Supervised by: Nikos Tsardakas Renhuldt, Nick Sirijovski

---------

## Software versions:

The project was carried out on Ubuntu 18.04.4 with Linux kernel (4.15.0-76) 
and on 

Commandline environment:
- ImageMagick 6.9.7-4

(Fiji is just) ImageJ version 2.0.0

Python packages were installed through conda environment and run in Jupyter Notebook. The fastai/pytorch environment was set up like:
```
conda install -c pytorch -c fastai fastai
conda install jupyter notebook
conda install -c conda-forge jupyter_contrib_nbextensions 
```
```
conda create $fastai2020 
conda activate $fastai2020
```
The full environemnt can be set up with the environmentfile environment.yml like:

```
conda env create -f environment.yml
```

---------

## Data preprocessing

Coversion of coloured labels to label values and storing the new labels in NewLabels directory
run as 
```
./create_new_labels_with_fuzz_oat.zsh
```
Rotate the image stacks and the labels with ImageJ 

for 90° degree turn use:
TransformJ Turn 

for the 70°, 20°, 10° use: 
TransformJ Rotate

Suggested structure:
```
Oat/
	Oat_variety/ #various seeds
		Images Labels NewLabels Images_rotated NewLabels_rotated
```
---------

## Data loading

Before moving to the script make sure you have the following file structure for training:
```
	train/
		Images NewLabels valid.txt
```
All images and labels required for training should be linked or copied to their respective directories. 
Labels should have the same file name as their corresponding image.
Images used for validation during trainging should be specified in valid.txt 
e.g.
```
cd Images
ln -s ~/seed_images/Data_for_ML_Test/Oat/Omat/Images/* .
ls Omat*.tif > valid.txt
cd ..
mv Images/valid.txt .
```
---------

## Training and evaluation of classifier

The classifier is trained and evaluated in Jupyter Notebook with the file 2D_segmentation.ipynb or 3D_segmentation.ipynb.
Directories and some parameters can be changed using 2D_train.ini or 3D_train.ini.
The classifier exported by this file is stored in a .pkl file such as 2D_oat.pkl and can also be used for label predictions of entire seeds.

Evaluation of the model is done is the same files 2D_segmentation.ipynb or 3D_segmentation.ipynb and follows after training. 2D_test.ini or 3D_test.inin can be used to specify model and test directory.

---------

## Predictions

In the fastai v.1 library predictions on new images can only be carried out on images with even size width and height (e.g. 630x650 is fine, but 630x649 is not). 
In that case resize the image before predictions with
```
mogrify -crop widthxheight+0+0 *.tif
```
where width and height and the new (even numbers) width and height.

Then the predictions can be done using the 2D_predict.ipynb script. The classifier should be in the same directory as this script.
The new unlabelled images should be in a subdirectory. The prediction script will automatically create a directory to store the new labels in. 
Classifier name and directory names can be specified in pred_2D.ini . 
Suggested structure: 

```
pred/
	NewImages  PredLabels  2D_oat.pkl
```

Optionally labels can be converted back to colours with label_to_colour.sh
However volume calculations require the original label values.

---------

## Volume calculations

volume calculations on labels can be carried out with the calc_vol.py script. 
use like 
```
./calc_vol.py -p pathtolabels -vox voxelsize -out outputfile
```
