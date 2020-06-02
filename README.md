# Oat_Thesis

Project overview:






Author: Suze Julia Roostee
Supervised by: Nikos Tsardakos Renhuldt, Nick Sirijovski

Software versions:

The project was carried out on Ubuntu 18.04.4 with Linux kernel (4.15.0-76) 
and on 

Commandline environment:
- ImageMagick 6.9.7-4

Fiji

Python packages were installed through conda environment and run in Jupyter Notebook. The fastai/pytorch environment was set up like:

conda install -c pytorch -c fastai fastai
conda install jupyter notebook
conda install -c conda-forge jupyter_contrib_nbextensions 

conda create $fastai2020 
conda activate $fastai2020

The full environemnt can be set up with the environmentfile environment.yml like:
conda env create -f environment.yml

Data preprocessing

Coversion of coloured labels to label values and storing the new labels in NewLabels directory
run as ./create_new_labels_with_fuzz_oat.zsh
and create_new_labels_with_fuzz_oat.zsh

Rotate the image stacks and the labels with the 

TransformJ Turn
TransformJ Rotate


Suggested structure:
Oat/
	Oat_variety/ #various seeds
		Images Labels NewLabels


Data loading

Before moving to the script make sure you have the following file structure for training:

	train/
		Images NewLabels valid.txt

All images and labels required to training should be linked to their respective directories. 
Images used for validation during trainging should be specified in valid.txt 
e.g.
ln -s ~/seed_images/Data_for_ML_Test/Oat/Om2/Images/* .
ls Omat*.tif > valid.txt
cd ..
mv Images/valid.txt .

Training classifier

Predictions

After training directory will look like
mogrify -crop sizexsize+0+0 *.tif


train/
	Images  Labels  models  valid.txt predict_barley.pkl predict_oat.pkl

To run predictions only the predict_barley.pkl and predict_oat.pkl are required
