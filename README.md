# Oat_Thesis

Project overview



Author: Suze Julia Roostee
Supervised by: Nikos Tsardakos Renhuldt, Nick Sirijovski

Software versions:

The project was carried out on Ubuntu 18.04.4 with Linux kernel (4.15.0-76) 
and on 

Commandline environment:
- ImageMagick 6.9.7-4

Python packages were installed through conda environment and run in Jupyter Notebook

conda install -c pytorch -c fastai fastai
conda install jupyter notebook
conda install -c conda-forge jupyter_contrib_nbextensions 

conda create $envname 
conda activate $envname

Data preprocessing

Coversion of coloured labels to greyscale levels and storing the new labels in NewLabels directory
run as ./create_new_labels_with_fuzz.zsh
and create_new_labels_with_fuzz_oat.zsh

Suggested structure:
Barley/
	Barley_variety/ #various seeds
		Images Labels NewLabels
Oat/
	Oat_variety/ #various seeds
		Images Labels NewLabels


Data loading

Before moving to the script make sure you have the following file structure for training:

	train/
		Images NewLabels valid.txt

All images and labels required to training should be linked to their respective directories. 
Images used for validation during trainging should be specified in valid.txt 




Training classifier

Predictions

After training directory will look like

train/
	Images  Labels  models  valid.txt predict_barley.pkl predict_oat.pkl

To run predictions only the predict_barley.pkl and predict_oat.pkl are required
