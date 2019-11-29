#!/usr/bin/env python3

"""
Author: S.J. Roostee

Description:

TODO:

"""

from fastai import *
from fastai.vision import *
from torch import nn
from collections import defaultdict

############	DATA LOADING	############

path = Path('/home/suze/seed_images/Data_for_ML_Test/test_model/Belinda_test')

path_img = path/'Images' #should later be provided through ini file
path_lbl = path/'NewLabels' #should later be provided through ini file

img_names=get_image_files(path_img)
lbl_names=get_image_files(path_lbl)

############	FUNCTIONS		############

def get_mask(img):
    return (path_lbl)/img.name


"""function to filter out images that only contain background; used to calculate appropriate weights; 
function returns the images that also contain seed in a list"""
def filter_background(img_list):
    include=[]
    for img in img_list:
        mask = open_mask(get_mask(img))
        count_total=(torch.unique(mask.data, return_counts=True))
        if not count_total[0].tolist() == [0]:
            include.append(img)
    return include


"""function to filter out images that only contain background; used to filter in the API; 
function has to return a boolean"""
def check_back(img):
    mask = open_mask(get_mask(img))
    count_total=(torch.unique(mask.data, return_counts=True))
    if not count_total[0].tolist() == [0]:
        return True
    else:
        return False

def count_mask(img_list):
    img_list = filter_background(img_list)
    count_classes = defaultdict(int)
    for img in img_list:
        #return the occurence of every class in the mask for an image
        mask = open_mask(get_mask(img))
        count_total = torch.unique(mask.data, return_counts=True)
        classes=count_total[0].tolist()
        count_real=count_total[1].tolist()
        for x, y in zip(classes, count_real):
            count_classes[x] += y
    return count_classes

#from Nikos
def acc_seeds(input, target):
    target = target.squeeze(1)
    mask = target != 0 #not interested in background
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()

############	WEIGHTS AND OTHER VARIABLES	################

classes_count = count_mask(img_names)
##takes long so print progress
print("Weights calculated")
#append occurences in a list 
#seems redundant
counts = []
for c in classes_count:
    counts.append(classes_count[c])

weight_ratios =[min(counts)/x for x in counts]

metrics=acc_seeds
wd=1e-2
lr = 1e-4
lrs = slice(lr/100,lr)

############	BUILD API 	################

np.random.seed(42)
src = (SegmentationItemList.from_folder(path)
       .filter_by_func(check_back)
       .split_by_fname_file('valid.txt')
       .label_from_func(get_mask, classes=list(range(4))))

data = (src.transform(get_transforms(), tfm_y=True, size=128)
       .databunch(bs=8)
       .normalize())

learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)

class_weights=torch.FloatTensor(weight_ratios)
learn.crit = nn.CrossEntropyLoss(weight=class_weights)

print("API ready, start training")

############    TRAINING   ################

n_cycle = 4
print("Training for " +str(n_cycle)+ " epochs.")

learn.fit_one_cycle(4, lr)


