#!/usr/bin/env python3

#from fastai import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
from fastai.vision import *
from torch.utils.data import SequentialSampler
from fastai3D import mysampler
from fastai3D import loader 

#seed = 42
#np.random.seed(seed)

path = Path('/home/suze/Documents/Thesis/seed_images/Data_for_ML_Test/train')
path.ls()

path_img = path/'Images' #should later be provided through ini file
path_lbl = path/'Labels' #should later be provided through ini file

#img_names=get_image_files(path_img)
#lbl_names=get_image_files(path_lbl)

classes = range(4)

def get_mask(img):
	return (path_lbl)/img.name

src = (SegmentationItemList.from_folder(path)
	   .split_subsets(train_size=0.2, valid_size=0.1)
	   #.split_by_rand_pct()
	   .label_from_func(get_mask, classes=list(classes)))

data = (src.transform(get_transforms(), tfm_y=True, size=128)
	   .databunch(bs=1)
	   .normalize())

print(torch.utils.data.DataLoader)

data.train_dl = data.train_dl.new(shuffle=False, drop_last=False, 
	sampler=None, batch_sampler=mysampler.OrderedBatchSampler(SequentialSampler(data.train_dl), 3, False))

print(data.train_dl.batch_size)
