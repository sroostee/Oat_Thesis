from fastai import *
from fastai.vision import *
from torch import nn

path = Path('/home/suze/seed_images/Data_for_ML_Test/test_model/train')
path.ls()

path_img = path/'Images' #should later be provided through ini file
path_lbl = path/'NewLabels' #should later be provided through ini file

def get_mask(img):
    return (path_lbl)/img.name

img_names=get_image_files(path/'Images')
lbl_names=get_image_files(path/'NewLabels')

mask = open_mask(get_mask(img_names[5]))
mask.show()

img = open_image(img_names[1])
img.show()
src_size = np.array(mask.shape[1:])
src_size,mask.data

np.random.seed(42)
src = (SegmentationItemList.from_folder(path)
       #.split_subsets(train_size=0.2, valid_size=0.1)
       .split_by_rand_pct()
       .label_from_func(get_mask, classes=list(range(6))))

data = (src.transform(get_transforms(), tfm_y=True, size=128)
       .databunch(bs=8)
       .normalize())

#from Nikos

def acc_seeds(input, target):
    target = target.squeeze(1)
    mask = target != 0 #not interested in background
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()

metrics=acc_seeds
wd=1e-2

learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)

weights = [0.1, 0.7, 1, 1 ]
class_weights=torch.FloatTensor(weights)
learn.crit = nn.CrossEntropyLoss(weight=class_weights)

lr = 1e-4

lrs = slice(lr/100,lr)

learn.fit_one_cycle(4, lr)
learn.save('stage-small_1')
learn.load('stage-small_1');
learn.unfreeze()
learn.fit_one_cycle(2, lrs)
learn.save('stage-small_2')
data_med = (src.transform(get_transforms(), tfm_y=True, size=256)
       .databunch(bs=8)
       .normalize())

       learn = unet_learner(data_med, models.resnet34, metrics=metrics, wd=wd)
learn.crit = nn.CrossEntropyLoss(weight=class_weights)

learn.load('stage-small_2');
learn.unfreeze()
learn.fit_one_cycle(2, lrs)