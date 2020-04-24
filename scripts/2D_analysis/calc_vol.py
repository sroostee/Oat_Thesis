from fastai.vision import image.py
from collections import defaultdict

#have the masks in a separate directory

def count_mask(masks_list):
    #img_list = filter_background(img_list)
    count_classes = defaultdict(int)
    for m in mask_list:
        #return the occurence of every class in the mask for an image
        mask = open_mask(mask)
        count_total = torch.unique(mask.data, return_counts=True)
        classes=count_total[0].tolist()
        count_real=count_total[1].tolist()
        for x, y in zip(classes, count_real):
            count_classes[x] += y
    return count_classes