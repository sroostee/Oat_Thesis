#!/usr/bin/env python3

"""
Author: S.J. Roostee

Description:

TODO:

"""

from fastai.vision import *	#image.py
from collections import defaultdict

import argparse
import sys

usage = "This program calculates the volume of every class present based on the masks"
usage += " that are segmented through prediction or manual annotation."

parser = argparse.ArgumentParser(description=usage)

parser.add_argument(
	"-v", "-version",
	action="version",
	version="%(prog) 1.0")

parser.add_argument(
	"-p",
	dest="path_to_masks",
	required=True,
	help = "Sets the path to the masks directory."
	)

parser.add_argument(
	"-vox",
	dest="vox", 
	type=float,
	required=True, 
	help="Sets the voxel size in cubic micrometres."
	)

parser.add_argument(
    '-o',
    dest='outfile', 
    metavar = 'OUTFILE',
    type=argparse.FileType('w'),
    required = False, 
    default = sys.stdout, 
    help = "Defines the output file. If not provided print to sys.stdout"
    )

args = parser.parse_args()

path_to_masks = Path(args.path_to_masks)
# mask_list = path_to_masks.ls()
# print(mask_list)
out = args.outfile

def count_mask(masks_list):
    #img_list = filter_background(img_list)
    count_classes = defaultdict(int)
    for m in masks_list.ls():
        #return the occurence of every class in the mask for an image
        mask = open_mask(m)
        count_total = torch.unique(mask.data, return_counts=True)
        classes=count_total[0].tolist()
        count_real=count_total[1].tolist()
        for x, y in zip(classes, count_real):
            count_classes[x] += y
    return count_classes

counts = count_mask(path_to_masks)

print('#volumes in cubic micrometres', file=out)

#volumes considered as square blocks
#interpolation in z-direction could optimise volume estimate further

for mask,vol in counts.items():
    volume = vol * (args.vox**3)
    print('{}\t{}'.format(mask, volume), file = out) #, file = chr_f)


