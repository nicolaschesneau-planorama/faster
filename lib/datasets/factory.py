# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
import datasets.planoDeploy
import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))


for baseId in ['5511acb455c467fd66faab4e_5550e6cc673ddd5b3b6e34d4','55681a9de1d50db239964b88_55681ae9e1d50db239964c72']:
    for split in ["train","test","full"]:
        name='planoDeploy_{}_{}'.format(baseId,split)
        __sets[name] = (lambda split=split, baseId=baseId: datasets.planoDeploy.planoDeploy(split,baseId))

def get_imdb(name,path=None,baseId=None):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
