import os
import os.path as osp
import PIL
from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from  datasets.imdb import imdb
import cPickle
import pdb

import sys

sys.path.append('/mnt/py-faster-rcnn/lib/datasets/planorama')

#sys.path.remove('/home/lear/pweinzae/video_src/human')

import planoBase
d = planoBase.Planorama()


class planoTest(imdb):

    def __init__(self, image_set):
        assert image_set in ["train","test","full"]
        imdb.__init__(self, 'planoTest_'+image_set)
        self._image_set = image_set
        self._data_path = d.PATH
        self._classes = ('__background__','1')
        self._class_to_ind = dict(zip(self._classes, xrange(self.num_classes)))
        self._image_index = [d.get_image_file(i) for i in {'full': range(len(d.ILIST)), 'train': d.get_train_indices(), 'test': d.get_test_indices()}[image_set] ]
        assert os.path.exists(self._data_path), \
        'Path does not exist: {}'.format(self._data_path)


    def image_path_at(self, i):
        return self._image_index[i]


    def gt_roidb(self):
	d.show_boxes_html('')
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [ {'boxes': d.get_bbox(os.path.basename(index)), 'gt_classes': np.ones( len(d.get_bbox(os.path.basename(index))),dtype=np.int32), 'gt_overlaps': scipy.sparse.csr_matrix(np.array(np.hstack(([0.0],np.ones(1))),dtype=np.float32)), 'flipped': False} for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)

        print 'wrote gt roidb to {}'.format(cache_file)


        return gt_roidb
