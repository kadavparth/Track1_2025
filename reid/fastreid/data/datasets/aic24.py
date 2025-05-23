# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class AIC24(ImageDataset):
    """AIC24.

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    # dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    # dataset_name = "market1501"
    dataset_name = "aic24"

    def __init__(self, root='/workspace/reid_dataset', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        # self.root = root
        # self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.dataset_dir = '/workspace/reid_dataset'

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'reid_dataset')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"reid_dataset".')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        # self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        # self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        # if self.market1501_500k:
        #     required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir)
        query = lambda: self.process_dir(self.query_dir, is_train=False)
        gallery = lambda: self.process_dir(self.gallery_dir, is_train=False)
        # gallery = lambda: self.process_dir(self.gallery_dir, is_train=False) + \
        #                   (self.process_dir(self.extra_gallery_dir, is_train=False) if self.market1501_500k else [])

        super(AIC24, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')
        # pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # print('@', pid, camid, img_path)
            # raise
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 4000  # pid == 0 means background
            assert 1 <= camid <= 2000
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
