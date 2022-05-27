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
class VTXReid(ImageDataset):
    dataset_dir = 'vtx-reid'
    dataset_name = 'vtx-reid'
    
    def __init__(self, root='datasets', market1501_500k=False, **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'test', 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test', 'gallery')
        
        self.id_mapper = {}
        self.camid_mapper = {}
        self.id_counter = 0
        self.camid_counter = 0

        required_files = [
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]

        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir)
        query = lambda: self.process_dir(self.query_dir, is_train=False)
        gallery = lambda: self.process_dir(self.gallery_dir, is_train=False)

        super(VTXReid, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        data = []
        for img_path in img_paths:
            img_name = osp.basename(img_path).split(".")[0]
            scene, duration, cam_view, frameId, obj_id = img_name.split("_")
            if(is_train):
                pid = f"{scene}_{obj_id}"
                camid = f"{scene}_{cam_view}"
            else:
                pid_str = f"{scene}_{obj_id}"
                if(pid_str not in self.id_mapper):
                    self.id_mapper[pid_str] = self.id_counter
                    self.id_counter += 1
                pid = self.id_mapper[pid_str]
                
                camid_str = f"{scene}_{cam_view}"
                if(camid_str not in self.camid_mapper):
                    self.camid_mapper[camid_str] = self.camid_counter
                    self.camid_counter += 1
                camid = self.camid_mapper[camid_str]
                
            data.append((img_path, pid, camid))

        return data
