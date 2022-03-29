# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import random
import warnings
import os
from collections import defaultdict

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VTXReidGroup(ImageDataset):
    dataset_dir = 'group-vtx-newdata'
    dataset_name = 'group-vtx-newdata'
    
    def __init__(self, root='datasets', market1501_500k=False, **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.query_dir = kwargs.get('query_path')
        self.gallery_dir = kwargs.get('gallery_path')
        
        self.id_mapper = {}
        self.camid_mapper = {}
        self.id_counter = 0
        self.camid_counter = 0
        self.query_samples = 5
        self.gallery_samples = 100

        required_files = [
            self.query_dir,
            self.gallery_dir,
        ]

        self.check_before_run(required_files)

        query = lambda: self.process_dir(self.query_dir, is_query=True)
        gallery = lambda: self.process_dir(self.gallery_dir, is_query=False)

        super(VTXReidGroup, self).__init__([], query, gallery, **kwargs)

    def process_dir(self, dir_path, is_query=True):
        ids = os.listdir(dir_path)
        id_samples = {}
        for id in ids:
            id_path = os.path.join(dir_path, id)
            imgs = glob.glob(osp.join(id_path, '*.jpg'))
            if(is_query):
                random.seed(26)
                query_k = min(len(imgs), self.query_samples)
                id_samples[id] = random.sample(imgs, query_k)
                # print(id_samples[id])
            else:
                random.seed(26)
                gallery_k = min(len(imgs), self.gallery_samples)
                id_samples[id] = random.sample(imgs, gallery_k)

        data = []
        for id, v in id_samples.items():
            for img_path in v:
                img_name = osp.basename(img_path).split(".")[0]
                scene, duration, cam_view, frameId, obj_id = img_name.split("_")
    
                pid_str = f"{scene}_{duration}_{obj_id}"
                if(pid_str not in self.id_mapper):
                    self.id_mapper[pid_str] = self.id_counter
                    self.id_counter += 1
                pid = self.id_mapper[pid_str]
                
                camid_str = f"{scene}_{duration}_{cam_view}"
                if(camid_str not in self.camid_mapper):
                    self.camid_mapper[camid_str] = self.camid_counter
                    self.camid_counter += 1
                camid = self.camid_mapper[camid_str]
                    
                data.append((img_path, pid, camid))

        return data
