# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
from email.policy import default
import logging
import sys
import os
import glob
from collections import defaultdict

import numpy as np
import torch
import tqdm
from torch.backends import cudnn

import cv2

sys.path.append('.')

from fastreid.evaluation.rank import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer

# import some modules added in project
# for example, add partial reid like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")

logger = logging.getLogger('fastreid.visualize_result')


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--dataset-name",
        help="a test dataset name for visualizing ranking list."
    )
    parser.add_argument(
        "--output",
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--num-vis",
        default=100,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort",
        default="ascending",
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort",
        default="ascending",
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank",
        default=10,
        help="maximum number of rank list to be visualized",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--dataset-path",
        help="path to dataset"
    )
    
    return parser

def get_avg_dist(distmat, num_query, k = 10):
    avg_dist = 0
    for q in range(num_query):
        top_k = np.sort(distmat[q])[:k]
        avg_dist += np.sum(top_k) / 10
    return avg_dist
        
def query_sim(q_feat, g_feat, q_pids, g_pids, q_camids, g_camids, test_loader):
    # compute cosine distance
    distmat = 1 - torch.mm(q_feat, g_feat.t())
    distmat = distmat.numpy()
    
    num_query = distmat.shape[0]
    num_gallery = distmat.shape[1]
    
    avg_dist = get_avg_dist(distmat, num_query)
    print(avg_dist)
    
    # dt = test_loader.dataset
    # for id in q_pids:
    #     print(dt[id])

    # visualizer = Visualizer(test_loader)
    # visualizer.get_model_output([1] * num_query, distmat, q_pids, g_pids, q_camids, g_camids)
    scene, q_dur, q_obj_id, _ = test_loader[0]["img_paths"].split("/")[-4:]
    scene, g_dur, g_obj_id, _ = test_loader[num_query]["img_paths"].split("/")[-4:]
    
    save_path = os.path.join(args.output, scene, f"{q_dur}_{g_dur}", q_obj_id, g_obj_id)
    logger.info("Saving rank list result ...")
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
        
    # query images
    for idx in range(num_query):
        img_path = test_loader[idx]["img_paths"]
        cam_view, frame_number = os.path.basename(img_path).split("_")[2:4]
        q_img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(args.output, scene, f"{q_dur}_{g_dur}", q_obj_id, f"query_{cam_view}_{frame_number}.jpg"), q_img)
        
    # gallery images
    num_sample_gallery = min(5, num_gallery)
    for idx in range(num_sample_gallery):
        img_path = test_loader[idx + num_query]["img_paths"]
        cam_view, frame_number = os.path.basename(img_path).split("_")[2:4]
        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(save_path, f"{cam_view}_{frame_number}.jpg"), img)
        
    # query_indices = visualizer.vis_rank_list(save_path, args.vis_label, args.num_vis,
    #                                          args.rank_sort, args.label_sort, args.max_rank)
    logger.info("Finish saving rank list results!")
    
    return avg_dist

def process_pair_dur(query_dur, gallery_dur, args, cfg):    
    test_loader, num_query = build_reid_test_loader(cfg, dataset_name=args.dataset_name, 
                                                    **{"query_path": query_dur, "gallery_path": gallery_dur})
    
    logger.info("Start extracting image features")
    feats = []
    pids = []
    camids = []
    for (feat, pid, camid) in tqdm.tqdm(EXTRACTOR_MODEL.run_on_loader(test_loader), total=len(test_loader)):
        feats.append(feat)
        pids.extend(pid)
        camids.extend(camid)

    feats = torch.cat(feats, dim=0)
    q_feat = feats[:num_query]
    g_feat = feats[num_query:]
    q_pids = np.asarray(pids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])
    
    query_ids = defaultdict(list)
    gallery_ids = defaultdict(list)
    
    for idx, id in enumerate(q_pids):
        query_ids[id].append(idx)
        
    for idx, id in enumerate(g_pids):
        gallery_ids[id].append(idx)
    
    id_top_sim = defaultdict(list)
    for q_id, q_idx in query_ids.items():
        for g_id, g_idx in gallery_ids.items():
            logger.info(f"Query {q_id} -> Gallery {g_id}")
            cur_q_feat = np.take(q_feat, q_idx, 0) 
            cur_q_pids = np.take(q_pids, q_idx, 0)
            cur_q_camids = np.take(q_camids, q_idx, 0)
            
            cur_g_feat = np.take(g_feat, g_idx, 0)
            cur_g_pids = np.take(g_pids, g_idx, 0)
            cur_g_camids = np.take(g_camids, g_idx, 0)
            
            cur_dataset = []
            for i in q_idx:
                cur_dataset.append(test_loader.dataset[i]) 
            for i in g_idx:
                cur_dataset.append(test_loader.dataset[num_query + i]) 
            
            scene, q_dur, q_obj_id, _ = cur_dataset[0]["img_paths"].split("/")[-4:]
            scene, g_dur, g_obj_id, _ = cur_dataset[len(q_idx)]["img_paths"].split("/")[-4:]
            
            avg_dist = query_sim(cur_q_feat, cur_g_feat, cur_q_pids, 
                      cur_g_pids, cur_q_camids, cur_g_camids,
                      cur_dataset)
            id_top_sim[f"{scene}_{q_dur}_{q_obj_id}"].append((f"{scene}_{g_dur}_{g_obj_id}", avg_dist)) 
    
    for q_path, v in id_top_sim.items():
        scene, dur_q, obj_id_q = q_path.split("_")[-3:]
        _, dur_g, obj_id_g = v[0][0].split("_")[-3:]
        
        rank_id = sorted(v, key=lambda x: x[1])
    
        with open(os.path.join(args.output, scene, f"{dur_q}_{dur_g}", obj_id_q, "rank.txt"), "w") as f:
            for id in rank_id:
                f.write(f"Gallery ID: {id[0]} -> Score: {id[1]} \n")
            
def process_scene(scene_path, args, cfg):
    durs = os.listdir(scene_path)
    durs = sorted(durs, key=lambda x: int(x.split("-")[0]))
    logger.info(f"Duration: {durs}")
    num_dur = len(durs)
    
    for i in range(num_dur - 1):
        logger.info(f"Query dur: {durs[i]}")
        logger.info(f"Gallery dur: {durs[i+1]}")
        
        query_dur = os.path.join(scene_path, durs[i])
        gallery_dur = os.path.join(scene_path, durs[i+1])
        
        process_pair_dur(query_dur, gallery_dur, args, cfg)
            
def run(args, cfg):
    scenes = os.listdir(args.dataset_path)
    for scene in scenes:
        scene_path = os.path.join(args.dataset_path, scene)
        logger.info(f"Start scene: {scene_path}")
        process_scene(scene_path, args, cfg)

if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    
    EXTRACTOR_MODEL = FeatureExtractionDemo(cfg, parallel=args.parallel)
    # bash experiments/find_sim_group_dur.sh logs/vtx_reid/bagtricks_R50-ibn/config.yaml logs/vtx_reid/bagtricks_R50-ibn/model_best.pth logs/vtx_reid/test_vis datasets/group-vtx-newdata/

    run(args, cfg) 
    