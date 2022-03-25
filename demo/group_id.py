import argparse
from collections import defaultdict
import os 
import glob
import shutil

def construct_group(key, members, save_path):
    scene, duration, obj_id = key.split("_")
    save_path = os.path.join(save_path, scene, duration, obj_id)
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
    
    for member in members:
        shutil.move(member, save_path)
    
# scene, duration, cam_view, frameId, obj_id = image.split(".")[0].split("_")
def run(args):
    if(not os.path.exists(args.save_path)):
        os.mkdir(args.save_path)
    
    image_paths = glob.glob(os.path.join(args.data_path, "*.jpg"))
    id_group = defaultdict(list)
    for path in image_paths:
        scene, duration, cam_view, frameId, obj_id = path.split(".")[0].split("_")
        id_key = f"{scene}_{duration}_{obj_id}"
        id_group[id_key].append(path)
    
    for k, v in id_group.items():
        construct_group(k, v, args.save_path)

        
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="path to dataset folder")
    parser.add_argument('--save_path', type=str, help="path to dataset folder")
    args = parser.parse_args()
    
    run(args)
    