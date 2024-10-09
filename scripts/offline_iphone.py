import numpy as np
import os
import json
import cv2
import argparse
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

# exp_directory = '/ghome/l6/yqliang/littleduck/SplaTAM/experiments/iPhone_Captures'
# scene_name = "240926122854"
# directory = os.path.join(exp_directory, scene_name) + '/'

def readDepthImage(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.float32)
    arr = arr.reshape(transformsJson['depth_map_height'], transformsJson['depth_map_width'])
    return arr  # 1 unit = 1 meter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_name", default=None, type=str, help="Scene name.")
    parser.add_argument("--config", default="./configs/iphone/nerfcapture_off.py", type=str, help="Path to config file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load config
    experiment = SourceFileLoader(
        os.path.basename(args.config), args.config
    ).load_module()

    print(experiment.config)
    print(experiment)
    scene_name = args.scene_name.split('.')[0]

    directory = os.path.join(experiment.base_dir, scene_name)
    directory = Path(directory)
    with open(directory / 'transforms.json') as f:
        transformsJson = json.load(f)

    os.mkdir(directory / 'rgb')
    os.mkdir(directory / 'depth')
    paths = os.listdir(directory / 'images')
    rgb_w = 0
    rgb_h = 0
    for path in paths:
        if path.endswith('.png'):
            os.rename(directory / 'images' / path, directory / 'rgb' / path)
            rgb = cv2.imread(directory / 'rgb' / path)
            rgb_w = rgb.shape[1]
            rgb_h = rgb.shape[0]
    for path in paths:
        if path.endswith('.depth'):
            depthMap = readDepthImage(directory / 'images' / path)
            save_depth = (depthMap*65535/float(10)).astype(np.uint16)
            save_depth = cv2.resize(save_depth, dsize=(
                    rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(directory / 'depth' / path.replace('.depth', '.png'), save_depth)


    for frame in transformsJson['frames']:
        frame['file_path'] = frame['file_path'].replace('images', 'rgb')
        frame['file_path'] += '.png'
        frame['depth_path'] = frame['depth_path'].replace('.depth.png', '.png')
        frame['depth_path'] = frame['depth_path'].replace('images', 'depth')

    with open(directory / 'transforms.json', 'w') as f:
        json.dump(transformsJson, f)