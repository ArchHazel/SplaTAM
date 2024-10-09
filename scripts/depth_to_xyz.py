from pathlib import Path
import torch
import numpy as np
import math
import cv2
import os
import argparse
from importlib.machinery import SourceFileLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/iphone/nerfcapture_off.py", type=str, help="Path to config file.")
    return parser.parse_args()

def precompute_view_matrices(view):
    device = view.world_view_transform.device
    view.world_view_transform_inv = torch.inverse(view.world_view_transform)
    fx = 1 / (2 * math.tan(view.FoVx / 2.))
    fy = 1 / (2 * math.tan(view.FoVy / 2.))
    view.intrins = torch.tensor(
        [[fx * view.image_width, 0., 1/2. * view.image_width],
        [0., fy * view.image_height, 1/2. * view.image_height],
        [0., 0., 1.0]],
        device=device
    ).float().T
    view.intrins_inv = torch.inverse(view.intrins)
    view.gt_gray_img = view.original_image.mean(0).unsqueeze(0)

def depth_to_points_fast(depth, intrins_inv):
    H, W = depth.shape[1:]  # depth: 1 x H x W
    grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3).float().cuda() # N x 3
    rays_d = points @ intrins_inv  # N x 3 = N x 3 * 3 x 3
    points = depth.reshape(-1, 1) * rays_d  # N x 3 = N x 1 point_wise_* N x 3
    return points, rays_d


if __name__ == "__main__":
    args = parse_args()
    # Load config
    experiment = SourceFileLoader(
        os.path.basename(args.config), args.config
    ).load_module()
    experiment
    pts_xyz, rays_d = depth_to_points_fast(render_pkg[depth_name], view.intrins_inv)
    pts_xyz = pts_xyz.view(-1, 3)
    pts_color = view.original_image.permute(1, 2, 0).reshape(-1, 3)
    pts_xyz_view = torch.concat([pts_xyz, torch.ones_like(pts_xyz[:, 0:1])], dim=-1)
    pts_xyz_world = pts_xyz_view @ view.world_view_transform_inv
    pts_xyz_world = pts_xyz_world[:, :3]