from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.autograd import Variable
from loda_kitti import Kitti_Dataset
import os
import torch.multiprocessing
from tqdm import tqdm
from models.superpoint import SuperPoint
from models.superglue import SuperGlue
import time

seed_value = 42 
torch.manual_seed(seed_value)


parser = argparse.ArgumentParser(
    description='DepthImage and RGBImage matching ',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--epoch', type=int, default=60, help='Number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate')

parser.add_argument(
    '--nms_radius', type=int, default=4,
    help='SuperPoint Non Maximum Suppression (NMS) radius'
' (Must be positive)')
parser.add_argument(
    '--keypoint_threshold', type=float, default=0.005,
    help='SuperPoint keypoint detector confidence threshold')
parser.add_argument(
    '--max_keypoints', type=int, default=1024,
    help='Maximum number of keypoints detected by Superpoint'
            ' (\'-1\' keeps all keypoints)')
parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='SuperGlue match threshold')

parser.add_argument("--dataset_root", type=str, default='./dataset')
parser.add_argument("--name", type=str, default='3dmatch')
parser.add_argument("--image_size", type=int, nargs=2, default=[640,480], help='Width and height of input image')
parser.add_argument("--camera_intrinsics", type=float,nargs=9, default=[[585,0,320],[0,585,240],[0,0,1]])
parser.add_argument("--depth_scale", type=float, default=1000.0, help='Scale factor for depth values')

parser.add_argument("--checkpoint_dir", type=str, default="ck_Finall_F/")