import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation

class KITTIRangeImageDataset(Dataset):
    def __init__(self, data_root, sequence_range, img_size=(376, 1250)):
        self.data_root = data_root
        self.sequence_range = sequence_range
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        self.data = self.load_data()
        
        # 激光雷达参数 (KITTI Velodyne HDL-64E)
        self.vertical_fov = (-24.9, 2.0)  # 垂直视场角 (度)
        self.horizontal_res = 0.08         # 水平分辨率 (度/像素)
        self.vertical_res = 0.4            # 垂直分辨率 (度/像素)
        
    def load_data(self):
        data = []
        for seq in self.sequence_range:
            seq_dir = os.path.join(self.data_root, f"sequences/{seq:02d}")
            image_dir = os.path.join(seq_dir, "image_2")
            velodyne_dir = os.path.join(seq_dir, "velodyne")
            calib_dir = os.path.join(seq_dir, "calib.txt")
            
            # 加载标定参数
            calib = self.load_calibration(calib_dir)
            
            image_files = sorted(os.listdir(image_dir))
            pointcloud_files = sorted(os.listdir(velodyne_dir))
            
            for img_file, pc_file in zip(image_files, pointcloud_files):
                img_path = os.path.join(image_dir, img_file)
                pc_path = os.path.join(velodyne_dir, pc_file)
                data.append((img_path, pc_path, calib))
        return data
    
    def load_calibration(self, calib_path):
        calib = {}
        with open(calib_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    calib[key.strip()] = np.array([float(x) for x in value.strip().split()])
        return calib
    
    def cartesian_to_spherical(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)  # 水平角
        theta = np.arcsin(z / r)  # 俯仰角
        return r, phi, theta
    
    def create_range_image(self, points):
        # 转换为球坐标
        r, phi, theta = self.cartesian_to_spherical(points)
        
        # 计算图像尺寸
        height = int((self.vertical_fov[1] - self.vertical_fov[0]) / self.vertical_res)
        width = int(360 / self.horizontal_res)
        
        # 创建空的距离图像
        range_image = np.zeros((height, width), dtype=np.float32)
        
        # 将点投影到距离图像
        for i in range(len(points)):
            # 计算像素坐标
            v = int((np.degrees(theta[i]) - self.vertical_fov[0]) / self.vertical_res)
            h = int((np.degrees(phi[i]) + 180) / self.horizontal_res)
            
            # 确保坐标在图像范围内
            if 0 <= v < height and 0 <= h < width:
                # 保留最近的点
                if range_image[v, h] == 0 or r[i] < range_image[v, h]:
                    range_image[v, h] = r[i]
        
        # 归一化距离值
        max_range = np.max(range_image) if np.max(range_image) > 0 else 1
        range_image = range_image / max_range
        
        # 转换为三通道图像
        range_image = np.stack([range_image]*3, axis=-1)
        return Image.fromarray((range_image * 255).astype(np.uint8))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, pc_path, calib = self.data[idx]
        
        # 加载相机图像
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        # 加载点云并创建距离图像
        pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        range_img = self.create_range_image(pc[:, :3])
        range_img = self.transform(range_img.convert("L"))
        
        # 加载标定参数
        P2 = calib['P2'].reshape(3, 4)
        Tr_velo_to_cam = calib['Tr_velo_cam'].reshape(3, 4) if 'Tr_velo_cam' in calib else np.eye(3, 4)
        
        return {
            'image': img,
            'range_image': range_img,
            'calib': {
                'P2': P2,
                'Tr_velo_to_cam': Tr_velo_to_cam
            }
        }