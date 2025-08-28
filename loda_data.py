import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from models import superpoint
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import cv2

class image_depth_Dataset(Dataset):
    """Dataset for loading image and depth data.
    Args:
        data_root (str): Root directory of the dataset.
        name (str): Name of the dataset.
        image_size (tuple): Target image size (width, height).
        camera_intrinsics (np.array): 3x3 camera intrinsic matrix.
        depth_scale (float): Scale factor for depth values (default: 1000.0).
    """
    def __init__(self, data_root, name,image_size,camera_intrinsics = [[585,0,320],[0,585,240],[0,0,1]], depth_scale=1000.0):
        self.data_root = data_root
        self.dataset_name = name
        self.dataset_path = os.path.join(data_root, name)
        self.dataset_classes = os.listdir(self.dataset_path)
        self.image_size = image_size
        self.camera_intrinsics = camera_intrinsics
        self.depth_scale = depth_scale
        self.data = self.load_data()
        self.superpoint = superpoint.SuperPoint()
    

    def __len__(self):
        """Returns the number of total data in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Returns the image and depth data at the specified index."""
        pose_file, image_file, depth_file = self.data[idx]
        camera_to_world_pose_matrix = self.load_pose(pose_file)
        world_to_camera_pose_matrix = torch.inverse(camera_to_world_pose_matrix)
        image = self.load_image(image_file)
        depth = self.load_depth(depth_file)
        normailzed_depth = self.normalized_depth(depth)

        pred_rgb = self.superpoint({'image':image[None]})
        pred_depth = self.superpoint({'image':normailzed_depth[None]})

        kpts_rgb = pred_rgb['keypoints'][0]
        kpts_depth = pred_depth['keypoints'][0]
        desc_rgb = pred_rgb['descriptors'][0]
        desc_depth = pred_depth['descriptors'][0]

        kpts_depth_pointclouds = self.depth_image_to_point_cloud(depth,kpts_depth, self.camera_intrinsics, camera_to_world_pose_matrix)
        _, xy_points = self.tr3d2d(kpts_depth_pointclouds, self.camera_intrinsics, world_to_camera_pose_matrix)


        return {
            'world_to_camera_pose_matrix': world_to_camera_pose_matrix,              # 4x4位姿矩阵
            'image': image,                   # 预处理后的RGB图像 [3, H, W]
            'depth': normailzed_depth,        # 预处理后的深度图 [1, H, W]
            'keypoints_rgb': kpts_rgb,        # RGB图像特征点 [M, 2]
            'keypoints_depth': kpts_depth,    # 深度图像特征点 [N, 2]
            'xy_points': xy_points,           # 点云投影到图像平面的点 [N, 2]
            'desc_image': desc_rgb,           # RGB图像描述子 [D, M]
            'desc_depth': desc_depth,         # 深度图像描述子 [D, N]
            'image_path': image_file,         # 图像路径
            'depth_path': depth_file          # 深度图路径
        }
        
    def load_data(self):
        """Loads the image and depth data from the dataset."""
        data = []
        for cls in self.dataset_classes:
            cls_path = os.path.join(self.dataset_path, cls, 'query')
            if os.path.isdir(cls_path):
                for file in os.listdir(cls_path):
                    if '_pose.txt' in file:
                        pose_file = os.path.join(cls_path, file)
                        # 查找对应的图像和深度文件
                        base_name = file.replace('_pose.txt', '')
                        image_file = os.path.join(cls_path, f"{base_name}_color.png")
                        depth_file = os.path.join(cls_path, f"{base_name}_depth.png")
                        
                        if os.path.exists(image_file) and os.path.exists(depth_file):
                            data.append((pose_file, image_file, depth_file))
        return data
    
    def load_pose(self, pose_file):
        """加载并解析位姿文件为4x4变换矩阵  camera-to-world"""
        with open(pose_file, 'r') as f:
            # 假设文件包含7个数值: tx, ty, tz, qx, qy, qz, qw
            data = np.loadtxt(f)
            
        translation = data[:3]
        quaternion = data[3:]  # 顺序: qx, qy, qz, qw
        
        # 创建旋转矩阵
        rotation = R.from_quat(quaternion)
        rotation_matrix = rotation.as_matrix()
        
        # 构建4x4变换矩阵
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = translation
        
        return torch.from_numpy(pose_matrix).float()

    def load_image(self, image_file):
        """加载并预处理RGB图像"""
        image = Image.open(image_file).convert('RGB')
        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np)  # 仍是 [0,255], 形状 (H,W,3)
        image_tensor = image_tensor.permute(2, 0, 1)  # 转为 (C,H,W) 格式
        return image_tensor.float() / 255.0  # 归一化到 [0,1]
 
    def load_depth(self, depth_file):
        """加载并预处理16位深度图(单位：毫米)"""
        # 使用PIL打开深度图并保留原始位深
        depth_img = Image.open(depth_file)
        
        # 检查并处理16位深度数据[1,7](@ref)
        if depth_img.mode == 'I;16':
            # 方法1：转换为32位整型容器再转为uint16（避免自动降为8位）
            depth_img = depth_img.convert('I')
            depth_array = np.array(depth_img).astype(np.uint16)
        else:
            # 方法2：直接读取二进制数据（更可靠）[7](@ref)
            byte_data = depth_img.tobytes()
            dtype = np.dtype('uint16').newbyteorder('>' if depth_img.mode == 'I;16' else '<')
            depth_array = np.frombuffer(byte_data, dtype=dtype)
            depth_array = depth_array.reshape(depth_img.size[1], depth_img.size[0])

        # 转换为PyTorch张量
        depth_tensor = torch.from_numpy(depth_array.astype(np.float32))
    
        # 添加通道维度并转换单位（毫米→米）
        depth_tensor = depth_tensor.unsqueeze(0) / self.depth_scale
    
        # 处理无效深度值（0→NaN）[3](@ref)
        depth_tensor[depth_tensor == 0] = float('nan')
    
        return depth_tensor
    
    def normalized_depth(self,depth):
        depth_min = torch.nanmin(depth)
        depth_max = torch.nanmax(depth)
        normalized_depth = (depth - depth_min) / (depth_max - depth_min)
        return normalized_depth
  
    def tr3d2d(self,pointclouds, inter_matrix, transform_matrix, T, H = 480, W = 640):
        pointclouds = torch.cat(
                    (pointclouds, torch.ones(pointclouds.size(0), pointclouds.size(1), 1).to(pointclouds.device)),
                    dim=2)
        inter_matrix = inter_matrix[:, :, :3]  
        transform_matrix = torch.bmm(transform_matrix.float(), T.float())
        points = torch.matmul(pointclouds.to('cpu').float(), transform_matrix.transpose(1, 2))
        points = torch.matmul(points, inter_matrix.transpose(1, 2))
        z_coords = points[:, :, 2:3]
        
        positive_mask = (points >= 0).all(dim=2)
        positive_mask_z = (z_coords >= 0).all(dim=2)    

        points[~positive_mask] = 0
        z_coords[~positive_mask_z] = 0
        points = points / z_coords
        xy_points = points[:, :, :2]

        xy_points = self.replace_nan_values(xy_points)
        fov_mask = (xy_points[:, :, 0] <= W) & (xy_points[:, :, 1] <= H)
        xy_points[~fov_mask] = 0
        heatmap = self.proj(xy_points, pointclouds.size(0), H, W)
        heatmap = heatmap.permute(0, 3, 2, 1)

        return heatmap, xy_points
    
    def proj(self,xy_points, batch_size, H, W):
        image_mask = torch.zeros(batch_size, W, H, 1)  
        for point in range(xy_points.shape[1]):

            x = xy_points[0][point][0]
            y = xy_points[0][point][1]
        

            if 0 <= x < W and 0 <= y < H:
                image_mask[0, int(x), int(y), 0] = 1
        return image_mask
  
    def replace_nan_values(self,point_set):
        point_set.permute(0, 2, 1)
        nan_mask = torch.isnan(point_set) 
        cleaned_point_set = torch.where(nan_mask, torch.tensor([0.0, 0.0], device=point_set.device), point_set)
        cleaned_point_set.permute(0, 2, 1)
        return cleaned_point_set
 
    def depth_image_to_point_cloud(self,depth,kpts_depth, K, pose):
        """ pose.txt: camera-to-world, 4*4 matrix in homogeneous coordinates
            K.txt: camera intrinsics(3*3 matrix
        """

        # 确保深度图是二维张量
        if depth.dim() == 3:
            depth = depth.squeeze(0)
        
        # 提取特征点坐标
        u = kpts_depth[:, 0]  # x坐标 (列索引)
        v = kpts_depth[:, 1]  # y坐标 (行索引)
        
        # 将坐标转换为整数索引 (四舍五入)
        u_idx = torch.round(u).long()
        v_idx = torch.round(v).long()
        
        # 创建有效索引掩码 (防止越界)
        valid_mask = (u_idx >= 0) & (u_idx < depth.shape[1]) & \
                    (v_idx >= 0) & (v_idx < depth.shape[0])
        
        # 获取有效特征点深度值
        Z = torch.zeros_like(u)
        Z[valid_mask] = depth[v_idx[valid_mask], u_idx[valid_mask]]
        
        # 进一步筛选有效深度点 (深度 > 0)
        depth_mask = (Z > 0) & valid_mask
        u_valid = u[depth_mask]
        v_valid = v[depth_mask]
        Z_valid = Z[depth_mask]
        
        # 计算相机坐标系下的3D坐标
        X = (u_valid - K[0, 2]) * Z_valid / K[0, 0]
        Y = (v_valid - K[1, 2]) * Z_valid / K[1, 1]
        
        # 组合成相机坐标系下的点云 (N_valid, 3)
        points_cam = torch.stack((X, Y, Z_valid), dim=1)
        
        # 转换为齐次坐标 (N_valid, 4)
        ones = torch.ones((points_cam.shape[0], 1), device=depth.device)
        points_cam_homo = torch.cat((points_cam, ones), dim=1)
        
        # 变换到世界坐标系
        points_world = torch.matmul(pose, points_cam_homo.t()).t()  # (N_valid, 4)
        points = points_world[:, :3]  # 去除齐次坐标维度 (N_valid, 3)
        
        return points

