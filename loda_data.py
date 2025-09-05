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
    def __init__(self, config,data_root, name,image_size,camera_intrinsics = [[585,0,320],[0,585,240],[0,0,1]], depth_scale=1000.0):
        self.data_root = data_root
        self.dataset_name = name
        self.dataset_path = os.path.join(data_root, name)
        self.dataset_classes = os.listdir(self.dataset_path)
        self.image_size = image_size
        self.camera_intrinsics = torch.tensor(camera_intrinsics, dtype=torch.float32)
        self.depth_scale = depth_scale
        self.data = self.load_data()
        self.superpoint = superpoint.SuperPoint(config.get('superpoint', {}))
    

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
        scores_rgb = pred_rgb['scores'][0]
        desc_rgb = pred_rgb['descriptors'][0]

        kpts_depth = pred_depth['keypoints'][0]
        scores_depth = pred_depth['scores'][0]
        desc_depth = pred_depth['descriptors'][0]

        # 筛选 kpts_depth 对应的深度图为 0 的点
        u = torch.round(kpts_depth[:, 0]).long()
        v = torch.round(kpts_depth[:, 1]).long()
        valid_mask = (u >= 0) & (u < depth.shape[2]) & (v >= 0) & (v < depth.shape[1])
        depth_values = torch.zeros_like(u, dtype=depth.dtype)
        depth_values[valid_mask] = depth[0, v[valid_mask], u[valid_mask]]
        nonzero_mask = (depth_values > 0) & valid_mask

        kpts_depth = kpts_depth[nonzero_mask]
        scores_depth = scores_depth[nonzero_mask]
        desc_depth = desc_depth[:, nonzero_mask]

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
            'scores_rgb': scores_rgb,         # RGB图像特征点分数 [M]
            'scores_depth': scores_depth,     # 深度图像特征点分数 [N]
            'image_path': image_file,         # 图像路径
            'kpts_depth_pointclouds': kpts_depth_pointclouds, # 深度图特征点对应的点云坐标 [N_valid, 3]
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
                        image_file = os.path.join(cls_path, f"{base_name}.color.png")
                        depth_file = os.path.join(cls_path, f"{base_name}.depth.png")
                        data.append((pose_file, image_file, depth_file))
        return data
    
    def load_pose(self, pose_file):
        """加载并解析位姿文件为4x4变换矩阵  camera-to-world"""
        pose_matrix = np.loadtxt(pose_file)  # 直接读取为4x4矩阵
        assert pose_matrix.shape == (4, 4), "Pose matrix must be 4x4"
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
        
        return depth_tensor
    
    def normalized_depth(self, depth):
        valid_mask = ~torch.isnan(depth)
        depth_min = torch.min(depth[valid_mask])
        depth_max = torch.max(depth[valid_mask])
        normalized_depth = (depth - depth_min) / (depth_max - depth_min)
        return normalized_depth
  
    def tr3d2d(self, pointclouds, inter_matrix, transform_matrix, H=480, W=640):
        # pointclouds: (N, 3)
        # inter_matrix: (3, 3)
        # transform_matrix: (4, 4)
        # 转为齐次坐标
        ones = torch.ones((pointclouds.shape[0], 1), device=pointclouds.device)
        points_homo = torch.cat([pointclouds, ones], dim=1)  # (N, 4)
        # 世界坐标 -> 相机坐标
        points_cam = torch.matmul(torch.inverse(transform_matrix), points_homo.t()).t()  # (N, 4)
        points_cam = points_cam[:, :3]
        # 相机坐标 -> 像素坐标
        xy = torch.matmul(inter_matrix, points_cam.t()).t()  # (N, 3)
        z = xy[:, 2]
        # 防止除零
        valid_mask = z > 0
        xy_proj = torch.zeros_like(xy[:, :2])
        xy_proj[valid_mask] = xy[valid_mask, :2] / z[valid_mask].unsqueeze(1)
        # 清理无效值
        xy_proj = self.replace_nan_values(xy_proj)
        # 视野范围mask
        fov_mask = (xy_proj[:, 0] >= 0) & (xy_proj[:, 0] < W) & (xy_proj[:, 1] >= 0) & (xy_proj[:, 1] < H)
        xy_proj[~fov_mask] = 0
        # 生成heatmap
        heatmap = self.proj(xy_proj.unsqueeze(0), 1, H, W)  # batch_size=1
        heatmap = heatmap.permute(0, 3, 2, 1)
        return heatmap, xy_proj
    
    def proj(self,xy_points, batch_size, H, W):
        image_mask = torch.zeros(batch_size, W, H, 1)  
        for point in range(xy_points.shape[1]):

            x = xy_points[0][point][0]
            y = xy_points[0][point][1]
        

            if 0 <= x < W and 0 <= y < H:
                image_mask[0, int(x), int(y), 0] = 1
        return image_mask
  
    def replace_nan_values(self, point_set):
        # point_set: (N, 2)
        nan_mask = torch.isnan(point_set)
        cleaned_point_set = torch.where(nan_mask, torch.zeros_like(point_set), point_set)
        return cleaned_point_set
    
    def depth_image_to_point_cloud(self,depth,kpts_depth, K, pose):
        """ pose.txt: camera-to-world, 4*4 matrix in homogeneous coordinates
            K.txt: camera intrinsics(3*3 matrix
        """
        # 处理无效深度值（0→NaN）[3](@ref)
        depth[depth == 0] = float('nan')
        
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

if __name__ == "__main__":
    data_root = "./dataset"

    sequence_range_train = range(10) 
    data = image_depth_Dataset(data_root,'3dmatch',[640,480])
    print(data)
    print(data.data_root)
    print(data.dataset_name)
    print(data.dataset_path)
    print(data.dataset_classes)
    datas = []
    for cls in data.dataset_classes:
        cls_path = os.path.join(data.dataset_path, cls, 'query')
        if os.path.isdir(cls_path):
            for file in os.listdir(cls_path):
                if '_pose.txt' in file:
                    pose_file = os.path.join(cls_path, file)
                    # 查找对应的图像和深度文件
                    base_name = file.replace('_pose.txt', '')
                    image_file = os.path.join(cls_path, f"{base_name}_color.png")
                    depth_file = os.path.join(cls_path, f"{base_name}_depth.png")
                    datas.append((pose_file, image_file, depth_file))
    print(len(datas))    

