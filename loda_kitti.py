"""
对kitti数据中的图像的处理选择 裁剪至 160*512
图像的尺寸可以改变，为了保证点云投影到图像的正确性，需要调整相应的内参。

将点云投影成自己雷达图像坐标系下的深度图
将点云投影到相机图像坐标系下的深度图

点云的反射率信息，暂时先不考虑使用

点云随机变换后，需要补充一个变换矩阵

点云数据并没有进行下采样处理

superpoint + superglue 模型 输入的图像size不一致?

对视场角的处理？

superglue的注意力层数改变？


识别到的keypoints数量很小,很有可能点云的特征点最后没有一个投影到图像上

"""
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

class Kitti_Dataset(Dataset):
    """Dataset for loading image and depth data.
    Args:
        data_root (str): Root directory of the dataset.
        sequence_range (list): List of sequence numbers to include in the dataset.
    """
    def __init__(self,dataset_root,sequence_range):
        self.data_root = dataset_root
        self.sequence_range = range(sequence_range)
        #最终处理后的图像尺寸
        self.img_H = 160
        self.img_W = 512
        #点云随机变换范围
        self.trans_T = 10
        self.trans_R = 2*np.pi
        #range_image的尺寸
        self.proj_H = 64
        self.proj_W = 1024


        self.data = self.load_data()
    

    def __len__(self):
        """Returns the number of total data in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path,pointcloud_path,calib_path,cam = self.data[idx]
        #读取点云数据  kitti中的点云数据集为（x, y, z, reflectance）
        pointcloud_full = np.fromfile(pointcloud_path, dtype=np.float32)
        pointcloud_full = pointcloud_full.reshape(-1, 4)        
        pointcloud  = pointcloud_full[:, :3]
        #点云数据随机旋转变换，随机平移变换
        pointcloud, rotation_mat = self.random_rotation_z(pointcloud, max_angle=self.trans_R)
        pointcloud_xyz, translation_mat = self.random_translation(pointcloud, max_translation=self.trans_T)


        #读取图像数据，并将其size从320*1224调整为160*612
        img = cv2.imread(image_path)
        img = cv2.resize(img,
                        (int(round(img.shape[1] * 0.5)),
                        int(round((img.shape[0] * 0.5)))),
                        interpolation=cv2.INTER_LINEAR)
        #对图像中心裁剪 从160*612裁剪至160*512
        img_crop_dx = int((img.shape[1] - self.img_W) / 2)
        img_crop_dy = int((img.shape[0] - self.img_H) / 2)
        img = img[img_crop_dy:img_crop_dy + self.img_H,
            img_crop_dx:img_crop_dx + self.img_W  , :]
        #对图像数据增强处理
        img = self.augment_img(img)


        # 解析标定文件，提取相机内参矩阵、外参矩阵
        # calib文件格式参考链接：https://blog.csdn.net/weixin_43389152/article/details/129782159
        """
            Pi矩阵 = 相机i的内参矩阵K*相机0相对于相机i的外参矩阵Ti
            Pi =  [fx  0  cx  0]       [1 0 0 tx]
                  [0  fy  cy  0]   *  `[0 1 0 ty]
                  [0   0   1  0]       [0 0 1 tz]

            Tcami_velo=Ti*Tr
        """
        with open(calib_path, "r") as calib_file:
            lines = calib_file.readlines()
        if cam == "P2":
            P_matrix_str = lines[2].strip().split(" ")[1:]
        elif cam == "P3":
            P_matrix_str = lines[3].strip().split(" ")[1:]
        Tr_matrix_str = lines[-1].strip().split(" ")[1:]

        P_matrix = [float(x) for x in P_matrix_str]
        Tr_matrix = [float(y) for y in Tr_matrix_str]

        P_matrix = np.array(P_matrix).reshape(3, 4)
        K = P_matrix[:3, :3]
        # 调整内参矩阵K以适应裁剪后的图像尺寸
        K = self.camera_matrix_scaling(K, 0.5)
        K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)

        Tr_matrix = np.array(Tr_matrix).reshape(3, 4)
        fx = P_matrix[0, 0]
        fy = P_matrix[1, 1]
        cx = P_matrix[0, 2]
        cy = P_matrix[1, 2]
        tz = P_matrix[2, 3]
        tx = (P_matrix[0, 3] - cx * tz) / fx
        ty = (P_matrix[1, 3] - cy * tz) / fy
        Tcami_velo = Tr_matrix
        Tcami_velo[0, 3] += tx
        Tcami_velo[1, 3] += ty
        Tcami_velo[2, 3] += tz



        proj_depth = np.full((self.proj_H, self.proj_W), 0,dtype=np.float32)
        corrd_map = np.full((self.proj_H, self.proj_W, 3), [0,0,0], dtype=np.float32)

        # 计算球面坐标
        depth, phi, theta = self.cartesian_to_spherical(pointcloud_xyz)

        #phi 和 yaw 互为相反数？
        yaw = - np.arctan2(pointcloud_xyz[:, 1],pointcloud_xyz[:, 0])  # 水平方位角
        proj_x = 0.5 * (yaw / np.pi + 1.0)* self.proj_W       #方位角归一化到 [0.0, proj_W]

        #Velodyne激光雷达的垂直视场角约为26.8度
        fov = 26.8 * np.pi/180
        pitch = np.arcsin(pointcloud_xyz[:, 2] / depth)  # 俯仰角
        proj_y = ((fov/2 -pitch) / fov) * self.proj_H      #俯仰角映射到 [0, proj_H]


        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]

        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]

        corrd = pointcloud_xyz
        
  
        corrd = corrd[order]
        depth = depth[order]
        indices = indices[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        proj_depth[proj_y, proj_x] = depth
        corrd_map[proj_y, proj_x] = corrd
        image3d_D = (proj_depth / proj_depth.max() * 255).astype(np.uint8)

        T = np.eye(4)
        T[:3, :3] = rotation_mat
        T[:2, 3] = translation_mat[:2]

        T_inv = np.linalg.inv(T).astype(np.float32)

        #返回会归一化后的图像，归一化后的深度图，点云投影的深度图坐标映射，调整后的相机内参矩阵K，相机到激光雷达的外参矩阵Tcami_velo，点云随机变换矩阵T_inv
        return torch.from_numpy(img.astype(np.float32) / 255.),torch.from_numpy(image3d_D.astype(np.float32) / 255.),  \
                torch.from_numpy(corrd_map),torch.from_numpy(K),torch.from_numpy(Tcami_velo),torch.from_numpy(T_inv),
        


    def load_data(self):
        """Loads the image and depth data from the dataset."""
        data = []
        color_dir = os.path.join(self.data_root, 'data_odometry_color')
        velodyne_dir = os.path.join(self.data_root, 'data_odometry_velodyne')
        calib_dir = os.path.join(self.data_root, 'data_odometry_calib')
        for seq in self.sequence_range:
            image2_dir = os.path.join(color_dir, f'dataset/sequences/{seq:02d}', 'image_2')
            image3_dir = os.path.join(color_dir, f'dataset/sequences/{seq:02d}', 'image_3')
            pointcloud_dir = os.path.join(velodyne_dir, f'dataset/sequences/{seq:02d}', 'velodyne')
            calib_path = os.path.join(calib_dir, f'dataset/sequences/{seq:02d}', 'calib.txt')
            
            image2_files = sorted(os.listdir(image2_dir))
            image3_files = sorted(os.listdir(image3_dir))
            pointcloud_files = sorted(os.listdir(pointcloud_dir))

            for img2_file, img3_file, pointcloud_file in zip(image2_files, image3_files, pointcloud_files):
                img2_path = os.path.join(image2_dir, img2_file)
                img3_path = os.path.join(image3_dir, img3_file)
                pointcloud_path = os.path.join(pointcloud_dir, pointcloud_file)
                data.append((img2_path, pointcloud_path,calib_path,"P2"))
                data.append((img3_path, pointcloud_path,calib_path,"P3"))
        return data




    #图像增强
    def augment_img(self, img_np):
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(img_np)))

        return img_color_aug_np

    #resize图像后调整相机内参矩阵K
    def camera_matrix_scaling(self, K: np.ndarray, s: float):
        K_scale = s * K
        K_scale[2, 2] = 1
        return K_scale
    #裁剪图像后调整内参矩阵K
    def camera_matrix_cropping(self, K: np.ndarray, dx: float, dy: float):
        K_crop = np.copy(K)
        K_crop[0, 2] -= dx
        K_crop[1, 2] -= dy
        return K_crop
    #随机旋转变换
    def random_rotation_z(self,points, max_angle=np.pi):
        angle = np.random.uniform(-max_angle, max_angle)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]])
        points = np.dot(points, rotation_matrix.T)
        return points, rotation_matrix
    #随机平移变换
    def random_translation(self,points, max_translation=10):
        translation = np.random.uniform(-max_translation, max_translation, size=(3,))
        points[:, :2] += translation[:2]
        return points, translation

    def cartesian_to_spherical(self,cartesian_points):
        x, y, z = cartesian_points[:, 0], cartesian_points[:, 1], cartesian_points[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x) 
        theta = np.arccos(z / r)  
        return r, phi, theta 



class KITTIDepthProjection:
    """KITTI点云到深度图的投影工具"""

    def __init__(self, img_width=1242, img_height=375):
        self.img_width = img_width
        self.img_height = img_height

    def load_calib(self, calib_file):
        """加载KITTI标定文件"""
        calib = {}
        with open(calib_file, 'r') as f:
            lines = f.readlines()

        # P2: 左相机投影矩阵
        P2 = lines[2].strip().split()[1:]
        P2 = np.array([float(x) for x in P2]).reshape(3, 4)
        calib['P2'] = P2

        # Tr: 激光雷达到相机的变换矩阵
        Tr = lines[-1].strip().split()[1:]
        Tr = np.array([float(x) for x in Tr]).reshape(3, 4)
        Tr_4x4 = np.eye(4)
        Tr_4x4[:3, :] = Tr
        calib['Tr'] = Tr_4x4

        return calib

    def project_pointcloud_to_depth(self, pointcloud_file, calib_file):
        """
        将点云投影到图像平面生成深度图

        Args:
            pointcloud_file: 点云文件路径
            calib_file: 标定文件路径

        Returns:
            depth_map: 深度图 (H, W)
            depth_mask: 有效深度区域掩码 (H, W)
        """
        # 加载数据
        calib = self.load_calib(calib_file)
        pointcloud = np.fromfile(pointcloud_file, dtype=np.float32).reshape(-1, 4)

        # 提取XYZ坐标
        points = pointcloud[:, :3]  # (N, 3)

        # 转换为齐次坐标
        points_homo = np.hstack((points, np.ones((points.shape[0], 1))))

        # 激光雷达到相机坐标系的变换
        points_cam = np.dot(calib['Tr'], points_homo.T).T

        # 过滤掉相机后方的点
        valid_mask = points_cam[:, 2] > 0
        points_cam = points_cam[valid_mask]

        # 投影到图像平面
        points_2d_homo = np.dot(calib['P2'], points_cam.T).T

        # 转换为笛卡尔坐标
        depths = points_2d_homo[:, 2]
        points_2d = points_2d_homo[:, :2] / depths.reshape(-1, 1)

        # 创建深度图
        depth_map = np.zeros((self.img_height, self.img_width))
        depth_mask = np.zeros((self.img_height, self.img_width), dtype=bool)

        # 将点投影到深度图
        for i, (x, y, depth) in enumerate(zip(points_2d[:, 0], points_2d[:, 1], depths)):
            if 0 <= x < self.img_width and 0 <= y < self.img_height:
                ix, iy = int(x), int(y)
                # 使用最近点的深度值
                if not depth_mask[iy, ix] or depth < depth_map[iy, ix]:
                    depth_map[iy, ix] = depth
                    depth_mask[iy, ix] = True

        return depth_map, depth_mask






if __name__ == "__main__":
    data_root = "./dataset"

    sequence_range_train = range(10) 
    data = Kitti_Dataset(data_root,'3dmatch',[640,480])
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

