import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.matching import RangeImageMatcher
from loda_data import KITTIRangeImageDataset
import numpy as np

def train(config):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建数据集
    dataset = KITTIRangeImageDataset(
        data_root=config['data_root'],
        sequence_range=range(0, 10)
    )
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 初始化模型
    model = RangeImageMatcher(config['model']).to(device)
    
    # 设置优化器
    optimizer = Adam(model.parameters(), lr=config['lr'])
    
    # 训练循环
    for epoch in range(config['epochs']):
        for i, batch in enumerate(dataloader):
            # 将数据移至设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 前向传播
            results = model(batch)
            
            # 计算损失 (这里使用重投影误差作为监督信号)
            loss = compute_reprojection_loss(results, batch['calib'])
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 打印训练信息
            if i % config['log_interval'] == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
        
        # 保存检查点
        if epoch % config['save_interval'] == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")

def compute_reprojection_loss(results, calib):
    """
    使用标定参数计算重投影误差作为损失函数
    """
    # 获取匹配点对
    matches = results['matches']
    valid_matches = matches[0, :] > -1
    
    # 获取匹配的相机图像关键点
    image_kpts = results['image_keypoints'][0, valid_matches]
    
    # 获取匹配的距离图像关键点
    range_kpts = results['range_keypoints'][0, matches[0, valid_matches]]
    
    # 将距离图像关键点转换为3D点 (反向投影)
    points_3d = range_keypoints_to_3d(range_kpts, calib)
    
    # 将3D点投影到相机图像平面
    projected_points = project_3d_to_image(points_3d, calib['P2'])
    
    # 计算重投影误差
    reprojection_error = torch.norm(image_kpts - projected_points, dim=1)
    return torch.mean(reprojection_error)

def range_keypoints_to_3d(keypoints, calib):
    """
    将距离图像关键点转换回3D坐标
    """
    # 实现细节：根据距离图像坐标和深度值计算3D坐标
    # 这里需要根据具体参数实现
    pass

def project_3d_to_image(points_3d, P2):
    """
    使用相机投影矩阵将3D点投影到图像平面
    """
    # 添加齐次坐标
    points_3d_h = torch.cat([points_3d, torch.ones(points_3d.shape[0], 1)], dim=1)
    
    # 投影
    points_2d_h = torch.mm(points_3d_h, P2.T)
    
    # 归一化
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:]
    return points_2d

if __name__ == "__main__":
    config = {
        'data_root': '../dataset',
        'batch_size': 4,
        'epochs': 100,
        'lr': 0.001,
        'log_interval': 10,
        'save_interval': 5,
        'model': {
            'superpoint': {
                'descriptor_dim': 256,
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'descriptor_dim': 256,
                'weights': 'indoor',
                'keypoint_encoder': [32, 64, 128, 256],
                'GNN_layers': ['self', 'cross'] * 9,
                'sinkhorn_iterations': 100
            }
        }
    }
    train(config)