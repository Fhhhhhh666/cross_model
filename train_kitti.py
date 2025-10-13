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


torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')
[[]]
parser = argparse.ArgumentParser(
    description='DepthImage and RGBImage matching ',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--epoch', type=int, default=5, help='Number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate')

parser.add_argument(
    '--nms_radius', type=int, default=4,
    help='SuperPoint Non Maximum Suppression (NMS) radius'
' (Must be positive)')
parser.add_argument(
    '--keypoint_threshold', type=float, default=0.0,
    help='SuperPoint keypoint detector confidence threshold')
parser.add_argument(
    '--max_keypoints', type=int, default=512,
    help='Maximum number of keypoints detected by Superpoint'
            ' (\'-1\' keeps all keypoints)')
parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')

parser.add_argument("--dataset_root", type=str, default='/media/autolab/225nas/KITTI')
parser.add_argument("--sequences", type=int, default=9, help='training sequences')


parser.add_argument("--checkpoint_dir", type=str, default="checkpoint")


def tr3d2d(corrd_map,selected, K, transform, T, H, W):
    indices = []
    good_values_int = []
    for i in range(corrd_map.shape[0]):
        #ones_array (proj_H,proj_W,1）
        ones_array = torch.ones((corrd_map[i].shape[0], corrd_map[i].shape[1], 1),device=corrd_map[i].device)
        # corrds (proj_H,proj_W,4)
        corrds = torch.cat((corrd_map[i], ones_array), axis=2)
        #corrds_flat (proj_H*proj_W,4)
        corrds_flat = corrds.view(-1, 4) 
        inter_matrix = K[i]
        transform_matrix = torch.matmul(transform[i].float(), T[i].float())
        corrds = torch.matmul(transform_matrix, corrds_flat.float().t())
        corrds = torch.matmul(inter_matrix.float(), corrds).t()
        #corrds (proj_H,proj_W,3)
        corrds = corrds.view(corrd_map[i].shape[0], corrd_map[i].shape[1], 3)
        z_coords = corrds[:, :, 2:3]
        #positive_mask (proj_H,proj_W)  x,y,z都大于等于0
        positive_mask = (corrds >= 0).all(dim=2)
        #positive_mask_z (proj_H,proj_W)  z大于等于0
        positive_mask_z = (z_coords >= 0).all(dim=2)
        # 将不满足条件的点置零
        corrds = torch.where(positive_mask.unsqueeze(-1), corrds, torch.zeros_like(corrds))  #把投影坐标负数的点筛掉
        corrds = torch.where(selected[i].unsqueeze(-1), corrds, torch.zeros_like(corrds))    #把非特征点筛掉  
        z_coords = torch.where(positive_mask_z.unsqueeze(-1), z_coords, torch.zeros_like(z_coords)) #把z负数的点筛掉
        #把z为0的点置为一个很小的数，防止除零溢出
        z_coords = torch.where(z_coords == 0, torch.tensor(1e-9, device=z_coords.device), z_coords)
        #得到最终坐标
        corrds = corrds / z_coords
        #去掉z坐标
        xy_points = corrds[:, :, :2]
        
        org_mask = (corrd_map[i] != torch.tensor([0, 0, 0], device=corrd_map[i].device)).any(dim=2)  
        non_zero_mask = (xy_points != torch.tensor([0, 0], device=xy_points.device)).any(dim=2)
        fov_mask = (xy_points[:, :, 0] <= W) & (xy_points[:, :, 1] <= H)
        final_mask = org_mask & non_zero_mask & fov_mask

        indices.append(torch.nonzero(final_mask))   #（n,2)
        # good_values = xy_points[final_mask]         #(n,2)
        # print(good_values.shape)
        xy_points[~final_mask] = 0
        good_values_int.append(xy_points[: , :, [1, 0]].long())

    return indices, good_values_int


def fov_loss(xy, fov):
    loss = 0
    # assert len(xy) == 0, "xy is empty"
    if len(xy) == 0:
        return torch.tensor(0.0, device='cuda')
    for i in  range(len(xy)):
        xy_points = xy[i].to('cuda')
        fov_score = fov[i].to('cuda')
        BCELoss = torch.nn.BCELoss()
        score_tensor = torch.ones(fov_score.shape).squeeze(0).to('cuda')
        zero_indices = (xy_points[:, 0] == 0) & (xy_points[:, 1] == 0).to('cuda')
        score_tensor[zero_indices] = 0
        loss += BCELoss(fov_score, score_tensor)
    return loss/len(xy)

def des_loss_org(sxy_points, kp2d, sim_list, scores0, scores1, th1=3, th2=3):
    loss1 = 0
    loss2 = 0
    BCELoss = torch.nn.BCELoss()

    for i in range(kp2d.shape[0]):
        p1 = sxy_points[i].clone().to('cuda')
        p2 = kp2d[i].clone().to('cuda').to(torch.float32)
        sc0 = torch.exp(scores0[i]).squeeze(0).to('cuda')
        sc1 = torch.exp(scores1[i]).squeeze(0).to('cuda')

        score_tensor0 = torch.zeros(sc0.shape).to('cuda')
        score_tensor1 = torch.zeros(sc1.shape).to('cuda')
        zero_indices = (p1[:, 0] == 0) & (p1[:, 1] == 0).to('cuda')
        zero_indices = zero_indices.nonzero().squeeze(1).to('cuda')

        distances = torch.cdist(p1.unsqueeze(0), p2.unsqueeze(0)).squeeze(0)

        min_distances1, min_index1 = torch.min(distances, dim=1)
        min_distances2, min_index2 = torch.min(distances, dim=0)
        min_distances1[zero_indices] = th2
        pairs_indices1 = torch.nonzero(min_distances1 < th1, as_tuple=False)
        no_pairs_indices1 = torch.nonzero(min_distances1 >= th2, as_tuple=False)
        pairs_indices2 = torch.nonzero(min_distances2 < th1, as_tuple=False)
        no_pairs_indices2 = torch.nonzero(min_distances2 >= th2, as_tuple=False)

        des_pairs1 = sim_list[i][pairs_indices1, min_index1[pairs_indices1]]
        # des_pairs2 = sim_list[i][min_index2[pairs_indices2], pairs_indices2]

        score_tensor0[pairs_indices1[:, 0]] = 1
        score_tensor1[pairs_indices2[:, 0]] = 1
        # 添加数值稳定性检查
        # assert torch.isnan(des_pairs1).any()," des_pairs1 contains NaN values"
        # assert des_pairs1.numel() == 0," des_pairs1 is empty"

        if des_pairs1.numel() > 0 and not torch.isnan(des_pairs1).any():
            sc0loss = BCELoss(sc0, score_tensor0)
            sc1loss = BCELoss(sc1, score_tensor1)
            loss1 += - des_pairs1.mean()
            loss2 += sc0loss + sc1loss
    return loss1 / len(kp2d), loss2 / len(kp2d)


if __name__ == '__main__':
    opt = parser.parse_args()
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
        }
    }
    # 在开头定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(opt)

    # load training data
    train_set = Kitti_Dataset(opt.dataset_root, opt.sequences)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=opt.batch_size, drop_last=True)
    #注意迁移到指定设备上
    superpoint = SuperPoint(config.get('superpoint', {})).to(device)
    superglue = SuperGlue(config.get('superglue', {})).to(device)

    optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
    
    superglue.train()
    #计算每个epoch的平均loss
    loss_epoch_avg = []

    for epoch in range(1,opt.epoch+1):
        start_time = time.time()
        print(f"epoch #{epoch}")
        loss_epoch = []
        for i, batch in enumerate(tqdm(train_loader)):
            #获取数据
            """ 
            img: (B, H, W, 3) RGB图像
            img3d: (B, H, W) 深度图像
            corrd_map: (B, proj_H, proj_W, 3) 点云图像
            inter_matrix: (B, 3, 3) 相机内参矩阵
            transform_matrix: (B, 3, 4) 相机外参矩阵
            T_inv: (B, 4, 4) 点云到相机坐标系的变换矩阵
                """
            img,img3d,corrd_map,inter_matrix,transform_matrix,T_inv = batch
            #调整数据格式,送入superpoint
            img = img.permute(0, 3, 1, 2)
            img3d = img3d.unsqueeze(-1).permute(0, 3, 1, 2)
            #将数据移动到cuda上
            img = img.to(device)
            img3d = img3d.to(device)
            corrd_map = corrd_map.to(device)
            inter_matrix = inter_matrix.to(device)
            transform_matrix = transform_matrix.to(device)
            T_inv = T_inv.to(device)

            #生成特征点
            img_pred = superpoint({"image": img})
            img3d_pred = superpoint({"image": img3d})
            
            #将list转为tensor
            for k in img_pred:
                if isinstance(img_pred[k], (list, tuple)):
                    img_pred[k] = torch.stack(img_pred[k])
            for k in img3d_pred:
                if isinstance(img3d_pred[k], (list, tuple)):
                    img3d_pred[k] = torch.stack(img3d_pred[k])

            optimizer.zero_grad()

            matches, sim_list,fov_score,score0, score1 = superglue({
                'image' : img,
                'depth' : img3d,
                'keypoints_rgb': img_pred['keypoints'],
                'keypoints_depth': img3d_pred['keypoints'],
                'scores_rgb': img_pred['scores'],
                'scores_depth': img3d_pred['scores'],
                'desc_rgb': img_pred['descriptors'],
                'desc_depth': img3d_pred['descriptors']
            })

            kp2d = img_pred['keypoints']
            kp3d = img3d_pred['keypoints']

            #创建一个在corrd_map上的掩码，标记识别到的特征点
            selected = torch.zeros(corrd_map.shape[0], corrd_map.shape[1],corrd_map.shape[2],dtype=torch.bool).to(device)
            for b in range(corrd_map.shape[0]):
                points = kp3d[b]
                #注意这里要转换为整数索引
                X = points[:, 0].long()
                Y = points[:, 1].long()
                selected[b, Y, X] = True
            #xy_points (B, proj_H, proj_W, 2),只有投影在图像内的特征点坐标不为0
            indices, xy_points = tr3d2d(corrd_map, selected, inter_matrix, transform_matrix, T_inv, img.shape[-2], img.shape[-1])
            map = torch.zeros(kp3d.shape[0], kp3d.shape[1], 2).to(device)
            for b in range(kp3d.shape[0]):
                for idx in range(kp3d.shape[1]):
                    if (xy_points[b][int(kp3d[b][idx][1])][int(kp3d[b][idx][0])] != torch.tensor([0, 0]).device):
                        map[b][idx] = xy_points[b][int(kp3d[b][idx][1])][int(kp3d[b][idx][0])]
            
            loss1 = fov_loss(map, fov_score)
            loss2, loss3 = des_loss_org(map, kp2d, sim_list, score0, score1)
            loss = loss1 + loss2 + loss3

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected at epoch {epoch}, batch {i}")
                print(f"loss1: {loss1}, loss2: {loss2}, loss3: {loss3}")
                continue

            print(loss)
            loss.requires_grad_(True) 
            loss.backward()
            optimizer.step()
 
            loss_epoch += [loss.item()]
            # print("--- %s seconds ---" % (time.time() - start_time))
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f'Epoch [{epoch}/{opt.epoch}] Total Time: {epoch_duration:.4f} seconds')

        torch.save(superglue.state_dict(), opt.checkpoint_dir + f"Encoder_epoch_{epoch}.t7")
        loss_epoch_avg += [sum(loss_epoch) / len(loss_epoch)]