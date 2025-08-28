
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.autograd import Variable
from loda_data import image_depth_Dataset
import os
import torch.multiprocessing
from tqdm import tqdm
from models.superglue import SuperGlue
import time


torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(
    description='DepthImage and RGBImage matching ',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')

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

def des_loss_org(sxy_points, kp2d, sim_list, scores0, scores1, th1=3, th2=3):
    loss1 = 0
    loss2 = 0
    BCELoss = torch.nn.BCELoss()

    for i in range(kp2d.shape[0]):
        p1 = sxy_points[i].clone().to('cpu')
        p2 = kp2d[i].clone().to('cpu').to(torch.float32)
        sc0 = torch.exp(scores0[i]).squeeze(0).to('cpu')
        sc1 = torch.exp(scores1[i]).squeeze(0).to('cpu')
        score_tensor0 = torch.zeros(sc0.shape)
        score_tensor1 = torch.zeros(sc1.shape)
        zero_indices = (p1[:, 0] == 0) & (p1[:, 1] == 0)
        zero_indices = zero_indices.nonzero().squeeze(1)

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
        sc0loss = BCELoss(sc0, score_tensor0)
        sc1loss = BCELoss(sc1, score_tensor1)
        # des_no_pairs1 = sim_list[i][no_pairs_indices1, :].squeeze(-1)
        # des_no_pairs2 = sim_list[i][:, no_pairs_indices2].squeeze(-1)
        # score_no_pairs1 = sc0[no_pairs_indices1]
        # score_no_pairs2 = sc1[no_pairs_indices2]
        # loss += 2 - des_pairs1.mean() - des_pairs2.mean() + des_no_pairs1.mean()  + des_no_pairs2.mean()
        # loss += - (des_pairs2.mean() + torch.log(1 - torch.exp(score_no_pairs1.mean())) + torch.log(1 - torch.exp(score_no_pairs2.mean())))
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
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    print(opt)

    # load training data
    train_set = image_depth_Dataset(opt.dataset_root, opt.name, opt.image_size, opt.camera_intrinsics, opt.depth_scale)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=opt.batch_size, drop_last=True)

    superglue = SuperGlue(config.get('superglue', {}))

    if torch.cuda.is_available():
        superglue.cuda() # make sure it trains on GPU
    else:
        print("### CUDA not available ###")
    optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
    superglue.double().train()
    loss_epoch_avg = []
    for epoch in range(1,opt.epoch+1):
        start_time = time.time()
        print(f"epoch #{epoch}")
        loss_epoch = []
        running_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader)):
            batch_start_time = time.time()
            for k in batch:
                if k == 'keypoints_rgb' or k == 'keypoints_depth' or k== 'desc_image' or 'desc_depth' or k=='xy_points':
                    if type(batch[k]) == torch.Tensor:
                        batch[k] = Variable(batch[k].cuda())
                    else:
                        batch[k] = Variable(torch.stack(batch[k]).cuda())
                              
            matches, sim_list, score0, score1 = superglue(batch)
            loss1, loss2 = des_loss_org(batch['xy_points'], batch['keypoints_rgb'], sim_list, score0, score1)
            loss = loss1 + loss2

            print(loss) 

            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_epoch += [loss.item()]
            print("--- %s seconds ---" % (time.time() - start_time))
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f'Epoch [{epoch}/{opt.epoch}] Total Time: {epoch_duration:.4f} seconds')
        torch.save(superglue.state_dict(), opt.checkpoint_dir + f"Encoder_epoch_{epoch}.t7")
        loss_epoch_avg += [sum(loss_epoch) / len(loss_epoch)]