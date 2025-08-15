import torch
import torch.nn as nn
from models.superpoint import SuperPoint
from models.superglue import SuperGlue

class RangeImageMatcher(nn.Module):
    def __init__(self, config={}):
        super().__init__()
        # 初始化SuperPoint模型
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        
        # 初始化SuperGlue模型
        self.superglue = SuperGlue(config.get('superglue', {}))
        
    def forward(self, data):
        # 处理相机图像
        image_data = {
            'image': data['image'],
            'keypoints0': None,
            'scores0': None,
            'descriptors0': None
        }
        image_result = self.superpoint(image_data)
        
        # 处理距离图像
        range_image_data = {
            'image': data['range_image'],
            'keypoints0': None,
            'scores0': None,
            'descriptors0': None
        }
        range_result = self.superpoint(range_image_data)
        
        # 使用SuperGlue进行匹配
        matches_data = {
            'keypoints0': image_result['keypoints'][0],
            'keypoints1': range_result['keypoints'][0],
            'scores0': image_result['scores'][0],
            'scores1': range_result['scores'][0],
            'descriptors0': image_result['descriptors'][0],
            'descriptors1': range_result['descriptors'][0]
        }
        matches_result = self.superglue(matches_data)
        
        return {
            'image_keypoints': image_result['keypoints'],
            'range_keypoints': range_result['keypoints'],
            'matches': matches_result['matches0'],
            'matching_scores': matches_result['matching_scores0']
        }