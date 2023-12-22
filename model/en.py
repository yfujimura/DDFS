import torch
from torch import nn
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    
    def __init__(self, frame_num=1, depth_steps=64, cmax=0.3, f1=0.999, cv_outlier_removal=1, cv_normalization=1):
        super().__init__()
        
        self.frame_num = frame_num
        self.depth_steps = depth_steps
        self.cmax = cmax
        self.f1 = f1
        self.cv_outlier_removal = cv_outlier_removal
        self.cv_normalization = cv_normalization
        
        self.conv1 = self._downConvLayer(frame_num*3+depth_steps, 128, 7)
        self.conv2 = self._downConvLayer(128, 256, 5)
        self.conv3 = self._downConvLayer(256, 512, 3)
        self.conv4 = self._downConvLayer(512, 512, 3)
        self.conv5 = self._downConvLayer(512, 512, 3)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, images, cost_volume):
        # images: N x F x 3 x H x W
        # cost_volume: N x depth_steps x height x width 
        
        if self.cv_outlier_removal == 1:
            cost_volume = self.remove_outlier(cost_volume, self.cmax, f1 = self.f1)
        if self.cv_normalization == 1:
            cost_volume = self.normalize_cost_volume(cost_volume)
            
        x = torch.cat((images[:,-1,:,:,:], cost_volume), 1)  # 67 x H x W
        
        conv1 = self.conv1(x)  # 128 x H/2 x W/2
        conv2 = self.conv2(conv1)  # 256 x H/4 x W/4  (48, 64)
        conv3 = self.conv3(conv2)  # 512 x H/8 x W/8  (24, 32)
        conv4 = self.conv4(conv3)  # 512 x H/16 x W/16  (12, 16)
        conv5 = self.conv5(conv4)  # 512 x H/32 x W/32   (6, 8)
        
        return [conv1, conv2, conv3, conv4, conv5]
        
        
        
    def _downConvLayer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def remove_outlier(self, cost_volume, cmax, f1=0.999):
        a = math.log((1+f1) / (1-f1)) / (2 * cmax)
        return F.tanh(a * cost_volume)
    
    def normalize_cost_volume(self, cost_volume):
        # cost_volume: N x depth_steps x height x width 
        cmax = torch.max(cost_volume, dim=1, keepdim=True)[0]
        cmin = torch.min(cost_volume, dim=1, keepdim=True)[0]
        return (cost_volume - cmin) / (cmax-cmin + 1e-8)
    