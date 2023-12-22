import torch
from torch import nn
import torch.nn.functional as F
import math


class CostAggregation(nn.Module):
    
    def __init__(self, in_channels, out_channels, height, width, dilation=2, neighbors=9, depth_steps=64, blocks=1):
        super().__init__()
        
        self.height = height
        self.width = width
        self.dialtion = dilation
        self.neighbors = neighbors
        self.depth_steps = depth_steps
        self.blocks = blocks
        
        iconvs = []
        score_convs = []
        offset_convs = []
        weight_convs = []
        
        for i in range(blocks):
            iconvs.append(self._convLayer(depth_steps+in_channels, out_channels, 3))
            score_convs.append(self._scoreLayer(out_channels, depth_steps))
            offset_convs.append(nn.Conv2d(out_channels, 2*neighbors, 3, stride=1, padding=dilation, dilation=dilation))
            weight_convs.append(nn.Conv2d(out_channels, neighbors, 3, stride=1, padding=dilation, dilation=dilation))
            
        self.iconvs = nn.ModuleList(iconvs)
        self.score_convs = nn.ModuleList(score_convs)
        self.offset_convs = nn.ModuleList(offset_convs)
        self.weight_convs = nn.ModuleList(weight_convs)
        
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
                    
        for offset_conv in self.offset_convs:
            nn.init.constant_(offset_conv.weight, 0.0)
            nn.init.constant_(offset_conv.bias, 0.0)
            
        
        y_grid, x_grid = torch.meshgrid(
            [
                torch.arange(0, height, dtype=torch.float32),
                torch.arange(0, width, dtype=torch.float32),
            ])
        pixel = torch.stack((x_grid, y_grid), 2)
        pixel = pixel.unsqueeze(0).to("cuda") # 1 x H x W x 2
            
        original_grid = torch.tensor([[-dilation, -dilation],
                                        [0,-dilation],
                                        [dilation,-dilation],
                                        [-dilation, 0],
                                        [0, 0],
                                        [dilation, 0],
                                        [-dilation, dilation],
                                        [0, dilation],
                                        [dilation, dilation]]).to("cuda") # neighbors x 2
            
        original_grid = pixel + original_grid.unsqueeze(0).unsqueeze(2).unsqueeze(3) # 1 x neighbors x H x W x 2
        self.original_grid = original_grid.view(1, neighbors*height, width, 2) # 1 x neighbors*H x W x 2
            
            
    def forward(self, score, in_feature):
        
        for i in range(self.blocks):
            iconv = self.iconvs[i](torch.cat((score, in_feature), 1))
            score = self.score_convs[i](iconv)
            offset = self.offset_convs[i](iconv)
            weight = self.weight_convs[i](iconv)
            weight = F.softmax(weight, dim=1)
            score = self.cost_aggregation(score, offset, weight)
            
        return score
            
    def cost_aggregation(self, score, offset, weight):
        offset = offset.view(offset.shape[0], self.neighbors, 2, self.height, self.width) # N x neighbors x 2 x H x W
        offset = offset.transpose(2,3).transpose(3,4) # N x neighbors x H x W x 2
        offset[:,4,:,:] = 0
        offset = offset.contiguous().view(offset.shape[0], -1, offset.shape[3], 2) # N x neighbors*H x W x 2
        
        grid = self.original_grid + offset # 1 x neighbors*H x W x 2
        grid[:,:,:,0] = 2 * (grid[:,:,:,0] / (self.width - 1)) - 1
        grid[:,:,:,1] = 2 * (grid[:,:,:,1] / (self.height - 1)) - 1
        
        score = F.grid_sample(score, grid, mode="bilinear", align_corners=True, padding_mode="border") 
        score = score.view(score.shape[0], score.shape[1], self.neighbors, self.height, self.width) # N x 64 x neighbors x H x W
        score = torch.sum(score*weight.unsqueeze(1), dim=2)
        
        return score
    
    def _convLayer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def _scoreLayer(self, in_channels, depth_steps):
        return nn.Sequential(
            nn.Conv2d(in_channels, depth_steps, 3, padding=1),
            nn.Softmax(1)
        )
        
        

class Decoder(nn.Module):
    
    def __init__(self, depth_steps=64, depth_min = 0.1, depth_max = 3,
                 height = 256, width = 256, dilation=2, neighbors=9, blocks=[1,1,1], channels=[256,128,64]):
        super().__init__()
        
        self.height = height
        self.width = width
        self.dilation = dilation
        self.neighbors = neighbors
        
        depth_min /= depth_max
        depth_step = (1 - depth_min) / (depth_steps - 1)
        self.depth_slices = torch.tensor([depth_min + i * depth_step for i in range(depth_steps)]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to("cuda")
        
        self.upconv4 = self._upConvLayer(512, 512, 3)
        self.iconv4 = self._convLayer(1024, 512, 3)
        
        self.upconv3 = self._upConvLayer(512, 512, 3)
        self.iconv3 = self._convLayer(1024, 512, 3)
        self.score_conv3 = self._scoreLayer(512, depth_steps)
        
        self.upconv2 = self._upConvLayer(depth_steps, depth_steps, 3)
        self.upconv1 = self._upConvLayer(depth_steps, depth_steps, 3)
        self.upconv0 = self._upConvLayer(depth_steps, depth_steps, 3)
        
        
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
                    
        self.cost_aggregation2 = CostAggregation(256, channels[0], int(height/4), int(width/4), 
                                                 dilation=dilation, 
                                                 neighbors=neighbors, 
                                                 depth_steps=depth_steps, 
                                                 blocks=blocks[0])
        self.cost_aggregation1 = CostAggregation(128, channels[1], int(height/2), int(width/2),
                                                 dilation=dilation, 
                                                 neighbors=neighbors, 
                                                 depth_steps=depth_steps, 
                                                 blocks=blocks[1])
        self.cost_aggregation0 = CostAggregation(3, channels[2], height, width,
                                                 dilation=dilation, 
                                                 neighbors=neighbors, 
                                                 depth_steps=depth_steps, 
                                                 blocks=blocks[2])
                    
        
    def forward(self, convs, img=None):
        # convs = [conv1, conv2, conv3, conv4, conv5]
        
        upconv4 = self.upconv3(convs[4]) # 512 x H/16 x W/16
        iconv4 = self.iconv3(torch.cat((upconv4, convs[3]), 1)) # 512 x H/16 x W/16
        
        upconv3 = self.upconv3(iconv4) # 512 x H/8 x W/8
        iconv3 = self.iconv3(torch.cat((upconv3, convs[2]), 1)) # 512 x H/8 x W/8
        score3 = self.score_conv3(iconv3) # 64 x H/8 x W/8
        depth3 = torch.sum(score3*self.depth_slices, 1, keepdim=True)
        
        upconv2 = self.upconv2(score3) # 64 x H/4 x W/4
        score2 = self.cost_aggregation2(upconv2, convs[1])
        depth2 = torch.sum(score2*self.depth_slices, 1, keepdim=True)
        
        upconv1 = self.upconv1(score2) # 64 x W/2 x H/2
        score1 = self.cost_aggregation1(upconv1, convs[0])
        depth1 = torch.sum(score1*self.depth_slices, 1, keepdim=True)
        
        upconv0 = self.upconv0(score1)
        score0 = self.cost_aggregation0(upconv0, img[:,-1])
        depth0 = torch.sum(score0*self.depth_slices, 1, keepdim=True)
        
        return [depth0, depth1, depth2, depth3], score0
        
    
    def _upConvLayer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def _convLayer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def _scoreLayer(self, in_channels, depth_steps):
        return nn.Sequential(
            nn.Conv2d(in_channels, depth_steps, 3, padding=1),
            nn.Softmax(1)
        )