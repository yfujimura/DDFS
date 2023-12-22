import os
import glob
import numpy as np
from skimage.transform import resize

from torch.utils.data import Dataset, DataLoader


from utils import *

def crop_img(img, crop_pixel):
    # img: H x W
    height, width = img.shape
    cropped = img[crop_pixel:-crop_pixel, crop_pixel:-crop_pixel]
    return cropped

class MobileDepthDataset(Dataset):
    
    def __init__(self, dataset_dir, max_depth=3., stack_num=3):
        self.max_depth = max_depth
        self.stack_num = stack_num
        self.dataset_dir = dataset_dir
        self.scenes = ["balls", "bottles", "fruits", "GT", "window", "keyboard", "metal", "plants", "telephone"]
        self.rotation = [0, 3, 3, 0, 3, 0, 3, 3, 3]
        self.crop_pixel = 10
        
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        depth = np.load(os.path.join(self.dataset_dir, self.scenes[idx], "depth.npy")) / self.max_depth
        depth = np.rot90(depth,k=self.rotation[idx])
        depth = resize(crop_img(depth, self.crop_pixel), (256,256))
        
        _imgs = np.load(os.path.join(self.dataset_dir, self.scenes[idx], "imgs.npy"))
        imgs = np.zeros((self.stack_num,256,256,3))
        for i in range(self.stack_num):
            for ch in range(3):
                imgs[i,:,:,ch] = resize(crop_img(_imgs[i,:,:,ch], self.crop_pixel), (256,256))
                imgs[i,:,:,ch] = np.rot90(imgs[i,:,:,ch],k=self.rotation[idx])
        imgs = imgs.astype(np.float32)
        
        _cost_volume = np.load(os.path.join(self.dataset_dir, self.scenes[idx], "cost_volume.npy"))
        cost_volume = np.zeros((64,256,256))
        for i in range(64):
            cost_volume[i] = resize(crop_img(_cost_volume[i], self.crop_pixel), (256,256))
            cost_volume[i] = np.rot90(cost_volume[i],k=self.rotation[idx])
        cost_volume = cost_volume.astype(np.float32)
        
        output = [imgs, cost_volume, depth, self.scenes[idx]]
        
        return output
    
#class NYUDepthV2Dataset(Dataset):
#    
#    def __init__(self, dataset_dir, max_depth=3., stack_num=3):
#        self.max_depth = max_depth
#        self.stack_num = stack_num
#        self.dataset_dir = dataset_dir
#        self.dataset = sorted(os.listdir(dataset_dir))
#        
#    def __len__(self):
#        return len(self.dataset)
#    
#    def __getitem__(self, idx):
#        imgs = np.load(os.path.join(self.dataset_dir, self.dataset[idx], "imgs.npy"))
#        cost_volume = np.load(os.path.join(self.dataset_dir, self.dataset[idx], "cost_volume.npy"))
#        depth = np.load(os.path.join(self.dataset_dir, self.dataset[idx], "depth.npy")) / self.max_depth
#        
#        output = [imgs, cost_volume, depth]
#        
#        return output

class NYUDepthV2Dataset(Dataset):
    
    def __init__(self, dataset_dir, max_depth=3., stack_num=3, 
                 crop_size=256, crop_end_x=50, crop_end_y=50, crops_x=4, crops_y=3):
        self.max_depth = max_depth
        self.stack_num = stack_num
        self.dataset_dir = dataset_dir
        self.dataset = sorted(os.listdir(dataset_dir))
        self.crop_size = crop_size
        self.crop_end_x = crop_end_x
        self.crop_end_y = crop_end_y
        self.crops_x = crops_x
        self.crops_y = crops_y
        
        self.dx = (640 - 2*crop_end_x - crop_size) / (crops_x - 1)
        self.dy = (480 - 2*crop_end_y - crop_size) / (crops_y - 1)
        
    def __len__(self):
        return int(len(self.dataset) * self.crops_x * self.crops_y)
    
    def __getitem__(self, idx):
        original_idx = idx // (self.crops_x * self.crops_y)
        crop_idx = idx % (self.crops_x * self.crops_y)
        
        y0 = int(self.crop_end_y + (crop_idx // self.crops_x) * self.dy)
        x0 = int(self.crop_end_x + (crop_idx % self.crops_x) * self.dx)
        
        imgs = np.load(os.path.join(self.dataset_dir, self.dataset[original_idx], "imgs.npy"))[:, y0:y0+self.crop_size, x0:x0+self.crop_size, :]
        cost_volume = np.load(os.path.join(self.dataset_dir, self.dataset[original_idx], "cost_volume.npy"))[:, y0:y0+self.crop_size, x0:x0+self.crop_size]
        depth = np.load(os.path.join(self.dataset_dir, self.dataset[original_idx], "depth.npy"))[y0:y0+self.crop_size, x0:x0+self.crop_size] / self.max_depth
        
        output = [imgs, cost_volume, depth]
        
        return output
    
class NikonD5300Dataset(Dataset):
    
    def __init__(self, dataset_dir, max_depth=3., stack_num=3):
        self.max_depth = max_depth
        self.stack_num = stack_num
        self.dataset_dir = dataset_dir
        self.crop_pixel = 10
        self.scenes = ["desk", "trashcan", "sofa", "fridge", "printer", "display"]
        
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        _imgs = np.load(os.path.join(self.dataset_dir, self.scenes[idx], "imgs.npy"))
        imgs = np.zeros((self.stack_num,256,256,3))
        for i in range(self.stack_num):
            for ch in range(3):
                imgs[i,:,:,ch] = resize(crop_img(_imgs[i,:,:,ch], self.crop_pixel), (256,256))
        imgs = imgs.astype(np.float32)
        
        _cost_volume = np.load(os.path.join(self.dataset_dir, self.scenes[idx], "cost_volume.npy"))
        cost_volume = np.zeros((64,256,256))
        for i in range(64):
            cost_volume[i] = resize(crop_img(_cost_volume[i], self.crop_pixel), (256,256))
        cost_volume = cost_volume.astype(np.float32)
        
        output = [imgs, cost_volume, self.scenes[idx]]
        
        return output


    

        