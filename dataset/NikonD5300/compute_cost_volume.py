import numpy as np
from PIL import Image
import os
import tqdm
from skimage.restoration import unsupervised_wiener, wiener
import skimage.filters
import argparse
import copy
import sys
from skimage.transform import resize
import scipy.ndimage
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--max_coc", type=float, default=20.)
    parser.add_argument("--balance", type=float, default=0.01)
    parser.add_argument("--kernel_dir", default="../../../disk_kernel")
    parser.add_argument("--depth_min", type=float, default=0.1)
    parser.add_argument("--depth_max", type=float, default=3.)
    parser.add_argument("--depth_samples", type=int, default=64)
    args = parser.parse_args()
    return args

def read_image(filename):
    img = Image.open(filename)
    img = np.asarray(img)
    img = img.astype(np.float32)
    img = img / 255.
    return img

def get_focus_distance(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Focus Distance" in line:
                focus_distance = float(line.split(" ")[-2])
                
    return focus_distance

def compute_cocmap(depth, focus_distance, focal_length, f_number):
    cocmap = (focal_length*focal_length * np.absolute(depth - focus_distance)) / (depth * f_number * (focus_distance - focal_length))
    return cocmap

def get_near_values(ls, n):
    i1 = np.abs(np.asarray(ls) - n).argmin()
    v1 = ls[i1]
    if v1 == n:
        v2 = v1
    else:
        ls2 = copy.deepcopy(ls)
        ls2.remove(v1)
        i2 = np.abs(np.asarray(ls2) - n).argmin()
        v2 = ls2[i2]
    ret = [v1, v2]
    ret.sort()
    return ret
    
def get_kernel(coc, kernel_radius=10, kernel_dir=None):
    radius = coc / 2
    if radius < 0.5:
        kernel = np.zeros((kernel_radius*2+1, kernel_radius*2+1))
        kernel[kernel_radius, kernel_radius] = 1
    else:
        lookup_idx = [i/10 for i in range(5, 101, 1)]
        r1, r2 = get_near_values(lookup_idx, radius)
        if r1 == radius:
            kernel = np.loadtxt(os.path.join(kernel_dir, str(r1)+".txt"), delimiter=",")
        else:
            kernel1 = np.loadtxt(os.path.join(kernel_dir, str(r1)+".txt"), delimiter=",")
            kernel2 = np.loadtxt(os.path.join(kernel_dir, str(r2)+".txt"), delimiter=",")
            d1 = radius - r1
            d2 = r2 - radius
            w1 = d2 / (d1+d2)
            w2 = d1 / (d1+d2)
            kernel = w1 * kernel1 + w2 * kernel2
        
    return kernel
    
def compute_cost(_deblurred_imgs):
    mean_f =  np.mean(_deblurred_imgs, axis=0) # height x width x 3
    mean_pw = scipy.ndimage.gaussian_filter(mean_f,1) # height x width x 3
    diff2 = np.square(_deblurred_imgs - mean_pw[np.newaxis,:,:,:]) # fn x height x width x 3
    diff2_mean = np.mean(diff2, axis=0) # height x width x 3
    diff2_mean_pw = scipy.ndimage.gaussian_filter(diff2_mean,1) # height x width x 3
    cost = np.sum(np.sqrt(diff2_mean_pw), axis=2)
            
    return cost

def get_pairs(a):
    return [(x, y) for x in range(a) for y in range(x+1,a)]
            

def compute_cost_volume(deblurred_imgs):
    steps, _, height, width, _ = deblurred_imgs.shape
    cost_volume = np.zeros((steps, height, width))
    for i in range(steps):
        cost_volume[i] = compute_cost(deblurred_imgs[i])
    return cost_volume


def get_out_dir(args):
        
    out_dir = args.out_dir + "_{}_{}_{}".format(args.depth_samples, args.depth_min, args.depth_max)
        
    return out_dir
                

if __name__ == "__main__":
    args = parse_args()
    
    out_dir = args.out_dir
    max_coc = args.max_coc
    balance = args.balance
    kernel_dir = args.kernel_dir
    
    root_dir = "nikon_focal_stack"
    scenes = ["desk", "trashcan", "sofa", "fridge", "printer", "display"]
    
    width = 6000
    height = 4000
    focal_length = 30 * 1e-3
    f_number = 1.8
    baseline = 0.0039 * 1e-3
    resize_scale = 0.1
    
    width = int(width * resize_scale)
    height = int(height * resize_scale)
    
    depth_min = args.depth_min
    depth_max = args.depth_max
    steps = args.depth_samples
    depth_step = (depth_max - depth_min) / (steps-1)
    
    out_dir = get_out_dir(args)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    for j, scene in enumerate(scenes):
        print(scene, j+1, "/", len(scenes))
        input_imgs = np.zeros((3, height, width, 3))
        deblurred_imgs = np.zeros((steps, 3, height, width, 3))
        focus_distances = []
        
        files = sorted(glob.glob(os.path.join(root_dir, scene, "DSC_*.jpg")))
        
        for i in range(3):
            focus_distance = get_focus_distance(files[i] + ".txt")
            focus_distances.append(focus_distance)
            
            img = read_image(files[i])
            img = resize(img, (height,width))
            input_imgs[i] = img
            for di in range(steps):
                depth = depth_min + di * depth_step
                coc = compute_cocmap(depth, focus_distance, focal_length, f_number)
                coc = coc / baseline
                coc = coc*resize_scale
                if coc > max_coc:
                    coc = max_coc
                kernel = get_kernel(coc, kernel_dir=kernel_dir)
                
                for ch in range(3):
                    ret = wiener(img[:,:,ch], kernel, balance=balance, clip=False)
                    deblurred_imgs[di, i, :, :, ch] = ret
            
        out_scene_dir = os.path.join(out_dir, scene)
        if not os.path.exists(out_scene_dir):
            os.mkdir(out_scene_dir)
        np.save(os.path.join(out_scene_dir, "imgs"), input_imgs.astype(np.float32))
        cost_volume = compute_cost_volume(deblurred_imgs)
        np.save(os.path.join(out_scene_dir, "cost_volume"), cost_volume.astype(np.float32))
    