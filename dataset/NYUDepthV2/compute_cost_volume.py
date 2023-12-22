import numpy as np
from PIL import Image
import os
import tqdm
from skimage.restoration import unsupervised_wiener, wiener
import skimage.filters
import argparse
import copy
import h5py
import scipy.ndimage

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("idx_file")
    parser.add_argument("--depth_samples", type=int, default=64)
    parser.add_argument("--scale", type=float, default=0.3)
    parser.add_argument("--max_coc", type=float, default=10.)
    parser.add_argument("--balance", type=float, default=0.01)
    parser.add_argument("--kernel_dir", default="../../disk_kernel")
    args = parser.parse_args()
    return args

def read_image(filename):
    img = Image.open(filename)
    img = np.asarray(img)
    img = img.astype(np.float32)
    img = img / 255.
    return img

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

def get_calib_params(fn, th=1e+3):
    focus_distances = []
    with open(fn, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if not i == len(lines) - 1:
                focus_distance = float(line.split()[0])
                aperture = float(line.split()[1])
                if focus_distance < th:
                    focus_distances.append((i, focus_distance))
            else:
                focal_length = float(line)
    return aperture, focal_length, focus_distances
                

if __name__ == "__main__":
    args = parse_args()
    
    out_dir = args.out_dir
    idx_file = args.idx_file
    scale = args.scale
    max_coc = args.max_coc
    balance = args.balance
    kernel_dir = args.kernel_dir
    
    root_dir = "./focal_stack"
    blur_dirs = ["blurred_rgb_2", "blurred_rgb_4", "blurred_rgb_8"]
    fn = "nyu_depth_v2_labeled.mat"
    f = h5py.File(fn)
    height, width, _ = f['images'][0].T.shape
    
    focus_distances = [2., 4., 8.] # m
    focal_length = 15 * 1e-3 # m
    f_number = 2.8
    baseline = 5.6 * 1e-6 # m / px
    
    depth_min = 0.1
    depth_max = 3
    steps = args.depth_samples
    depth_step = (depth_max - depth_min) / (steps-1)
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    with open(idx_file, "r") as f2:
        indices = f2.readlines()
        indices = [int(i) for i in indices]
        
    for sample_i in tqdm.tqdm(indices):
        sample_dir = os.path.join(out_dir, "{:06}".format(sample_i))
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
            
        depth_gt = (f['depths'][sample_i].T * scale).astype(np.float32) 
        input_imgs = np.zeros((len(focus_distances), height, width, 3))
        deblurred_imgs = np.zeros((steps, len(focus_distances), height, width, 3)) 
        for fi in range(len(focus_distances)):
            focus_distance = focus_distances[fi]
            fn = os.path.join(root_dir, blur_dirs[fi], "{}.png".format(sample_i+1))
            img = np.asarray(Image.open(fn)).astype(np.float32) / 255.
            input_imgs[fi] = img
            
            for di in range(steps):
                depth = depth_min + di * depth_step
                depth = depth / scale
                coc = compute_cocmap(depth, focus_distance, focal_length, f_number)
                coc = coc / baseline
                if coc > max_coc:
                    coc = max_coc
                kernel = get_kernel(coc, kernel_dir=kernel_dir)
                for ch in range(3):
                    ret = wiener(img[:,:,ch], kernel, balance=balance, clip=False)
                    deblurred_imgs[di, fi, :, :, ch] = ret
                        
        np.save(os.path.join(sample_dir, "imgs"), input_imgs.astype(np.float32))
        np.save(os.path.join(sample_dir, "depth"), depth_gt.astype(np.float32))
        
        cost_volume = compute_cost_volume(deblurred_imgs)
        np.save(os.path.join(sample_dir, "cost_volume"), cost_volume.astype(np.float32))