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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--focus_num", type=int, default=3)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--max_coc", type=float, default=20.)
    parser.add_argument("--balance", type=float, default=0.01)
    parser.add_argument("--kernel_dir", default="../../../disk_kernel")
    parser.add_argument("--max_focus_distance", type=float, default=200.)
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


def compute_coc(depth, focus_distance, focal_length, aperture):
    return aperture * np.absolute(focus_distance - depth) * focal_length / (depth * (focus_distance - focal_length))

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
    

def compute_cost(_deblurred_imgs, variance=1.):
    mean_f =  np.mean(_deblurred_imgs, axis=0) # height x width x 3
    mean_pw = scipy.ndimage.gaussian_filter(mean_f,variance) # height x width x 3
    diff2 = np.square(_deblurred_imgs - mean_pw[np.newaxis,:,:,:]) # fn x height x width x 3
    diff2_mean = np.mean(diff2, axis=0) # height x width x 3
    diff2_mean_pw = scipy.ndimage.gaussian_filter(diff2_mean,variance) # height x width x 3
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
                if th is None:
                    focus_distances.append((i, focus_distance))
                else:
                    if focus_distance < th:
                        focus_distances.append((i, focus_distance))
            else:
                focal_length = float(line)
    return aperture, focal_length, focus_distances


def get_out_dir(args):
        
    out_dir = args.out_dir + "_{}_{}_{}_{}".format(args.focus_num, args.depth_samples, args.depth_min, args.depth_max) + "_scale_" + str(args.scale)
        
    if args.max_focus_distance is not None:
        out_dir = out_dir + "_max_focus_distance_" + str(args.max_focus_distance) 
        
    return out_dir
                

if __name__ == "__main__":
    args = parse_args()
    
    out_dir = args.out_dir
    scale = args.scale
    focus_num = args.focus_num
    max_coc = args.max_coc
    balance = args.balance
    kernel_dir = args.kernel_dir
    max_focus_distance = args.max_focus_distance
    
    root_dir = "../data"
    depth_dir = "../data/depth"
    
    scenes = ["balls", "bottles", "fruits", "GT", "GTLarge", "GTSmall", "keyboard", "metal", "plants", "telephone", "window"]
    #scenes = ["keyboard"]
    img_dirs = [os.path.join(root_dir, "Aligned/Figure7/balls"),
               os.path.join(root_dir, "Aligned/Figure5/bottles"),
               os.path.join(root_dir, "Aligned/Figure5/fruits"),
               os.path.join(root_dir, "Aligned/Figure6/zeromotion"),
               os.path.join(root_dir, "Aligned/Figure6/largemotion"),
               os.path.join(root_dir, "Aligned/Figure6/smallmotion"),
               os.path.join(root_dir, "Aligned/Figure1/keyboard"),
               os.path.join(root_dir, "Aligned/Figure5/metals"),
               os.path.join(root_dir, "Aligned/Figure5/plants"),
               os.path.join(root_dir, "Aligned/Figure5/telephone"),
               os.path.join(root_dir, "Aligned/Figure5/window")
    ]
    
    
    depth_min = args.depth_min
    depth_max = args.depth_max
    steps = args.depth_samples
    depth_step = (depth_max - depth_min) / (steps-1)
    
    out_dir = get_out_dir(args)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    for i, (scene, img_dir) in enumerate(zip(scenes, img_dirs)):
        print(scene, i+1, "/", len(scenes))
        calib_path = os.path.join(root_dir, "calibration", scene, "calibrated.txt")
        aperture, focal_length, focus_distances = get_calib_params(calib_path, th=max_focus_distance)
        focus_step = int((len(focus_distances)-1) / (focus_num-1))
        
        img = read_image(os.path.join(img_dir, "a00.jpg"))
        height, width, _ = img.shape
        depth_gt = np.loadtxt(os.path.join(depth_dir, "{}/depth.csv".format(scene)), delimiter=",")
        depth_gt = depth_gt * scale
        input_imgs = np.zeros((focus_num, height, width, 3))
        deblurred_imgs = np.zeros((steps, focus_num, height, width, 3)) 
        
        focal_length = focal_length * scale
        
        idx_list = []
        focus_distance_list = []
        for fi in range(focus_num):
            idx = focus_distances[int(focus_step * fi)][0]
            focus_distance = focus_distances[int(focus_step * fi)][1]
            img = read_image(os.path.join(img_dir, "a{:02}.jpg".format(idx)))
            input_imgs[fi] = img
            
            idx_list.append(idx)
            focus_distance_list.append(focus_distance)
            
            focus_distance = focus_distance * scale
            
            for di in range(steps):
                depth = depth_min + di * depth_step
                coc = compute_coc(depth, focus_distance, focal_length, aperture)
                if coc > max_coc:
                    coc = max_coc
                kernel = get_kernel(coc, kernel_dir=kernel_dir)
                for ch in range(3):
                    ret = wiener(img[:,:,ch], kernel, balance=balance, clip=False)
                    deblurred_imgs[di, fi, :, :, ch] = ret
                
        
              
        out_scene_dir = os.path.join(out_dir, scene)
        if not os.path.exists(out_scene_dir):
            os.mkdir(out_scene_dir)
            
        np.save(os.path.join(out_scene_dir, "imgs"), input_imgs.astype(np.float32))
        np.save(os.path.join(out_scene_dir, "depth"), depth_gt.astype(np.float32))
        
        cost_volume = compute_cost_volume(deblurred_imgs)
        np.save(os.path.join(out_scene_dir, "cost_volume"), cost_volume.astype(np.float32))
        
        fn = os.path.join(out_scene_dir, "focus_distances.txt")
        with open(fn, "w") as f:
            for _i, _f in zip(idx_list, focus_distance_list):
                f.write(str(_i) + " " + str(_f) + "\n")
        
        
    
    
    