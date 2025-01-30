# How to use
# python compute_cost_volume_scaling.py path_to_defocus_net_dataset path_to_output_dir 


import numpy as np
from PIL import Image
import os
import Imath, OpenEXR, array
import tqdm
from skimage.restoration import unsupervised_wiener, wiener
import skimage.filters
import argparse
import copy
import scipy.ndimage


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--sample0", type=int, default=400)
    parser.add_argument("--sample_num", type=int, default=500)
    parser.add_argument("--rate", type=float, default=4.)
    parser.add_argument("--max_coc", type=float, default=10.)
    parser.add_argument("--min_coc", type=float, default=1.)
    parser.add_argument("--sampling_type", type=int, default=1, help="1: depth, 2: disparity")
    parser.add_argument("--kernel_type", type=int, default=2, help="1: gaussian, 2: pillbox")
    parser.add_argument("--wiener_type", type=int, default=1, help="1: wiener, 2:unsupervised_wiener")
    parser.add_argument("--balance", type=float, default=0.01)
    parser.add_argument("--criterion", type=int, default=2, help="1: variance, 2: standard deviation, 3: max")
    parser.add_argument("--kernel_dir", default="./disk_kernel")
    parser.add_argument("--patch_wise", type=int, default=1)
    parser.add_argument("--depth_samples", type=int, default=64)
    #parser.add_argument("--normalize", type=int, default=0)
    args = parser.parse_args()
    return args

def read_image(filename):
    img = Image.open(filename)
    img = np.asarray(img)
    img = img.astype(np.float32)
    img = img / 255.
    return img

def read_depth(img_dpt_path):
    # pt = Imath.PixelType(Imath.PixelType.HALF)  # FLOAT HALF
    dpt_img = OpenEXR.InputFile(img_dpt_path)
    dw = dpt_img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    (r, g, b) = dpt_img.channels("RGB")
    dpt = np.fromstring(r, dtype=np.float16)
    dpt.shape = (size[1], size[0])
    dpt = dpt.astype(np.float32)
    return dpt

def compute_cocmap(depth, focus_distance, focal_length, f_number):
    cocmap = (focal_length*focal_length * np.absolute(depth - focus_distance)) / (depth * f_number * (focus_distance - focal_length))
    #cocmap = (np.absolute(depth - focus_distance) / depth) * (focal_length*focal_length / (f_number * (focus_distance - focal_length)))
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
    
def get_kernel(coc, kernel_type=1, kernel_radius=10, rate=4, kernel_dir=None):
    if kernel_type == 1: # gaussian
        kernel = np.zeros((kernel_radius*2+1, kernel_radius*2+1))
        kernel[kernel_radius, kernel_radius] = 1
        if coc > 1:
            kernel = skimage.filters.gaussian(kernel, sigma=coc/rate)
    elif kernel_type == 2: # pillbox
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
    
def compute_cost(_deblurred_imgs, criterion=1, patch_wised=1):# normalized=0):
    # _deblurred_imgs: 5 x height x width x 3
    if criterion == 3:
        pairs = get_pairs(5)
        _, height, width, _ = _deblurred_imgs.shape
        costs = np.zeros((len(pairs), height, width))
        for i, [i1, i2] in enumerate(pairs):
            costs[i] = np.sum(np.abs(_deblurred_imgs[i1] - _deblurred_imgs[i2]), axis=2)
        cost = np.max(costs, axis=0)
    else:
        if patch_wised == 0:
            mean_img = np.mean(_deblurred_imgs, axis=0, keepdims=True) # 1 x height x width x 3
            variance = np.mean(np.square(_deblurred_imgs - mean_img), axis=0) # height x width x 3
            if criterion == 1:
                cost = np.sum(variance, axis=2)
            if criterion == 2:
                cost = np.sum(np.sqrt(variance), axis=2)
        else:
            mean_f =  np.mean(_deblurred_imgs, axis=0) # height x width x 3
            mean_pw = scipy.ndimage.gaussian_filter(mean_f,1) # height x width x 3
            diff2 = np.square(_deblurred_imgs - mean_pw[np.newaxis,:,:,:]) # 5 x height x width x 3
            diff2_mean = np.mean(diff2, axis=0) # height x width x 3
            diff2_mean_pw = scipy.ndimage.gaussian_filter(diff2_mean,1) # height x width x 3
            cost = np.sum(np.sqrt(diff2_mean_pw), axis=2)
            #if normalized == 0:
            #    mean_f =  np.mean(_deblurred_imgs, axis=0) # height x width x 3
            #    mean_pw = scipy.ndimage.gaussian_filter(mean_f,1) # height x width x 3
            #    diff2 = np.square(_deblurred_imgs - mean_pw[np.newaxis,:,:,:]) # 5 x height x width x 3
            #    diff2_mean = np.mean(diff2, axis=0) # height x width x 3
            #    diff2_mean_pw = scipy.ndimage.gaussian_filter(diff2_mean,1) # height x width x 3
            #    cost = np.sum(np.sqrt(diff2_mean_pw), axis=2)
            #else:
            #    patch_size = 9
            #    patch_size_half = int(patch_size / 2)
            #    focus_num, height, width, _ = _deblurred_imgs.shape
            #    _deblurred_imgs_pad = np.zeros((focus_num, height+2*patch_size_half, width+2*patch_size_half, 3))
            #    for fi in range(focus_num):
            #        for ch in range(3):
            #            _deblurred_imgs[fi, :, :, ch] = np.pad(deblurred_imgs[fi, :, :, ch], ((patch_size_half, patch_size_half), (patch_size_half, patch_size_half)), mode="edge")
            #    _deblurred_imgs_stack = np.zeros((focus_num, height, width, 3, patch_size*patch_size))
            #    i = 0
            #    for v in range(-patch_size_half, patch_size_half+1):
            #        for u in range(-patch_size_half, patch_size_half+1):
            #            v0 = patch_size_half + v
            #            u0 = patch_size_half + u
            #            _deblurred_imgs_stack[:,:,:,:,i] = _deblurred_imgs_pad[:,v0:v0+height, u0:u0+width,:]
            #            i+=1
            #    min_val = np.min(_deblurred_imgs_stack, axis=4, keepdims=True)
            #    max_val = np.max(_deblurred_imgs_stack,, axis=4, keepdims=True)
            #    normalized_imgs = (_deblurred_imgs_stack - min_val) / (max_val - min_val)
                
            #    weights = np.zeros((patch_size, patch_size))
            #    weights[patch_size_half, patch_size_half] = 1
            #    weights = scipy.ndimage.gaussian_filter(weights,1)
            #    weights = np.reshape(weights, patch_size*patch_size)
                
            
            
    return cost

def get_pairs(a):
    return [(x, y) for x in range(a) for y in range(x+1,a)]
            

def compute_cost_volume(deblurred_imgs, criterion=1, patch_wised=0):#, normalized=0):
    # deblurred_imgs: 64 x 5 x height x width x 3
    steps, _, height, width, _ = deblurred_imgs.shape
    cost_volume = np.zeros((steps, height, width))
    for i in range(steps):
        cost_volume[i] = compute_cost(deblurred_imgs[i],criterion=criterion, patch_wised=patch_wised)#, normalized=normalized)
    return cost_volume

def get_out_dir(args, scales):
    if args.sampling_type == 1:
        sampling_type = "depth"
    elif args.sampling_type == 2:
        sampling_type = "disp"
    elif args.sampling_type == 3:
        sampling_type = "coc"
    
    if args.kernel_type == 1:
        kernel_type = "gaussian_{}".format(args.rate)
    elif args.kernel_type == 2:
        kernel_type = "pillbox"
        
    if args.wiener_type == 1:
        wiener_type = "wiener_{}".format(args.balance)
    elif args.kernel_type == 2:
        wiener_type = "unsup"
        
    out_dir = args.out_dir + "_{}_".format(args.depth_samples) + sampling_type + "_" + kernel_type + "_" + wiener_type
    
    if args.patch_wise == 1:
        out_dir = out_dir + "_patchwised"
        
    out_dir = out_dir + "_scaling_{}".format(len(scales))
    
    return out_dir

if __name__ == "__main__":
    args = parse_args()
    
    root_dir = args.root_dir
    sample0 = args.sample0
    sample_num = args.sample_num
    rate = args.rate
    max_coc = args.max_coc
    min_coc = args.min_coc
    sampling_type = args.sampling_type
    kernel_type = args.kernel_type
    wiener_type = args.wiener_type
    balance = args.balance
    criterion = args.criterion
    kernel_dir = args.kernel_dir
    patch_wise = args.patch_wise
    steps = args.depth_samples
    
    scales = [1.0]
    out_dir = get_out_dir(args, scales)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    width = 256
    height = 256
    focus_distances= [0.1, 0.15, 0.3, 0.7, 1.5]
    focal_length = 2.9 * 1e-3
    f_number = 1.
    sensor_width = 3.0999999046325684 * 1e-3
    baseline = sensor_width / width
    
    depth_min = 0.1
    depth_max = 3
    #steps = 64
    depth_step = (depth_max - depth_min) / (steps-1)
    
    disp_min = 1 / depth_max
    disp_max = 1 / depth_min
    disp_step = (disp_max - disp_min) / (steps-1)
    
    coc_step = (max_coc - min_coc) / (steps - 1)
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    for sample_i in tqdm.tqdm(range(sample0, sample0+sample_num)):
        for scale in scales:
            depth_gt = read_depth(os.path.join(root_dir, "{:06}Dpt.exr".format(sample_i))) * scale
            
            input_imgs = np.zeros((len(focus_distances), height, width, 3))
            deblurred_imgs = np.zeros((steps, len(focus_distances), height, width, 3)) # 64 x 5 x 256 x 256 x 3
            
            for fi in range(len(focus_distances)):
                focus_distance = focus_distances[fi]
                fn = os.path.join(root_dir, "{:06}_{:02}All.tif".format(sample_i, fi))
                img = read_image(fn)
                input_imgs[fi] = img
                
                for di in range(steps):
                    if sampling_type == 1:
                        depth = depth_min + di * depth_step
                        depth = depth / scale
                        coc = compute_cocmap(depth, focus_distance, focal_length, f_number)
                        coc = coc / baseline
                        
                    elif sampling_type == 2:
                        disp = disp_min + di * disp_step
                        depth = 1 / disp
                        coc = compute_cocmap(depth, focus_distance, focal_length, f_number)
                        coc = coc / baseline
                    if coc > max_coc:
                        coc = max_coc
                    kernel = get_kernel(coc, kernel_type=kernel_type, rate=rate, kernel_dir=kernel_dir)
                    
                    for ch in range(3):
                        if wiener_type == 1:
                            ret = wiener(img[:,:,ch], kernel, balance=balance, clip=False)
                            deblurred_imgs[di, fi, :, :, ch] = ret
                        elif wiener_type == 2:
                            ret = unsupervised_wiener(img[:,:,ch], kernel)
                            deblurred_imgs[di, fi, :, :, ch] = ret[0]
                   
            sample_dir = os.path.join(out_dir, "{:06}_{}".format(sample_i, scale))
            if not os.path.exists(sample_dir):
                os.mkdir(sample_dir) 
            np.save(os.path.join(sample_dir, "imgs"), input_imgs.astype(np.float32))
            #np.save(os.path.join(sample_dir, "deblurred_imgs"), deblurred_imgs.astype(np.float32))
            np.save(os.path.join(sample_dir, "depth"), depth_gt.astype(np.float32))
            
            cost_volume = compute_cost_volume(deblurred_imgs, criterion=criterion, patch_wised=patch_wise)#, normalized=normalize)
            np.save(os.path.join(sample_dir, "cost_volume"), cost_volume.astype(np.float32))
        
    
    