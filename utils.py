import os
import numpy as np
from PIL import Image
#import Imath, OpenEXR, array

import torch

from model import *

def get_model(args):
    enc = en.Encoder(depth_steps=args.depth_samples)
    dec = de.Decoder(depth_steps=args.depth_samples, depth_min=args.depth_min, depth_max=args.depth_max,
                     blocks=args.blocks, channels=args.channels)
        
    return enc, dec

def import_params(model, filename):
    params = torch.load(filename)
    model.load_state_dict(params)

#def read_image(filename):
#    img = Image.open(filename)
#    img = np.asarray(img)
#    img = img.astype(np.float32)
#    img = img / 255.
#    return img

#def read_depth(img_dpt_path):
#    # pt = Imath.PixelType(Imath.PixelType.HALF)  # FLOAT HALF
#    dpt_img = OpenEXR.InputFile(img_dpt_path)
#    dw = dpt_img.header()['dataWindow']
#    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
#    (r, g, b) = dpt_img.channels("RGB")
#    dpt = np.fromstring(r, dtype=np.float16)
#    dpt.shape = (size[1], size[0])
#    dpt = dpt.astype(np.float32)
#    return dpt

def crop_img(img, crop_pixel):
    # img: H x W
    height, width = img.shape
    cropped = img[crop_pixel:-crop_pixel, crop_pixel:-crop_pixel]
    return cropped

def sc_inv(pred, target, mask=None):
    if mask is None:
        z = np.log(pred+1e-8)-np.log(target+1e-8)
        sum_z = np.sum(z)
        sum_z_2 = np.sum(z*z)
        n = z.shape[0]*z.shape[1]
        loss = np.sqrt( sum_z_2 / n - sum_z*sum_z/(n*n) )
    else:
        z = mask * (np.log(pred+1e-8)-np.log(target+1e-8))
        sum_z = np.sum(z)
        sum_z_2 = np.sum(z*z)
        n = np.sum(mask)
        loss = np.sqrt( sum_z_2 / n - sum_z*sum_z/(n*n) )
    return loss

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        