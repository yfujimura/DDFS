import os
import sys
import tqdm
import argparse
import matplotlib.pyplot as plt
import math
from statistics import mean

from torch import nn
from torch import optim

from utils import *
from loader import *
from model import *
#from loss import *

if os.path.exists("defocus-net"):
    sys.path.append('defocus-net/source/arch')
    from dofNet_arch1 import *

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--method", default="DDFS", help="DDFS or DEFOCUS_NET")
    parser.add_argument("--checkpoint_dir", default="checkpoint")
    parser.add_argument("--depth_min", type=float, default=0.1)
    parser.add_argument("--depth_max", type=float, default=3)
    parser.add_argument("--depth_samples", type=int, default=64)
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--blocks", type=int, nargs=3, default=[1,1,1])
    parser.add_argument("--channels", type=int, nargs=3, default=[256,128,8])
    parser.add_argument("--dataset", default="MOBILE_DEPTH", help="MOBILE_DEPTH or NYU_DEPTH_V2 or NIKON_D5300")
    parser.add_argument("--gpu", default="0")
    
    args = parser.parse_args()
    return args


def test(test_loader, output_dir, dataset, method, max_depth, en=None, de=None, net=None):
    if method == "DDFS":
        en.eval()
        de.eval()
    elif method == "DEFOCUS_NET":
        net.eval()
    
    maes = []
    rmses = []
    absrels = []
    sc_inv_errors = []
    
    for i, data in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        imgs = data[0] 
        cost_volume = data[1]
        
        if method == "DDFS":
            imgs = imgs.transpose(3,4).transpose(2,3).to("cuda")
            cost_volume = cost_volume.to("cuda")
            with torch.no_grad():
                convs = en(imgs, cost_volume)
                pred, score = de(convs,imgs)
            depth_pred = pred[0][0,0].to("cpu").detach().numpy() * max_depth
            
            
            
        elif method == "DEFOCUS_NET":
            # set focus distances
            x2_fcs = torch.ones(1,3,256,256)
            focus_distances = []
            if dataset == "MOBILE_DEPTH":
                scene = data[3][0]
                with open("./dataset/MobileDepth/python/test_3_64_0.1_3.0_scale_0.05_max_focus_distance_200.0/{}/focus_distances.txt".format(scene), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        focus_distance = float(line.split()[1]) 
                        focus_distances.append(focus_distance)
            elif dataset == "NYU_DEPTH_V2":
                focus_distances= [2,4,8]
                        
            for fi in range(3): 
                focus_distance = focus_distances[fi] / max(focus_distances)
                x2_fcs[:, fi:(fi + 1), :, :] = x2_fcs[:, fi:(fi + 1), :, :] * (focus_distance)
            x2_fcs = x2_fcs.float().to("cuda")
                
            # set images
            x = torch.zeros((1,256,256,0))
            for fi in range(3):
                img = imgs[:,fi,:,:,:]
                x = torch.cat((x, img), 3)
            x = x.transpose(2,3).transpose(1,2).to('cuda')
            
            # run model
            with torch.no_grad():
                out_step2, out = net(x, inp=3,k=3,x2=x2_fcs,flag_step2=True)
                depth_pred = out_step2[0,0].to("cpu").detach().numpy() * max_depth
                depth_pred[depth_pred<0] = 0
        
        
        # save results and compute errors
        if dataset == "MOBILE_DEPTH":
            depth = data[2]
            depth_gt = depth[0].to("cpu").detach().numpy() * max_depth
            
            if method == "DEFOCUS_NET":
                scale = np.median(depth_gt / (depth_pred+1e-8))
                depth_pred = depth_pred * scale
                
            scene = data[3][0]
            plt.imshow(depth_pred, clim=(0,max_depth), interpolation="nearest", cmap="jet_r")
            plt.savefig(os.path.join(output_dir, "{}_pred.png".format(scene)))
            plt.clf()
            
        if dataset == "NYU_DEPTH_V2":
            mask = np.ones_like(depth_pred)
            depth = data[2]
            depth_gt = depth[0].to("cpu").detach().numpy() * max_depth
            
            if method == "DEFOCUS_NET":
                scale = np.median(depth_gt / (depth_pred+1e-8))
                depth_pred = depth_pred * scale
                
            plt.imshow(depth_pred, clim=(0,max_depth*0.8), interpolation="nearest", cmap="jet_r")
            plt.savefig(os.path.join(output_dir, "{:06}_pred.png".format(i)))
            plt.clf()
            
            rmse = math.sqrt(np.sum(mask* np.absolute(depth_gt - depth_pred)**2) / np.sum(mask))
            mae = np.sum(mask * np.absolute(depth_gt - depth_pred)) / np.sum(mask)
            absrel = np.sum(mask * np.absolute(depth_gt - depth_pred)/(depth_gt+1e-8)) / np.sum(mask)
            sc_inv_error = sc_inv(depth_pred, depth_gt, mask)
            
            rmses.append(rmse)
            maes.append(mae)
            absrels.append(absrel)
            sc_inv_errors.append(sc_inv_error)
            
        if dataset == "NIKON_D5300":
            scene = data[2][0]
            
            plt.imshow(depth_pred, clim=(np.min(depth_pred)*0.9, np.max(depth_pred)*0.9), interpolation="nearest", cmap="jet_r")
            plt.rcParams["font.family"] = "Times New Roman" 
            plt.rcParams["font.size"] = 14
            
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            plt.tick_params(bottom=False, left=False, right=False, top=False)
            
            plt.colorbar()
            plt.savefig(os.path.join(output_dir, "{}_pred.png".format(scene)))
            plt.clf()
            
            
     
    if dataset == "NYU_DEPTH_V2":
        print("MAE:", mean(maes))
        print("RMSE:", mean(rmses))
        print("AbsRel:", mean(absrels))
        print("sc-inv:", mean(sc_inv_errors))
    
    

def main():
    args = parse_args()
    
    checkpoint_dir = args.checkpoint_dir
    depth_min = args.depth_min
    depth_max = args.depth_max
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    output_dir = "results"
    create_dir(output_dir)
    output_dir = os.path.join(output_dir, args.dataset)
    create_dir(output_dir)
    output_dir = os.path.join(output_dir, args.method)
    create_dir(output_dir)
    print("results are saved into {}".format(output_dir))
    
    
    # load trained model
    print("method: {}".format(args.method))
    print("load trained model:")
    en = None
    de = None
    net = None
    if args.method == "DDFS":
        en, de = get_model(args)
        en.to('cuda')
        de.to("cuda")
        import_params(en, os.path.join(args.checkpoint_dir, "en.pth"))
        import_params(de, os.path.join(args.checkpoint_dir, "de.pth"))
    elif args.method == "DEFOCUS_NET":
        net = AENet(in_dim=3, out_dim=1, num_filter=16, flag_step2=True)
        net.to("cuda")
        import_params(net, os.path.join(args.checkpoint_dir, "defocus-net.pth"))
    print("done.")
        
        
    # test
    print("test on {}".format(args.dataset))
    if args.dataset == "MOBILE_DEPTH":
        test_dir = "./dataset/MobileDepth/python/test_3_64_0.1_3.0_scale_0.05_max_focus_distance_200.0"
        test_dataset = MobileDepthDataset(test_dir, max_depth=depth_max)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        test(test_loader, output_dir, args.dataset, args.method, depth_max, en=en, de=de, net=net)
            
    elif args.dataset == "NYU_DEPTH_V2":
        test_dir = "./dataset/NYUDepthV2/test"
        test_dataset = NYUDepthV2Dataset(test_dir, max_depth=depth_max)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        test(test_loader, output_dir, args.dataset, args.method, depth_max, en=en, de=de, net=net)
        
    elif args.dataset == "NIKON_D5300":
        test_dir = "./dataset/NikonD5300/test_64_0.1_3.0"
        test_dataset = NikonD5300Dataset(test_dir, max_depth=depth_max)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        test(test_loader, output_dir, args.dataset, args.method, depth_max, en=en, de=de, net=net)
    
    
    

if __name__ == "__main__":
    main()

