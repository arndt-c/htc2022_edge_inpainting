import os 
import argparse
import torch
from pathlib import Path
from PIL import Image
from scipy.io import loadmat
import mat73
import numpy as np 

from unet import UNet_module
from classical_reconstructor import alt_grad_solver
from inpainting_util import extract_visible_edges

parser = argparse.ArgumentParser(description='Apply CT-reconstructor to every image in a directory.')

parser.add_argument('input_files')
parser.add_argument('output_files')
parser.add_argument('step', type=int)

step_to_angular_idx = {
    1: 181,
    2: 161,
    3: 141,
    4: 121,
    5: 101,
    6: 81,
    7: 61
}

step_to_angular_range = {
    1: 90,
    2: 80,
    3: 70,
    4: 60,
    5: 50,
    6: 40,
    7: 30
}

def load_image(path):
    try: 
        # for matplab 5.0 files
        ta_sinogram = loadmat(path, struct_as_record=False, simplify_cells=True)
    except:
        print("File could not be loaded using scipy.io.loadmat. Try mat73 instead.")
        ta_sinogram = mat73.loadmat(path)
        
    
    sinogram = ta_sinogram["CtDataLimited"]["sinogram"]
    angles = ta_sinogram["CtDataLimited"]["parameters"]["angles"]

    return sinogram, angles


def main(inputFolder, outputFolder, categoryNbr):
    
    # Load modules
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    angular_range = step_to_angular_range[categoryNbr]

    module_args = {'in_ch': 1, 'out_ch': 1, 'channels': [16,32,64,128,256,256], 'skip_channels': [None,16,32,64,128,256],
                   'kernel_size': 9, 'use_sigmoid': True, 'use_norm': True, 'num_groups': 8, 'normalize_input': False, 
                   'lr': 0.00002, 'resnet': False}
    
    model_inpaint =  UNet_module(**module_args)
    model_inpaint = model_inpaint.load_from_checkpoint(Path(str(base_dir) + '/network_weights/inpainting_bce_' + str(angular_range) + '.ckpt'), **module_args)
    model_inpaint.eval()

    model_seg = UNet_module(**module_args)
    model_seg = model_seg.load_from_checkpoint(Path(str(base_dir) + '/network_weights/segment_bce_' + str(angular_range) + '.ckpt'), **module_args)
    model_seg.eval()

    for f in os.listdir(inputFolder):
        sinogram, angles = load_image(os.path.join(inputFolder, f))
        
        start_angle = angles[0]
        stop_angle = angles[-1]
        
        sinogram = sinogram*2/sinogram.max()
        sinogram[sinogram<0]=0
        
        basic_rec, _, _, _, _ = alt_grad_solver(sinogram, start_angle=start_angle,
                                                stop_angle=stop_angle, alph=1,
                                                bet=1, lr=0.000004, steps=40)
        
        vis_edges = extract_visible_edges(basic_rec, start_angle+5, stop_angle-5, threshold=0.2)
        
        vis_edges = np.sqrt(vis_edges[0]**2 + vis_edges[1]**2)
        vis_edges[vis_edges>0] = 1
        vis_edges = torch.tensor(vis_edges, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            full_edges = model_inpaint(vis_edges)
            full_edges = torch.nn.functional.pad(torch.clip(full_edges,0, 1), (1,1,1,1))
        
            x_hat = model_seg(full_edges).squeeze()
            
        x_hat[x_hat<0.5] = 0
        x_hat[x_hat>=0.5] = 1
        
        im = Image.fromarray(x_hat.numpy()*255.).convert("L")

        os.makedirs(outputFolder, exist_ok=True)
        im.save(os.path.join(outputFolder,f.split(".")[0] + ".PNG"))

    return 0 


if __name__ == "__main__":

    args = parser.parse_args()


    main(args.input_files, args.output_files, args.step)