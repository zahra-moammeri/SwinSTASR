# Modified to debug black output issue
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.nn import functional as F
from scipy.io import loadmat
from swinstasr.archs.swinstasr_arch import swinSTASR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/real/test', help='input test image folder with .mat files')
    parser.add_argument('--output', type=str, default='results/swinSTASR/test_mat', help='output folder')
    parser.add_argument('--task', type=str, default='swinSTASR', help='swinSTASR')
    parser.add_argument('--training_patch_size', type=int, default=60, help='training patch size')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 2')
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/net_g_latest.pth')
    parser.add_argument('--mat_key', type=str, default='LF', help='key in .mat file containing image data')
    args = parser.parse_args()

    # os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = define_model(args)
    model.eval()
    model = model.to(device)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*.mat')))):
        # read .mat file
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        
        base_path = os.path.join(args.output, imgname)  # New: Create subfolder for this image
        os.makedirs(base_path, exist_ok=True)  
        # Load .mat file and extract image data
        mat_data = loadmat(path)
        if args.mat_key not in mat_data:
            raise KeyError(f"Key '{args.mat_key}' not found in {path}. Available keys: {mat_data.keys()}")
        
        img_array = mat_data[args.mat_key].astype(np.float32)
        print(f"Raw 'LF' shape: {img_array.shape}, Min: {img_array.min()}, Max: {img_array.max()}")

        # Normalize based on data range (assuming [0, 255] or [0, 1])
        # if img_array.max() > 1.0:
        #     img_array /= 255.  # Normalize if in [0, 255]
        #     print("Normalized to [0, 1] from [0, 255]")
        # else:
        #     print("Assumed data is already in [0, 1]")


        #Normalize input to [0, 255] and convert to uint8
        # if img_array.max() <= 1.0:
        #     img_array = (img_array * 255.0).astype(np.uint8)
        # else:
        #     img_array = img_array.astype(np.uint8)

        # Process each view in the 5x5 grid
        for i in range(img_array.shape[0]):  # Angular dimension 1
            for j in range(img_array.shape[1]):  # Angular dimension 2
                img = img_array[i, j]  # Shape: (108, 156, 3)
                
                # Save raw input for debugging (first view only)
                # if i == 0 and j == 0:
                #     # raw_input = (img * 255.0).round().astype(np.uint8)
                #     raw_input = img
                #     cv2.imwrite(os.path.join(args.output, f'{imgname}_raw_input.png'), raw_input[:, :, [2, 1, 0]])  # BGR

                # Convert to torch tensor (C, H, W)
                # img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))) # BGR to RGB
                img = torch.from_numpy(np.transpose(img, (2, 0, 1))) 
                img = img.unsqueeze(0).to(device)

                # inference
                with torch.no_grad():
                    # pad input image to be a multiple of window_size
                    if 'swinSTASR' in args.task:
                        window_size = 12
                        _, _, h, w = img.size()
                        mod_pad_h = (h // window_size + 1) * window_size - h
                        mod_pad_w = (w // window_size + 1) * window_size - w
                        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h + mod_pad_h, :]
                        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w + mod_pad_w]

                        output = model(img)
                        output = output[..., :h * args.scale, :w * args.scale]
                
                # Debug model output
                print(f"Output shape: {output.shape}, Min: {output.min().item()}, Max: {output.max().item()}")

                # save image
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if output.ndim == 3:
                    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # RGB to BGR for cv2
                # output = ((output + 1)* 255 / 2).round().astype(np.uint8)
                output = cv2.normalize(output, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                output = output.astype(np.uint8)
                
                output_filename = os.path.join(base_path, f'View_{i}_{j}.bmp')
                cv2.imwrite(output_filename, output)
                print(f"Saved {output_filename}")


def define_model(args):
    if args.task == 'swinSTASR':
        model = swinSTASR(
            upscale=args.scale,
            in_chans=3,
            img_size=args.training_patch_size,
            window_size=12,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='STASR')

    loadnet = torch.load(args.model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    return model


if __name__ == '__main__':
    main()