import torch
import torch.nn.functional as F
import os
from runpy import run_path
from skimage import img_as_ubyte
from PIL import Image
import numpy as np
import io

def save_img(filepath, img):
    img.save(filepath)

def save_gray_img(filepath, img):
    img.save(filepath)

def get_weights_and_parameters(task, parameters):
    if task == 'Motion_Deblurring':
        weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    elif task == 'Single_Image_Defocus_Deblurring':
        weights = os.path.join('Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
    elif task == 'Deraining':
        weights = os.path.join('Deraining', 'pretrained_models', 'deraining.pth')
    elif task == 'Real_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'real_denoising.pth')
        parameters['LayerNorm_type'] = 'BiasFree'
    elif task == 'Gaussian_Color_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_color_denoising_blind.pth')
        parameters['LayerNorm_type'] = 'BiasFree'
    elif task == 'Gaussian_Gray_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_gray_denoising_blind.pth')
        parameters['inp_channels'] = 1
        parameters['out_channels'] = 1
        parameters['LayerNorm_type'] = 'BiasFree'
    return weights, parameters

def restore_image(pil_img, task, result_dir='./demo/restored/', tile=None, tile_overlap=32):
    out_dir = os.path.join(result_dir, task)
    os.makedirs(out_dir, exist_ok=True)

    # Get model weights and parameters
    parameters = {'inp_channels': 3, 'out_channels': 3, 'dim': 48, 'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4, 'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66, 'bias': False, 'LayerNorm_type': 'WithBias', 'dual_pixel_task': False}
    weights, parameters = get_weights_and_parameters(task, parameters)

    load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
    model = load_arch['Restormer'](**parameters)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['params'])
    model.eval()

    img_multiple_of = 8

    print(f"\n ==> Running {task} with weights {weights}\n ")

    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        if task == 'Gaussian_Gray_Denoising':
            img = np.array(pil_img.convert('L'))
            img = np.expand_dims(img, axis=2)
        else:
            img = np.array(pil_img.convert('RGB'))

        input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)

        # Pad the input if not_multiple_of 8
        height, width = input_.shape[2], input_.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, ((width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        if tile is None:
            # Testing on the original resolution image
            restored = model(input_)
        else:
            # test the image tile by tile
            b, c, h, w = input_.shape
            tile = min(tile, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"
            tile_overlap = tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = torch.zeros(b, c, h, w).type_as(input_)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                    W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
            restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:, :, :height, :width]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        restored_img = Image.fromarray(restored)
        f = "restored_image"
        if task == 'Gaussian_Gray_Denoising':
            save_gray_img((os.path.join(out_dir, f + '.jpg')), restored_img)
        else:
            save_img((os.path.join(out_dir, f + '.jpg')), restored_img)

    print(f"\nRestored images are saved at {out_dir}")

    return restored_img