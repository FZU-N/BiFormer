import sys
sys.path.append('.')
import os
import glob
import time
from collections import OrderedDict
import argparse
import numpy as np
import torch
import cv2
from skimage.metrics import structural_similarity as skimage_ssim, peak_signal_noise_ratio as skimage_psnr
from torchvision.transforms.functional import to_tensor
import lpips
from natsort import natsorted

'''
Command: python measure_pair.py --dataset_name  LOL_v1   --model_name  BiFormer-T   --type png  --use_gpu

'''

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Measure:
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = lpips.LPIPS(net=net).to(self.device)

    def psnr(self, imgA, imgB):
        return skimage_psnr(imgA, imgB)

    def ssim(self, imgA, imgB, gray_scale=False):
        if gray_scale:
            imgA, imgB = cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY)
        score = skimage_ssim(imgA, imgB, channel_axis=None if gray_scale else 2)
        return score

    def lpips(self, imgA, imgB):
        tA, tB = self._to_tensor(imgA), self._to_tensor(imgB)
        return self.model.forward(tA.to(self.device), tB.to(self.device)).item()

    def mae(self, imgA, imgB):
        imgA, imgB = to_tensor(imgA).float(), to_tensor(imgB).float()
        return torch.mean(torch.abs(imgA - imgB)).item()

    def measure(self, imgA, imgB):
        return {
            'psnr': self.psnr(imgA, imgB),
            'ssim': self.ssim(imgA, imgB),
            'lpips': self.lpips(imgA, imgB),
            'mae': self.mae(imgA, imgB)
        }

    def to_4d(self, img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def _to_tensor(self, img):
        img = img.transpose(2, 0, 1)  # Convert HWC to CHW
        return torch.Tensor(self.to_4d(img)) / 127.5 - 1


def find_files(directory, file_extension):
    return natsorted(glob.glob(os.path.join(directory, f"*.{file_extension}")))


def format_result(result):
    return f"{result['psnr']:.4f}, {result['ssim']:.4f}, {result['lpips']:.4f}, {result['mae']:.4f}"


def measure_dirs(dirA, dirB, file_type, use_gpu, verbose=False):
    paths_A = find_files(dirA, file_type)
    paths_B = find_files(dirB, file_type)

    if len(paths_A) != len(paths_B):
        raise ValueError("Mismatched number of files in directories!")

    measure = Measure(use_gpu=use_gpu)
    results = []

    for pathA, pathB in zip(paths_A, paths_B):
        imgA, imgB = cv2.imread(pathA)[:, :, [2, 1, 0]],cv2.imread(pathB)[:, :, [2, 1, 0]]  # BGR to RGB
        result = measure.measure(imgA, imgB)
        results.append(result)
        if verbose:
            print(f"{os.path.basename(pathA)}, {os.path.basename(pathB)}, {format_result(result)}")

    averages = {key: np.mean([r[key] for r in results]) for key in results[0].keys()}
    print(f"Final Results: {format_result(averages)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure image quality metrics between two directories.")
    parser.add_argument('--dataset_name', type=str, default='LOL_v1', help='Dataset name (used for directory structure).')
    parser.add_argument('--model_name', type=str, default='BiFormer-T', help='Name of the model (used for directory structure).')
    parser.add_argument('--type', type=str, default='png', help='File type/extension to evaluate (e.g., png, jpg).')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for computation.')

    args = parser.parse_args()

    # Define dataset paths
    dataset_paths = {
        'LOL_v1': ('./Dataset/light_dataset/LOL_v1/eval15/high/', 'results_BiFormer/LOL_v1/'),
        'LOL_v2_real': ('./Dataset/light_dataset/LOL_v2/Real_captured/Test/high/', 'results_BiFormer/LOL_v2_real/'),
        'LOL_v2_sync': ('./Dataset/light_dataset/LOL_v2/Synthetic/Test/high/', 'results_BiFormer/LOL_v2_sync/'),
        'MIT_5K': ('./Dataset/light_dataset/MIT-Adobe-5K-512/test/high/', 'results_BiFormer/MIT_5K/'),
    }

    if args.dataset_name not in dataset_paths:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")

    dirA, dirB_base = dataset_paths[args.dataset_name]
    dirB = os.path.join(dirB_base, args.model_name)

    measure_dirs(dirA, dirB, args.type, args.use_gpu, verbose=True)
