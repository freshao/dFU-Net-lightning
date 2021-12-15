import os
import argparse

import SimpleITK as sitk
import nibabel as nib
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--mhd_dir", type=str, default='/opt/data/private/data/sliver/label')
parser.add_argument("--nii_dir", type=str, default='/opt/data/private/data/sliver/nii')
parser.add_argument("--rot", type=int, default=1)
args = parser.parse_args()

if not os.path.exists(args.nii_dir):
    os.makedirs(args.nii_dir)

# TODO: 添加多进程
for fname in tqdm(os.listdir(args.mhd_dir)):
    if not fname.endswith(".mhd"):
        continue
    scan = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.mhd_dir, fname)))
    scan = scan.swapaxes(0, 1).swapaxes(1, 2)
    for _ in range(args.rot):
        scan = np.rot90(scan)
        scan = np.fliplr(scan)

    # TODO: 研究mhd/raw格式是否带有更多头文件信息
    new_scan = nib.Nifti1Image(scan, np.eye(4))
    nib.save(new_scan, os.path.join(args.nii_dir, fname.replace("mhd", "nii.gz")))