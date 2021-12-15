import os
import re

# import nibabel as nib
from medpy.metric.binary import assd, dc, hd, jc, ravd
import numpy as np 
import json
import SimpleITK as sitk
# import argparse

# pred_path = 'dev_seg'
re_pattern = re.compile(r'(\d+)')
target_path = '/opt/data/default/medical/data/LITS/result/ground/'
pred_path = '/opt/data/default/medical/data/LITS/result/ce-resimg-test99-lits/nii_postprocessed_disconnect/'
# target_path = '/opt/data/private/data/chao_data/result/ce-resimg-test97-chaos/nii_gt/'
# pred_path = '/opt/data/private/data/chao_data/result/ce-resimg-test97-chaos/nii'

def evaluate_nii(pred_path, target_path):
    filenames = os.listdir(pred_path)

    label_name = os.listdir(target_path)
    spacing_dict_path = r'./LITS_spacing.json' # 用于提供空间信息以供距离类 metrics 使用

    filenames.sort(key=lambda x: int(re_pattern.findall(x)[-1]))
    label_name.sort(key=lambda x: int(re_pattern.findall(x)[-1]))

    with open(spacing_dict_path, 'r') as f:
        spacing_dict = json.load(f)

    IoU_per_case = []
    IoU_global = []
    Dice_per_case = []
    Dice_global = []
    RVD_per_case = []
    ASSD_per_case = []
    HD_per_case = []

    for name, label in zip(filenames, label_name):
        label_file = sitk.ReadImage(os.path.join(target_path, label))
        target = sitk.GetArrayFromImage(label_file)
        target[target > 1] = 1
        a = target.max()
        pred_file = sitk.ReadImage(os.path.join(pred_path, name))
        pred = sitk.GetArrayFromImage(pred_file)
        assert pred.shape == target.shape
        # pred = nib.load(os.path.join(pred_path, name)).get_fdata()
        # target = nib.load(os.path.join(target_path, label)).get_fdata()

        # target[target!=2] = 0
        # target[target==2] = 1

        # IoU_per_case.append(jc(pred, target))
        Dice_per_case.append(dc(pred, target))
        # RVD_per_case.append(ravd(pred, target))

        # ASSD_per_case.append(assd(pred, target, spacing_dict[str(name[13:-4])]))
        # HD_per_case.append(hd(pred, target, spacing_dict[str(name[13:-4])]))
        # ASSD_per_case.append(assd(pred, target, spacing_dict[str(name.split('.')[0])]))
        # HD_per_case.append(hd(pred, target,  spacing_dict[str(name.split('.')[0])]))
        print('=='*5+name+'=='*5)
        # print('IoU per case: {}'.format(np.mean(IoU_per_case)))
        print(name + '  Dice: ' + str(Dice_per_case[-1]))
        print('Dice per case: {}'.format(np.mean(Dice_per_case)))
        # print('RVD per case: {}'.format(np.mean(RVD_per_case)))
        # print('ASSD per case: {}'.format(np.mean(ASSD_per_case)))
        # print('MSSD per case: {}'.format(np.mean(HD_per_case)))
    print('=='*5+'summary'+'=='*5)
    # print('IoU per case: {}'.format(np.mean(IoU_per_case)))
    print('Dice per case: {}'.format(np.mean(Dice_per_case)))
    # print('RVD per case: {}'.format(np.mean(RVD_per_case)))
    # print('ASSD per case: {}'.format(np.mean(ASSD_per_case)))
    # print('MSSD per case: {}'.format(np.mean(HD_per_case)))

if __name__ == '__main__':
    pred_path = '/opt/data/default/medical/data/LITS/result/mce-resimg-5-test28-lits2chaos/nii'
    target_path = '/opt/data/default/medical/data/LITS/result/mce-resimg-5-test28-lits2chaos/nii_gt'
    evaluate_nii(pred_path, target_path)
