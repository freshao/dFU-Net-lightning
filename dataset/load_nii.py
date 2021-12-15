import re

import matplotlib
import torch
from matplotlib import pylab as plt
import nibabel as nib
import cv2
import numpy as np
import SimpleITK as sitk
import os

join = os.path.join
isdir = os.path.isdir
isfile = os.path.isfile
listdir = os.listdir
from PIL import Image
# from dataset.load_dcm import setDicomWinWidthWinCenter
from nibabel.viewers import OrthoSlicer3D

# img = nib.load(file)
# label = nib.load(label_file)
# width, height, queue = img.dataobj.shape
# # img = img.dataobj
# img = img.get_fdata()
# label = label.get_fdata()
from utils.file_and_folder_operations import maybe_mkdir_p

re_pattern = re.compile(r'(\d+)')
def load_pre_nii_gz(seg_file, label=False):
    flag = None
    if label:
        flag = sitk.sitkInt8
    else:
        flag = sitk.sitkInt16
    ct = sitk.ReadImage(seg_file, flag)
    # origin = ct.GetOrigin()
    # direction = ct.GetDirection()
    # xyz_thickness = ct.GetSpacing()
    ct_array = sitk.GetArrayFromImage(ct)
    # a = ct_array.min()
    # b = ct_array.max()
    # c = 1
    return ct_array


def load_nii(seg_file, label_file, tumor=False):
    ct = sitk.ReadImage(seg_file)
    # origin = ct.GetOrigin()
    # direction = ct.GetDirection()
    # xyz_thickness = ct.GetSpacing()
    ct_array = sitk.GetArrayFromImage(ct)
    seg_array = sitk.GetArrayFromImage(sitk.ReadImage(label_file))
    num = 1
    seg_bg = seg_array == 0
    seg_liver = seg_array >= 1
    if tumor:
        seg_tumor = seg_array == 2

        ct_bg = ct_array * seg_bg
        ct_tumor = ct_array * seg_tumor
        tumor_min = ct_tumor.min()
        tumor_max = ct_tumor.max()

        tumor_wide = tumor_max - tumor_min
        tumor_center = (tumor_max + tumor_min) / 2

        wl = window_transform(ct_array, tumor_wide, tumor_center, normal=True)
    else:

        ct_liver = ct_array * seg_liver
        liver_min = ct_liver.min()
        liver_max = ct_liver.max()
        liver_wide = liver_max - liver_min
        liver_center = (liver_max + liver_min) / 2
        wl = window_transform(ct_array, 250, 48, normal=True)
    return wl


# print(img)
# print(img.header['db_name'])


def window_transform(ct_array, windowWidth=250, windowCenter=45, normal=False):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg


def setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols):
    # if can not set 'writeable = True', make a new np.array()
    img_temp = np.array(img_data)
    img_temp.flags.writeable = True
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    for i in np.arange(rows):
        for j in np.arange(cols):
            img_temp[i, j] = int((img_temp[i, j] - min) * dFactor)

    min_index = img_temp < min
    img_temp[min_index] = 0
    max_index = img_temp > max
    img_temp[max_index] = 255

    return img_temp


# OrthoSlicer3D(img.dataobj).show()

# for i in range(0, queue, 10):
#     img_arr = img.dataobj[:, :, i]
#     plt.subplot(5, 4, num)
#     plt.imshow(img_arr, cmap='gray')
#     num += 1

def dcm2nii(dcms_path, nii_path, save_dir, type='dcm'):
    # if type == 'dcm':
    # 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
    # 2.将整合后的数据转为array，并获取dicom文件基本信息
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    # image_array = window_transform(image_array)
    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z
    # 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    sitk.WriteImage(image3, nii_path)


def png2nii(pngs_path, nii_path):
    # maybe_mkdir_p(nii_path)
    pngs_list = listdir(pngs_path)
    assert len(pngs_list) > 0
    re_pattern = re.compile(r'(\d+)')
    pngs_list.sort(key=lambda x: -int(re_pattern.findall(x)[-1]))
    arr_list = []
    for img in pngs_list:
        img_path = join(pngs_path, img)
        img = Image.open(img_path)
        img = np.array(img, dtype='uint8')
        arr_list.append(img)
    arr = np.array(arr_list)
    image = sitk.GetImageFromArray(arr)
    # image.SetSpacing(spacing)
    # image.SetDirection(direction)
    # image.SetOrigin(origin)
    sitk.WriteImage(image, nii_path)
    # a = sitk.ReadImage(nii_path, sitk.sitkInt8)
    # b = sitk.GetArrayFromImage(a)
    # c = b.max()
    # d = 1


def deal_chao(path, type, fold_prefix='train', slice=5):
    if type == 'CT':
        path = path + 'CT/'
        case_list = os.listdir(path)
        case_list.sort(key=lambda x: int(re_pattern.findall(x)[-1]))
        for case in case_list:
            dcm_data = path + case + '/DICOM_anon/'
            label_data = path + case + '/Ground/'
            save_path = path + case + '/CTdcm2niigz/volume-' + case + '.nii.gz'
            nii_save_dir = path + case + '/CTdcm2niigz/'
            res_save_dir = path + case + '/CTnii2res/'
            label_save_path = path + case + '/CTdcm2niigz/segmentation-' + case + '.nii.gz'

            dcm2nii(dcm_data, save_path, nii_save_dir)
            # nii2resimg(save_path, res_save_dir)
            png2nii(label_data, label_save_path)
            # 此处变换窗宽窗位， 不要重复变换
            nii2npz(file_path=label_save_path, save_path='/opt/data/private/data/chao_data/ann_dir/{}_slice_{}_{}'.format(fold_prefix, slice, 1), num_slice=slice, label=True)
            nii2npz(file_path=save_path, save_path='/opt/data/private/data/chao_data/img_dir/{}_slice_{}_{}'.format(fold_prefix, slice, 1), num_slice=slice)
    elif type == 'MR':
        path = path + 'MR/'
        for case in os.listdir(path):
            dcm_data = path + case + '/T2SPIR/DICOM_anon/'
            label_data = path + case + '/T2SPIR/Ground/'
            # save_path = path + case + '/T2SPIR/MRdcm2png/'
            nii_save_path = path + case + '/T2SPIR/MRdcm2nii/'
            nii_name = path + case + '/T2SPIR/MRdcm2nii/volume-{}.nii.gz'.format(case)
            # for img in os.listdir(dcm_data):
            label_save_path = path + case + '/T2SPIR/MRdcm2nii/segmentation-{}.nii.gz'.format(case)
            dcm2nii(dcm_data, nii_name,nii_save_path)
            png2nii(label_data, label_save_path)

            nii2npz(file_path=label_save_path, save_path='/opt/data/private/data/chao_data/ann_dir/mri_{}_slice_{}_{}'.format(fold_prefix, slice, 1), num_slice=slice, label=True)
            nii2npz(file_path=nii_name, save_path='/opt/data/private/data/chao_data/img_dir/mri_{}_slice_{}_{}'.format(fold_prefix, slice, 1), num_slice=slice)


def nii2resimg(file, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    seg_file = file
    ct = sitk.ReadImage(seg_file)
    ct_array = sitk.GetArrayFromImage(ct)
    tumor_wl = window_transform(ct_array, normal=True)
    for i in range(0, tumor_wl.shape[0] - 1):
        arr_min = tumor_wl.min()
        arr_max = tumor_wl.max()
        img_or = tumor_wl[i, :, :]
        img_or1 = tumor_wl[i + 1, :, :]
        img_arr3 = np.absolute(tumor_wl[i + 1, :, :] - tumor_wl[i, :, :])
        fin = (np.multiply(img_arr3, img_or1) + np.multiply(img_arr3, img_or)) * 255
        cv2.imwrite(save_path + "{}.png".format(i), fin.astype('uint8'))


def nii2npz(file_path, save_path, num_slice=1, label=False, prefix='', suffix='.nii.gz'):
    '''

    :param file_path: input path eg: /home/xx/data/segmentation-30.nii
    :param save_path: output path /home/xx/data/segnii2niigz
    :param num_slice: slice of each nii
    :return:
    '''
    flag = None
    if label:
        flag = sitk.sitkInt8
    else:
        flag = sitk.sitkInt16
    file = sitk.ReadImage(file_path, flag)
    tumor_wl = sitk.GetArrayFromImage(file)
    assert tumor_wl.shape[0] >= num_slice
    maybe_mkdir_p(save_path)
    if label == False:
        tumor_wl = window_transform(tumor_wl, normal=False)
    a = tumor_wl.max()
    b = tumor_wl.min()
    # len_step = 1 if (num_slice == 1) else num_slice // 2
    len_step = 1
    nii_shape = tumor_wl.shape
    spacing = file.GetSpacing()
    direction = file.GetDirection()
    origin = file.GetOrigin()
    name = file_path.split("/")[-1][:-4]
    for idx in range(0, nii_shape[0] - num_slice, len_step):
        image_array = tumor_wl[idx: idx + num_slice]
        image = sitk.GetImageFromArray(image_array)
        image.SetSpacing(spacing)
        image.SetDirection(direction)
        image.SetOrigin(origin)
        sitk.WriteImage(image, save_path + '/{}{}-{}-{}'.format(prefix, name, idx, num_slice) + suffix)
    # add last slice
    if num_slice > 1:
        image_array = tumor_wl[-num_slice:]
        image = sitk.GetImageFromArray(image_array)
        image.SetSpacing(spacing)
        image.SetDirection(direction)
        image.SetOrigin(origin)
        sitk.WriteImage(image, save_path + '/{}{}-{}-{}'.format(prefix, name, tumor_wl.shape[0] - num_slice,
                                                                num_slice) + suffix)


def lits2niigz(my_dataset, slice=5):
    '''
        │   ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    :param my_dataset: path of dataset
    :return:
    '''
    img_path = join(my_dataset, 'img_dir')
    label_path = join(my_dataset, 'ann_dir')

    # train data
    train_img_path = join(img_path, 'train')
    train_label_path = join(label_path, 'train')
    list_train_img_path = listdir(train_img_path)
    list_train_img_path.sort()
    list_train_label_path = listdir(train_label_path)
    list_train_label_path.sort()
    for img, label in zip(list_train_img_path, list_train_label_path):
        nii2npz(join(train_img_path, img), join(img_path, 'train_slice_{}_1'.format(slice)), num_slice=slice)
        nii2npz(join(train_label_path, label), join(label_path, 'train_slice_{}_1'.format(slice)), num_slice=slice,
                label=True)

    # val data
    val_img_path = join(img_path, 'val')
    val_label_path = join(label_path, 'val')
    list_val_img_path = listdir(val_img_path)
    list_val_img_path.sort()
    list_val_label_path = listdir(val_label_path)
    list_val_label_path.sort()
    for img, label in zip(list_val_img_path, list_val_label_path):
        nii2npz(join(val_img_path, img), join(img_path, 'val_slice_{}_1'.format(slice)), num_slice=slice)
        nii2npz(join(val_label_path, label), join(label_path, 'val_slice_{}_1'.format(slice)), num_slice=slice,
                label=True)


def test_a(file_path, save_path):
    file = sitk.ReadImage(file_path)
    arr_file = sitk.GetArrayFromImage(file)
    # assert arr_file.shape[0] >= num_slice
    maybe_mkdir_p(save_path)
    tumor_wl = window_transform(arr_file, normal=False)
    # seg_array = sitk.GetArrayFromImage(sitk.ReadImage(label_file))
    # seg_bg = seg_array == 0
    # seg_liver = seg_array >= 1
    # seg_tumor = seg_array == 2

    # ct_bg = ct_array * seg_bg
    # ct_tumor = ct_array * seg_tumor
    for i in range(0, tumor_wl.shape[0] - 1):
        # img_arr1 = img[:, :, i].transpose(1, 0)

        # shape = img_arr1.shape
        # tumor_wl = window_transform(ct_array, tumor_wide, tumor_center, normal=True)
        arr_min = tumor_wl.min()
        arr_max = tumor_wl.max()
        # img_arr1 = setDicomWinWidthWinCenter(img_arr1, winWidth, winCenter, shape[0], shape[1])
        # plt.subplot(1,4,num)
        # plt.imshow(liver_wl[i, :, :],cmap='gray')
        # plt.show()
        # plt.imshow(tumor_wl[i, :, :], cmap='gray')
        # plt.show()

        # img_arr2 = img[:, :, i + 1].transpose(1, 0)
        # img_arr2 = setDicomWinWidthWinCenter(img_arr2, winWidth, winCenter, shape[0], shape[1])
        # plt.imshow(img_arr2, cmap='gray')
        # plt.show()

        # plt.subplot(1, 3, num)
        # plt.imshow(img_arr1, cmap='gray')
        # plt.show()
        # plt.imshow(img_arr2, cmap='gray')
        # plt.show()
        # i += 100
        img_or = tumor_wl[i, :, :]
        img_or1 = tumor_wl[i + 1, :, :]
        # plt.imshow(tumor_wl[i, :, :], cmap='gray')
        # plt.show()

        # plt.imshow(tumor_wl[i, 100:228, 300:428], cmap='gray')
        # plt.show()
        img_arr3 = np.absolute(tumor_wl[i + 1, :, :] - tumor_wl[i, :, :])
        # label_arr3 = np.bitwise_and(seg_liver[i, :, :], seg_liver[i + 1, :, :])
        # label_arr3 = (seg_liver[i, :, :] & seg_liver[i + 1, :, :])

        # img_or_lab = img_or * label_arr3

        # index = np.where(img_or_lab > 0)
        # x_min = index[0].min()
        # x_max = index[0].max()
        # y_min = index[1].min()
        # y_max = index[1].max()
        # img_or_ = img_or[x_min-30: x_max+30, y_min-30: y_max+30] * 255
        # img_or_ = tumor_wl[i, 120:248, 100:228] * 255
        # img_arr3_ = img_arr3[x_min-30: x_max+30, y_min-30: y_max+30]
        # plt.imshow(img_or, cmap='gray')
        # plt.show()
        # plt.imshow(img_arr3, cmap='gray')
        # plt.show()
        # plt.imshow(np.multiply(img_arr3, img_or), cmap='gray')
        # plt.show()
        # plt.imshow(np.multiply(img_arr3, img_or1), cmap='gray')
        # plt.show()
        fin = (np.multiply(img_arr3, img_or1) + np.multiply(img_arr3, img_or)) * 255
        # plt.imshow(fin, cmap='gray')
        # plt.show()
        i
        # shape = img_arr3.shape
        # dFactor = 255.0 / (max - min)
        # for row in range(shape[0]):
        #     for col in range(shape[1]):
        #         img_arr3[row][col] = int((img_arr3[row][col] - min) * dFactor)

        # num += 1
        # plt.subplot(1,4,num)
        # plt.imshow(img_arr3, cmap='gray')
        # plt.show()
        #
        # plt.imshow(label_arr3, cmap='gray')
        # plt.show()

        # num += 1
        # plt.subplot(1,4,num)
        # label_arr4 = img_arr3 * (label_arr3)
        # max = label_arr4.max()
        # min = label_arr4.min()
        # mean = label_arr4[label_arr4>0].mean()
        # print("i:{}, max:{}".format(i, max))
        # print(mean)
        # plt.imshow(label_arr4, cmap='gray')
        # plt.show()

        # Grayimg = cv2.cvtColor(img_arr3, cv2.GRAY)
        cv2.imwrite(save_path + "{}.png".format(i), fin.astype('uint8'))
        # c = 1

def nii2result(file_path, save_path, num_slice=1, label=False, prefix='', suffix='.nii.gz'):
    '''

    :param file_path: input path eg: /home/xx/data/segmentation-30.nii
    :param save_path: output path /home/xx/data/segnii2niigz
    :param num_slice: slice of each nii
    :return:
    '''
    flag = None
    file_list = listdir(file_path)
    for case in file_list:
        # label_path = "/opt/data/private/data/LITS/LITS/weakly_data_30/segmentation-{}.nii".format(int(case))
        # label_file = sitk.ReadImage(label_path)
        # label_arr = sitk.GetArrayFromImage(label_file)
        case_path = join(file_path, case)
        case_list = listdir(case_path)
        if len(case_list) == 0:
            continue
        liver_list = []
        zero = np.zeros((512,512))
        # liver_list.append(zero)
        # liver_list.append(zero)
        case_list.sort(key=lambda x: int(re_pattern.findall(x)[-1]))
        for img in case_list:
            img_path = join(case_path, img)
            file = sitk.ReadImage(img_path)
            # a = sitk.GetArrayFromImage(file)
            liver_list.append(sitk.GetArrayFromImage(file)[0,...])
        # liver_list.append(zero)
        # liver_list.append(zero)

        arr = np.array(liver_list).astype('uint8')
        image = sitk.GetImageFromArray(arr)
        # image.SetSpacing(label_file.GetSpacing())
        # image.SetDirection(label_file.GetDirection())
        # image.SetOrigin(label_file.GetOrigin())

        sitk.WriteImage(image, save_path + '{}.nii.gz'.format(int(case)))
        a = 1



if __name__ == "__main__":
    # train_path = '/opt/data/private/data/sj/chao_cp/Train_Sets/'
    # test_path = '/opt/data/private/data/chao_data/Valid_Sets/'
    # deal_chao(train_path, 'CT', fold_prefix='train', slice=1)
    # deal_chao(test_path, 'CT', fold_prefix='val', slice=3)

    # file = '/data/LITS'
    file = '/opt/data/default/medical/data/LITS/LITS'
    lits2niigz(file, 1)

    # predict from nii.gz to nii


    # pred_file = '/opt/data/private/data/chao_data/result/ce-resimg-test97-chaos/pred/'
    # gt_file = '/opt/data/private/data/chao_data/result/ce-resimg-test97-chaos/gt/'
    # save_file = '/opt/data/private/data/chao_data/result/ce-resimg-test97-chaos/nii/'
    # save_gt_file = '/opt/data/private/data/chao_data/result/ce-resimg-test97-chaos/nii_gt/'
    # maybe_mkdir_p(save_file)
    # maybe_mkdir_p(save_gt_file)
    # nii2result(pred_file, save_path=save_file)
    # nii2result(gt_file, save_path=save_gt_file)

    # file = '/data/LITS'

    # file = '/opt/data/private/data/sliver/'
    # maybe_mkdir_p(file)
    # lits2niigz(file, 5)
    # lits2niigz(file, 9)