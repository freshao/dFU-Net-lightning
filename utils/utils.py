import re

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
from matplotlib import pylab as plt
from torchvision import transforms

from utils.file_and_folder_operations import maybe_mkdir_p
import os

join = os.path.join
re_pattern = re.compile(r'(\d+)')
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt, i_batch, image, test_save_path):
    '''

    :param pred:
    :param gt:
    :param i_batch:
    :param image: value limited of (0-255) without Normalize
    :param test_save_path:
    :return:
    '''
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    pred = pred.astype('uint8') * 255
    gt = gt.astype('uint8') * 255
    # img = image.astype('uint8') * 255
    # from matplotlib import pylab as plt
    # plt.imshow(image, cmap='gray')
    # plt.show()
    # # img = image.astype('uint8')
    # plt.imshow(pred[0], cmap='gray')
    # plt.show()
    # float类型变换到0-1
    # ##############################   cv2       ###########################################
    img = cv2.cvtColor(image / 255, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(pred[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 1, 0), 1)
    contours, hierarchy = cv2.findContours(gt[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 1), 1)
    img = img * 255
    cv2.imwrite('{}/{}.png'.format(test_save_path, i_batch), img)
    print('{}/{}.png'.format(test_save_path, i_batch))
    # ###############################   Image    ###########################################
    # img2 = np.asarray(img)
    # contours, hierarchy = cv2.findContours(pred[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img2, contours, -1, (0, 1, 0), 1)
    #
    # contours, hierarchy = cv2.findContours(gt[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img2, contours, -1, (1, 0, 0), 1)
    #
    # save_img = Image.fromarray(np.uint8(img2 * 255))
    # save_img = save_img.convert('RGB')
    # save_img.save('/data/chao_data/result/TransUnet/chao_contours/{}.png'.format(i_batch))
    # plt.imshow(img)
    # plt.show()
    ############################################################################


def test_single_volume(image, label, net, i_batch, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    label[label>1] = 1
    c = int(image.shape[1]//2)
    image, label = image.squeeze(0).cpu().detach().numpy(), label[:,c,:,:].cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        slice = image[c, :, :]
        x, y = slice.shape[0], slice.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
        input = transforms.Normalize((0.5,), (0.5,))(torch.from_numpy(image)).unsqueeze(0).float().cuda()

        if True:
            outputs = net(input)
            # out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = outputs > 0.7
            out = out.cpu().detach().numpy()

            if x != patch_size[0] or y != patch_size[1]:
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                pred = out
            prediction[0] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(0).float().cuda()
        # out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        outputs = net(input)
        out = outputs > 0.7
        prediction = out.cpu().detach().squeeze(0).numpy()

    metric_list = []
    for i in range(0, classes):
        maybe_mkdir_p(join(test_save_path, 'img'))
        metric_list.append(calculate_metric_percase(prediction == i, label == i, i_batch, image[c,:,:], join(test_save_path, 'img')))

    case_id = int(re_pattern.findall(case)[2])
    if test_save_path is not None:
        case = i_batch
        for i in range(31):
            maybe_mkdir_p(test_save_path + '/pred/{}/'.format(i))
            maybe_mkdir_p(test_save_path + '/gt/{}/'.format(i))
        # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        # img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/pred/{}/'.format(case_id) + str(case) + "_pred.nii.gz")
        # sitk.WriteImage(img_itk, test_save_path + '/img/'+ str(case) + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/gt/{}/'.format(case_id) + str(case) + "_gt.nii.gz")

    return metric_list


# def test_single_volume_cp(image, label, net, i_batch, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
#     label[label>1] = 1
#     image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
#
#     if len(image.shape) == 3:
#         prediction = np.zeros_like(label)
#         for ind in range(image.shape[0]):
#             slice = image[ind, :, :]
#             x, y = slice.shape[0], slice.shape[1]
#             if x != patch_size[0] or y != patch_size[1]:
#                 slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
#             input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
#             # net.eval()
#             # with torch.no_grad():
#             if True:
#                 outputs, swin_out = net(input)
#                 # out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
#                 out = outputs > 0.7
#                 out = out.cpu().detach().numpy()
#                 if x != patch_size[0] or y != patch_size[1]:
#                     pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#                 else:
#                     pred = out
#                 prediction[ind] = pred
#     else:
#         input = torch.from_numpy(image).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         # input = torch.from_numpy(image).unsqueeze(
#         #     0).unsqueeze(0).float().cuda()
#         # net.eval()
#         # with torch.no_grad():
#         # out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
#         outputs, swin_out = net(input)
#         out = outputs > 0.7
#         prediction = out.cpu().detach().numpy()
#
#     metric_list = []
#     for i in range(0, classes):
#         maybe_mkdir_p(test_save_path)
#         metric_list.append(calculate_metric_percase(prediction == i, label == i, i_batch, image, test_save_path))
#
#     # if test_save_path is not None:
#     #     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#     #     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#     #     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#     #     img_itk.SetSpacing((1, 1, z_spacing))
#     #     prd_itk.SetSpacing((1, 1, z_spacing))
#     #     lab_itk.SetSpacing((1, 1, z_spacing))
#     #     sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
#     #     sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
#     #     sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
#     return metric_list