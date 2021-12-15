import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
from matplotlib import pylab as plt

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


def calculate_metric_percase(pred, gt, i_batch, image):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    pred = pred.astype('uint8') * 255
    gt = gt.astype('uint8') * 255
    img = (image.astype('uint8') * 255)
    img = cv2.cvtColor(image[0], cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(image[0], cv2.COLOR_GRAY2BGR)
    # res = np.hstack([pred[0], gt[0]]) * 255
    # pred = pred * 255
    # gt = gt * 255
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(gt[0], cmap='gray')
    # plt.show()
    # ret, binary = cv2.threshold(pred[0], 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(pred[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    contours, hierarchy = cv2.findContours(gt[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    img = img + img2
    plt.imshow(img)
    plt.show()
    cv2.imwrite('/opt/data/private/data/chao_data/result/TransUnet/chao_contours/{}.png'.format(i_batch), img)
    if pred.sum() > 0 and gt.sum()>0:
        # dice = metric.binary.dc(pred, gt)
        specificity = metric.binary.specificity(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt)
        # precision = metric.binary.precision(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        # return dice, hd95
        return specificity, jaccard
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0