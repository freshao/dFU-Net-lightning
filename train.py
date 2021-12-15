import torch
# from medpy import metric
# from medpy.metric import dc
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger, TensorBoardLogger
import torch.nn as nn
# from sklearn.metrics import confusion_matrix
# from eval.confusion_matrix import base_confusion
# from torch.utils import model_zoo

# from network.ce_net import CEnet
# from network.vit_seg_modeling import VisionTransformer as ViT_seg
# from network.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torchmetrics.functional import confusion_matrix

from dataset.load_nii import nii2result
from evaluate import evaluate_nii
# from network.DenseUorigin import DenseUNet
from network.SmileNet import dFUNet, FitNet
# from network.umdnet import UMDnet
# from network.unet3 import UNet_3Plus
# from network.swin_transformer import Swin_uper
# from network.unet import Unet
from opt import get_opts
from utils import load_ckpt
from dataset.dataset import train_dataloader, ChaoDataset, SliceDataset

import pytorch_lightning as pl
from torch import optim
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import numpy as np
import os
# from flops_counter import get_model_complexity_info
# from torchstat import stat
from utils.file_and_folder_operations import maybe_mkdir_p
from utils.utils import DiceLoss, test_single_volume, join


class TrainSystem(pl.LightningModule):
    def __init__(self, param):
        super(TrainSystem, self).__init__()
        # self.hparams(param)
        self.my_hparams = param
        self.n_train = None
        self.n_val = None
        self.n_classes = self.my_hparams.num_classes
        self.n_channels = 1
        self.batch_size = self.my_hparams.batch

        # self.model = UNet_3Plus(1, 1)

        fitmodel = dFUNet(in_ch=3)
        in_ch = self.my_hparams.slice if self.my_hparams.slice > 1 else 1
        # print("input_channel: " + str(in_ch))
        # fitmodel = DenseUNet(n_classes=1, pretrained_encoder_uri='https://download.pytorch.org/models/densenet121-a639ec97.pth')
        # fitmodel = UMDnet(3,1)
        # fitmodel.encoder.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model = FitNet(fitmodel, in_ch=in_ch,
                            num_classes=self.n_classes)
        # self.model = fitmodel
        # self.model2 = Swin_uper()
        # self.model = CEnet(1, 1)
        # self.model = Unet(1, 1)
        # self.vit_name = 'R50-ViT-B_16'
        # self.img_size = 512
        # config_vit = CONFIGS_ViT_seg[self.vit_name]
        # self.model = ViT_seg(config_vit, img_size=self.img_size, num_classes=config_vit.n_classes).cuda()
        # self.model.load_state_dict(torch.load("/opt/data/private/data/lightning/model/TU_Chao512/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs6_512.cp/epoch_99.pth"))
        # stat(self.model, (1, 512, 512))
        self.criterion = nn.CrossEntropyLoss() if self.n_classes > 1 else nn.BCELoss()

        self.epoch_loss = 0
        self.val = {}
        self.iou_sum = 0
        self.dice_sum = 0
        self.sdu = self.scheduler()
        self.dice = DiceLoss(self.n_classes)
        # to unnormalize image for visualization
        self.unpreprocess = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((-1.5,), (1.0,))
            transforms.Normalize((0.5,), (0.5,))
        ])

        # model

        # device gpu number
        if self.my_hparams.num_gpus == 1:
            print('number of parameters : %.2f M' %
                  (sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6))

        # load model checkpoint path is provided
        if self.my_hparams.ckpt_path != '':
            print('Load model from', self.my_hparams.ckpt_path)
            load_ckpt(self.model, self.my_hparams.ckpt_path, self.my_hparams.prefixes_to_ignore)

    def forward(self, x):
        return self.model.forward(x)

    def on_train_epoch_start(self):
        self.epoch_loss = 0
        super(TrainSystem, self).on_train_epoch_start()

    def training_step(self, batch, batch_nb):
        input, label = batch.values()
        label[label > 1] = 1
        # # assert label.max() <= 1
        c = int(input.shape[1] // 2)
        # print(c)
        # input = input[:,::2,:,:]
        label = label[:, c, :, :]
        label = label.unsqueeze(1)
        output = self.forward(input)
        loss1 = self.criterion(output, label)
        eps = 1e-5
        m1 = output.contiguous().view(-1)
        m2 = label.contiguous().view(-1)
        inter = torch.sum(m1 * m2)  # 求交集，数据全部拉成一维 然后就点积
        union = torch.sum(m1) + torch.sum(m2) + eps  # 求并集，数据全部求和后相加，eps防止分母为0
        # iou = (inter.float() + eps) / (torch.sum(label | output) + eps)  # iou计算公式，交集比上并集
        dice = (2 * inter.float() + eps) / union.float()

        # loss2 = self.criterion(swin_out, label)

        # loss2 = self.criterion(y_res_hat, label)
        loss = loss1 + 1. - dice
        self.log_dict({'Loss/train_loss': loss, 'ACC/train_acc': dice}, on_step=True, on_epoch=True)
        # loss.backward()
        # loss = calc_loss(y_hat, label)

        # self.log('Loss/train', loss.item(), on_step=True, on_epoch=True)
        # self.logger.experiment.add_image('y_hat', y_hat, 0)

        # tensorboard_logs = {'train_loss': loss}

        self.epoch_loss += loss.item()
        return {'loss': loss}

    def on_train_epoch_end(self) -> None:
        for tag, value in self.model.named_parameters():
            tag = tag.replace('.', '/')
            # new
            self.logger.experiment.add_histogram('weights' + tag, value.data.cpu().numpy(), self.global_step)
            # self.logger.experiment.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), self.global_step)
        super(TrainSystem, self).on_train_epoch_end()

    def on_validation_epoch_start(self):
        self.val = {
            'DICE': [],
            'ACC': [],
            'PPV': [],
            'TPR': [],
            'TNR': [],
            'F1': [],
            'IOU': [],
            'LOSS': [],
        }
        self.times = 0

    def validation_step(self, batch, batch_idx):
        '''
            dice = (2 * tp) / (fp + 2 * tp + fn + eps)
            ACC = (TP + TN) / (TP + TN + FP + FN)
            PPV(Precision) = TP / (TP + FP)
            TPR(Sensitivity=Recall) = TP / (TP + FN)
            TNR(Specificity) = TN / (TN + FP)
            F1 = 2PR / (P + R)
            acc = (tp + tn) / (tp + tn + fp + fn + eps)
        :param batch:
        :param batch_idx:
        :return:
        '''
        input, label = batch.values()
        label[label > 1] = 1
        # # assert label.max() <= 1
        c = int(input.shape[1] // 2)
        label = label[:, c, :, :]
        label = label.unsqueeze(1)
        output = self.forward(input)
        loss = self.criterion(output, label)
        pred = output > 0.7

        ################################################################################################################
        if label.max() == 1 and pred.max() == 1:
            tn, fp, fn, tp = torch.flatten(
                confusion_matrix(pred.int().contiguous().view(-1), label.int().contiguous().view(-1), 2))
            dice = (2. * tp) / (fp + 2 * tp + fn)
            ppv = tp / (tp + fp)
            tpr = tp / (tp + fn)
            if ppv + tpr != 0:
                f1 = (2. * ppv * tpr) / (ppv + tpr)
                self.val['F1'].append(f1)

            iou = (tp / (tp + fp + fn))
            self.val['DICE'].append(dice)
            self.val['PPV'].append(ppv)
            self.val['TPR'].append(tpr)
            tnr = tn / (tn + fp)
            self.val['TNR'].append(tnr)

            # iou计算公式，交集比上并集
            if self.times < 15 and label.max() > 0 and tp > 0:
                self.times += 1
            iter_num = batch_idx
            image = input[0, c:c + 1, :, :]
            image = (image - image.min()) / (image.max() - image.min())
            self.logger.experiment.add_image('train/Image', image, iter_num)
            # outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            outputs = pred
            outputs = outputs.cpu().detach().numpy()
            self.logger.experiment.add_image('train/Prediction', outputs[0, ...], iter_num)
            labs = label[0, ...]
            self.logger.experiment.add_image('train/GroundTruth', labs, iter_num)
        elif label.max() == 0 and pred.max() == 0:
            iou = 1.
            dice = 1.
            self.val['DICE'].append(dice)
            # acc = (tp + tn) / (tp + tn + fp + fn + eps)
            # self.val['ACC'] = (self.val['ACC'] + acc) / 2
            self.val['PPV'].append(1)
            self.val['TPR'].append(1)
            self.val['TNR'].append(1)
            self.val['F1'].append(1)
            return {'/Loss/val_loss': self.val['LOSS']}
        else:
            dice = 0.0
            tn, fp, fn, tp = torch.flatten(
                confusion_matrix(output.contiguous().view(-1), label.long().contiguous().view(-1), 2, threshold=0.7))
            self.val['DICE'].append(dice)
            # acc = (tp + tn) / (tp + tn + fp + fn + eps)
            # self.val['ACC'] = (self.val['ACC'] + acc) / 2
            self.val['PPV'].append(0.0)
            self.val['TPR'].append(0.0)
            tnr = tn / (tn + fp)
            iou = tnr
            self.val['TNR'].append(tnr)
            self.val['F1'].append(0.0)

        self.log_dict({'Loss/val_loss': loss, 'ACC/val_acc': iou}, on_step=True, on_epoch=True)

        ################################################################################################################
        return {'/Loss/val_loss': self.val['LOSS']}

    def on_validation_epoch_end(self):
        # 应四舍五入
        self.times = 0
        percent = (
                          self.n_val + self.n_train) / self.my_hparams.batch * self.my_hparams.val_percent // self.my_hparams.num_gpus
        val_score = torch.mean(torch.Tensor(self.val['DICE'])).item()
        # self.log('iou', val_score, on_step=False, on_epoch=True)
        self.sdu.step(val_score)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True)

        # self.log('test/Dice/epoch', self.val['DICE'] * percent / self.n_val, on_step=False, on_epoch=True)
        # self.log('Atest/IOU', self.val['ACC'], on_step=False, on_epoch=True)
        self.log('Atest/Dice', torch.mean(torch.Tensor(self.val['DICE'])), on_step=False, on_epoch=True)
        # self.log('Atest/ACC', self.val['ACC'], on_step=False, on_epoch=True)
        self.log('Atest/PPV', torch.mean(torch.Tensor(self.val['PPV'])), on_step=False, on_epoch=True)
        self.log('Atest/TPR', torch.mean(torch.Tensor(self.val['TPR'])), on_step=False, on_epoch=True)
        self.log('Atest/TNR', torch.mean(torch.Tensor(self.val['TNR'])), on_step=False, on_epoch=True)
        self.log('Atest/F1', torch.mean(torch.Tensor(self.val['F1'])), on_step=False, on_epoch=True)

    def on_test_batch_start(self, batch, batch_idx: int, dataloader_idx: int) -> None:
        self.metric_list = 0.0
        super(TrainSystem, self).on_test_batch_start(batch, batch_idx, dataloader_idx)

    def test_step(self, batch, batch_nb):
        x, y = batch.values()
        test_single_volume(x, y, self.forward, batch_nb,
                           test_save_path=self.my_hparams.pred_imgs_save_dir + self.my_hparams.exp_name,
                           classes=self.n_classes, patch_size=[512, 512],
                           case=self.trainer.test_dataloaders[0].dataset.ids[batch_nb])

    def on_test_epoch_end(self) -> None:
        base_path = self.my_hparams.pred_imgs_save_dir + self.my_hparams.exp_name
        pred_file = join(base_path, 'pred/')
        gt_file = join(base_path, 'gt/')
        save_file = join(base_path, 'nii/')
        save_gt_file = join(base_path, 'nii_gt/')

        maybe_mkdir_p(save_file)
        maybe_mkdir_p(save_gt_file)
        nii2result(pred_file, save_path=save_file)
        nii2result(gt_file, save_path=save_gt_file)
        evaluate_nii(pred_path=save_file, target_path=save_gt_file)

    def __dataloader(self, imgs_dir=None, masks_dir=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((-1.5,), (1.0,))
            transforms.Normalize((0.5,), (0.5,))
            # transforms.Normalize(0.5, 0.5)
        ])
        target_transform = transforms.Compose([transforms.ToTensor()])
        # dataset = ChaoDataset(imgs_dir=self.my_hparams.imgs_dir, masks_dir=self.my_hparams.masks_dir, transform=transform,
        #                       target_transform=target_transform)

        # test_dataset = ChaoDataset(imgs_dir=self.my_hparams.test_imgs_dir, masks_dir=self.my_hparams.test_masks_dir,
        #                            transform=transform,
        #                            target_transform=target_transform)
        dataset = SliceDataset(imgs_dir=self.my_hparams.imgs_dir,
                               masks_dir=self.my_hparams.masks_dir,
                               transform=transform,
                               target_transform=target_transform)
        val_dataset = SliceDataset(imgs_dir=self.my_hparams.test_imgs_dir,
                                   masks_dir=self.my_hparams.test_masks_dir,
                                   transform=transform,
                                   target_transform=target_transform)
        test_dataset = SliceDataset(imgs_dir=self.my_hparams.test_imgs_dir,
                                    masks_dir=self.my_hparams.test_masks_dir,
                                    transform=transform,
                                    target_transform=target_transform, val=True)

        n_val = int(len(dataset) * self.my_hparams.val_percent)
        n_train = len(dataset) - n_val
        self.n_train = n_train
        self.n_val = n_val
        train, val = random_split(dataset, [n_train, n_val])

        if self.my_hparams.debug == True:
            num_workers = 0
        else:
            num_workers = 10
        train_loader = DataLoader(dataset=train, batch_size=self.batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset=val, batch_size=self.batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                                 pin_memory=True, drop_last=True)
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
        }

    # @pl.data_loader
    def train_dataloader(self):
        return self.__dataloader()['train']

    # @pl.data_loader
    def val_dataloader(self):
        return self.__dataloader()['val']

    def test_dataloader(self):
        return self.__dataloader()['test']

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.my_hparams.lr)

    def scheduler(self):
        return optim.lr_scheduler.ReduceLROnPlateau(self.configure_optimizers(), 'min' if self.n_classes > 1 else 'max',
                                                    patience=2)


if __name__ == '__main__':
    resimg_config = {
        "ser_lits_5_1": {
            "num_epochs": 100,
            "num_gpus": 8,
            "batch": 18,
            "slice": 5,
            "ckpt_path": '',
            "load_params": 'model',
            "imgs_dir": '/opt/data/default/medical/data/chao_data/img_dir/train_slice_5_1/',
            "masks_dir": '/opt/data/default/medical/data/chao_data/ann_dir/train_slice_5_1/',
            "test_imgs_dir": '/opt/data/default/medical/data/chao_data/img_dir/all_slice_5_1/',
            "test_masks_dir": '/opt/data/default/medical/data/chao_data/ann_dir/all_slice_5_1/',
            "pred_imgs_save_dir": '/opt/data/default/medical/data/LITS/result/',
            "val_percent": 0.2,
            "exp_name": 'ce-resimg-5-test97+LITS',
            "lr": 1e-3,
            "debug": False,
        }
    }
    # hparams = resimg_config['ser_lits_5_1']
    hparams = get_opts()
    systems = TrainSystem(hparams)
    checkpoint_callback = ModelCheckpoint(dirpath=f'../ckpts/{hparams.exp_name}',
                                          filename='{epoch:02d}',
                                          monitor='Loss/val_loss',  # 此处不带前置/
                                          mode='min',
                                          save_top_k=100)

    logger = TensorBoardLogger(save_dir="../logs",
                               name=hparams.exp_name,
                               )

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=checkpoint_callback,
                      # strategy='dp' if hparams.num_gpus > 1 else None,
                      strategy=hparams.strategy if hparams.num_gpus > 1 else None,
                      resume_from_checkpoint=hparams.ckpt_path if hparams.load_params == True else None,
                      logger=logger,
                      auto_scale_batch_size=True,

                      # early_stop_callback=None,
                      gpus=hparams.num_gpus,
                      enable_model_summary=True,
                      auto_lr_find=True,
                      check_val_every_n_epoch=1,
                      amp_backend='apex',
                      sync_batchnorm=True if hparams.num_gpus > 1 else False,
                      num_sanity_val_steps=0 if hparams.num_gpus > 1 else 5,
                      precision=16 if hparams.use_amp else 32,
                      amp_level='O2')

    if hparams.test == False:
        trainer.fit(systems)
    else:
        trainer.test(systems)
