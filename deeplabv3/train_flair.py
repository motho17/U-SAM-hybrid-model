# import sys
# import os
# # x = sys.path.append(os.path.abspath(os.path.join('..','utils')))
# sys.path.append('../U-SAM-hybrid-model')
# from utils.dataset import BratsDataset
from dataset import shuffle_split
from dataset import BratsDataset
# from model import DeeplabV3Plus
import numpy as np
import matplotlib.pyplot as plt
import os
import albumentations as A
join = os.path.join
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from albumentations.pytorch import ToTensorV2
# import tensorflow as tf
import keras
from keras import backend as K
from torch.optim import Adam
import monai
from typing import Tuple
from glob import glob
# import tensorflow as tf
import torch.nn as nn
import albumentations as album
import segmentation_models_pytorch as smp
import time

imgs_dir='/Users/jojo/Downloads/content/BraTs'
all_dirs = glob (f'{imgs_dir}/*')
all_dirs = all_dirs[0:100]
# print(len(all_dirs))
all_dirs.sort()

train_dirs, valid_dirs = shuffle_split (all_dirs, seed = 1)

trn_tfms = A.Compose([
    A.Resize(height=128, width=128),
    ToTensorV2(),
])

val_tfms = A.Compose([
    A.Resize(height=256, width=64),
    ToTensorV2(),
])

modality_types = ['flair']
#prepare dataloader
train_ds = BratsDataset(train_dirs, modality_types,transform = trn_tfms)
train_dl = DataLoader(train_ds, batch_size = 4, shuffle = False,pin_memory = True)

#val dataset
valid_ds = BratsDataset(valid_dirs, modality_types,transform = trn_tfms)
valid_dl = DataLoader(valid_ds, batch_size = 4, shuffle = False,pin_memory = True)

# device = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = 1
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

# create segmentation model with pretrained encoder
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=CLASSES,
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

batch_size = 4
IMAGE_SIZE = 128
num_class = 1
TRAINING = True
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=num_class)
# define loss function
loss = smp.utils.losses.DiceLoss()

# define metrics
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

# define optimizer
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.00001),
])

# define learning rate scheduler (not used in this NB)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2, eta_min=5e-5,
)

# load best saved model checkpoint from previous commit (if present)
if os.path.exists('/Users/jojo/Documents/U-SAM-hybrid-model/deeplabv3/checkpoint/best_model.pth'):
    model = torch.load('/Users/jojo/Documents/U-SAM-hybrid-model/deeplabv3/checkpoint/best_model.pth', map_location=DEVICE)


train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)




if TRAINING:

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(0, EPOCHS):

        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_dl)
        valid_logs = valid_epoch.run(valid_dl)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

# unet =  DeeplabV3Plus(nInputChannels=3, n_classes=1, os=16, _print=False)
# unet = unet.to(device)
# optimizer = Adam(unet.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
# criterion = nn.NLLLoss().to(device)
# seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

# from __future__ import division
# from sklearn.metrics import confusion_matrix
# import numpy as np
# import six


# def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
#     pred_labels = pred_labels.flatten()
#     gt_labels = gt_labels.flatten()
#     confusion = confusion_matrix(gt_labels, pred_labels)
#     if len(confusion)!= 2:
#         confusion =  np.array([confusion[0][0],0,0,0]).reshape(2,2)
#     return confusion


# def calc_semantic_segmentation_iou(confusion):
#     intersection = np.diag(confusion)
#     union = np.sum(confusion, axis=1) + np.sum(confusion, axis=0) - np.diag(confusion)
#     Ciou = (intersection / (np.maximum(1.0, union)+  1e-10) )
#     mIoU = np.nanmean(Ciou)
#     return mIoU

# def calc_semantic_segmentation_dice(confusion):
#     a, b = confusion
#     tn, fp = a
#     fn, tp = b
#     return 2*tp/(2*tp + fn + fp+  1e-10)

# def eval_semantic_segmentation(pred_labels, gt_labels):
#     confusion = calc_semantic_segmentation_confusion(pred_labels, gt_labels)
#     mIoU = calc_semantic_segmentation_iou(confusion) 
#     pixel_accuracy = np.nanmean(np.diag(confusion) / (confusion.sum(axis=1)+1e-10))
#     class_accuracy = np.diag(confusion) / ( confusion.sum(axis=1) +  1e-10 )
#     dice = calc_semantic_segmentation_dice(confusion)

#     return {'miou': mIoU,
#             'pixel_accuracy': pixel_accuracy,
#             'class_accuracy': class_accuracy,
#             'dice': dice}
# # Note: Hyperparameter tuning could improve performance here
# net = unet.train()

# best = [0]
# Epoch = 40
# train_acc_epoch = []
# train_miou_epoch = []
# train_dice_epoch = []
# train_class_acc_epoch = []

# test_acc_epoch = []
# test_miou_epoch = []
# test_dice_epoch = []
# test_class_acc_epoch = []

# # 训练轮次
# for epoch in range(Epoch):
    
#     train_loss = 0
#     train_acc = 0
#     train_miou = 0
#     train_dice = 0
#     train_class_acc = 0
#     error = 0
#     print('Epoch is [{}/{}]'.format(epoch + 1, Epoch))

#     # 训练批次
#     for i, sample in enumerate(train_dl):
#         # 载入数据
#         img_data = sample['img'].to(device)
#         img_label = sample['label'].to(device)
#         # 训练
#         out = net(img_data)
#         out = F.log_softmax(out, dim=1)
#         loss = criterion(out, img_label)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
        
#         pre_label = out.max(dim=1)[1].data.cpu().numpy()
#         true_label = img_label.data.cpu().numpy()
#         eval_metrix = eval_semantic_segmentation(pre_label, true_label)
#         train_acc += eval_metrix['pixel_accuracy']
#         train_miou += eval_metrix['miou']
#         if len(eval_metrix['class_accuracy']) < num_class:
#             eval_metrix['class_accuracy'] = 0
#             train_class_acc = train_class_acc + eval_metrix['class_accuracy']
#             error += 1
#         else:
#             train_class_acc = train_class_acc + eval_metrix['class_accuracy']
#         train_dice += eval_metrix['dice']
       
#         if i%500 ==0:
#             print('|batch[{}/{}]|batch_loss:{:.9f}|'.format(i + 1, len(train_dl), loss.item()))

#     metric_description = '|Train Acc|: {:.5f}\n|Train dice|: {:.5f}\n|Train Mean IoU|: {:.5f}\n|Train_class_acc|: {:}'.format(
#         train_acc / len(train_dl),
#         train_dice / len(train_dl),
#         train_miou / len(train_dl),
#         train_class_acc / (len(train_dl)-error))
