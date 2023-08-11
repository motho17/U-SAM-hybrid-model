import sys
import os
# x = sys.path.append(os.path.abspath(os.path.join('..','utils')))
sys.path.append('../U-SAM-hybrid-model')
<<<<<<< Updated upstream
from utils.dataset import BratsDataset
=======
# from utils.dataset import BratsDataset
from data import BratsDataset
>>>>>>> Stashed changes
# from dataset import shuffle_split
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
from glob import glob

imgs_dir='/Users/jojo/Downloads/content/BraTs'
all_dirs = glob (f'{imgs_dir}/*')
all_dirs = all_dirs[0:100]
# print(len(all_dirs))
all_dirs.sort()
# a = glob('/Users/jojo/Downloads/content/BraTs')
# print(a[0:10])

def shuffle_split (all_dirs, val_pct = 0.15, seed = 99):
    """ shuffling dataset with random state and split to train and valid """
    n_val = int (len (all_dirs) * val_pct)
    np.random.seed (seed)
    idx = np.random.permutation (len (all_dirs))
    all_dirs = np.array (all_dirs) [idx]
    
    return all_dirs [n_val:], all_dirs [:n_val]
train_dirs, valid_dirs = shuffle_split (all_dirs, seed = 1)

trn_tfms = A.Compose([
    A.Resize(height=256, width=256),
    ToTensorV2(),
])

val_tfms = A.Compose([
    A.Resize(height=256, width=64),
    ToTensorV2(),
])

modality_types = ['flair']
#prepare dataloader
train_ds = BratsDataset(train_dirs, modality_types,transform = trn_tfms)


<<<<<<< Updated upstream
train_dl = DataLoader(train_ds, batch_size = 1, shuffle = False,pin_memory = True)
=======
train_dl = DataLoader(train_ds, batch_size = 4, shuffle = False,pin_memory = True)
>>>>>>> Stashed changes

from segment_anything.utils.transforms import ResizeLongestSide
import matplotlib.patches as patches
#confirm that the data loader is working as expected
<<<<<<< Updated upstream
for images,masks,bboxes in train_dl:
   print(f"{images.shape=}, {masks.shape=},{bboxes.shape=}")
   val, counts = np.unique(masks, return_counts=True)
   if(1 - (counts[0]/counts.sum())) > 0.01:
    print('yes')
   else:
    print('no')
  
   print(bboxes)
   break
=======
# for images,masks,bboxes in train_dl:
#    print(f"{images.shape=}, {masks.shape=},{bboxes.shape=}")
#    val, counts = np.unique(masks, return_counts=True)
#    if(1 - (counts[0]/counts.sum())) > 0.01:
#     print('yes')
#    else:
#     print('no')
  
#    print(bboxes)
#    break
>>>>>>> Stashed changes


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "/Users/jojo/Downloads/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = torch.device("cpu")
# "cuda"

<<<<<<< Updated upstream
sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
=======
sam_model = sam_model_registry[model_type](num_classes = 3,checkpoint=sam_checkpoint)

>>>>>>> Stashed changes
sam_model.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam_model)

from torch.optim import Adam
import monai
from typing import Tuple


# Note: Hyperparameter tuning could improve performance here
optimizer = Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

#train
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn as nn

import os
join = os.path.join
work_dir = './Users/jojo/Documents/U-SAM-hybrid-model/results'
task_name = 'Demo1'
model_save_path = join(work_dir, task_name)
os.makedirs(model_save_path, exist_ok=True)
#%% train
num_epochs = 10
losses = []
best_loss = 1e10
<<<<<<< Updated upstream
=======
num_classes = 3
if num_classes > 1:
    multimask_output = True
else:
    multimask_output = False


>>>>>>> Stashed changes
for epoch in range(num_epochs):
    epoch_loss = 0
    # train
    for step, (image_embedding, gt2D, boxes) in enumerate(tqdm(train_dl)):
<<<<<<< Updated upstream
      val, counts = np.unique(masks, return_counts=True)
      if(1 - (counts[0]/counts.sum())) > 0.01:
        image_embedding = image_embedding.repeat(1, 3, 1, 1)
=======
      val, counts = np.unique(gt2D, return_counts=True)
      if(1 - (counts[0]/counts.sum())) > 0.01:
        # image_embedding = image_embedding.repeat(1, 3, 1, 1)
>>>>>>> Stashed changes
        sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
        trans_image = sam_trans.apply_image_torch(image_embedding)
        trans_image = trans_image.float()
        trans_image = trans_image.to(device)
        
        # print(image_embedding.size(0))
        # do not compute gradients for image encoder and prompt encoder
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(trans_image)
            # convert box to 1024x1024 grid
            box_np = boxes.numpy()
            sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
            box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

            # print(image_embedding.shape)
            # print(gt2D.shape)
            # get prompt embeddings 
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
<<<<<<< Updated upstream
        # predicted masks
        batch_size = image_embedding.size(0)
        dense_prompt_embeddings = dense_embeddings.repeat(batch_size, 1, 1, 1)
=======

            # if num_classes > 1:
            #     multimask_output = True
            # else:
            #     multimask_output = False

            # mask_predictions = image_embedding.size * 3


        # predicted masks
        batch_size = image_embedding.size(0)
        dense_prompt_embeddings = dense_embeddings#.repeat(batch_size, 1, 1, 1)
        
>>>>>>> Stashed changes
        mask_predictions, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_prompt_embeddings, #dense_embeddings.repeat(image_embedding.size(0), 1, 1, 1), # (B, 256, 64, 64)
<<<<<<< Updated upstream
            multimask_output=False,
          )

=======
            multimask_output=True,
          )

        # mask_predictions = image_embedding.size * 4


        # if num_classes > 1:
        #     multimask_output = True
        # else:
        #     multimask_output = False

        # mask_predictions = image_embedding.size * 4

>>>>>>> Stashed changes
        loss = seg_loss(mask_predictions, gt2D.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    epoch_loss /= step
    losses.append(epoch_loss)
    print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
    # save the latest model checkpoint
    torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_latest.pth'))
    # save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_best.pth'))