import torch.nn as nn
import torch
import sys
import os
from torch.utils.data import Dataset, DataLoader
import monai
sys.path.append('../U-SAM-hybrid-model')
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from unet import UNet
from data import BratsDataset
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import os
import albumentations as A
from glob import glob
import torch.nn.functional as F

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
modality_types = ['flair', 't1', 't1ce']

#prepare dataloader
train_ds = BratsDataset(train_dirs, modality_types,transform = trn_tfms)


train_dl = DataLoader(train_ds, batch_size = 4, shuffle = False,pin_memory = True)

sam_checkpoint = "/Users/jojo/Downloads/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = torch.device("cpu")
# "cuda"

sam_model = sam_model_registry[model_type](num_classes = 3,checkpoint=sam_checkpoint)

sam_model.to(device=device)
unet_model = UNet()
unet_model.to(device=device)



class IntegratedModel(nn.Module):
    def __init__(self,sam,unet):
        super(IntegratedModel, self).__init__()

        self.sam_encoder = sam.image_encoder
        self.sam_decoder = sam.mask_decoder
        self.unet_encoder = unet
        self.alpha = nn.Parameter(torch.zeros(1))
        self.Sigmoid = nn.Sigmoid()
        # self.unet_decoder = unet_decoder

    def forward(self, x,box_torch):
        u_input = x
        unet_features = self.unet_encoder(u_input)
        # print(unet_features.shape)
       
        # input_images = self.preprocess(x)
        image_embedding = x
        sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
        trans_image = sam_trans.apply_image_torch(image_embedding)
        # image_embeddings = self.sam_encoder(input_images)
        sam_features = self.sam_encoder(trans_image)
        # print(sam_features.shape)
        # print('...............')
    
        gate = self.Sigmoid(self.alpha)
        # resized_gate = F.interpolate(gate, size=sam_features.shape[2:], mode='nearest')


        # Resize the UNet features to match the size of the SAM features
        unet_features = self.resize(unet_features, sam_features.size()[2:])
        unet_features = torch.mean(unet_features, dim=1, keepdim=True)

        # print(unet_features.shape)
        # print('2...............')


        # integrated_features = sam_features + unet_features
        integrated_features = gate*sam_features + (1-gate) * unet_features
        # integrated_features = resized_gate * sam_features + (1 - resized_gate) * unet_features

        


        # print(integrated_features.shape)



        # output = self.sam_decoder(integrated_features)
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        # predicted masks
        # batch_size = integrated_features.size(0)
        # dense_prompt_embeddings = dense_embeddings.repeat(batch_size, 1, 1, 1)
        output, _ = self.sam_decoder(
            image_embeddings=integrated_features.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, #dense_embeddings.repeat(image_embedding.size(0), 1, 1, 1), # (B, 256, 64, 64)
            multimask_output=True,
          )

        return output

    def resize(self, x, size):
        return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)

# Freeze the parameters of the SAM encoder
for param in sam_model.image_encoder.parameters():
    param.requires_grad = False

# Create an instance of the integrated model
model = IntegratedModel(sam_model, unet_model)
model.to(device=device)

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=1e-5, weight_decay=0),
])

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

num_epochs = 10
losses = []
for epoch in range(num_epochs):
  epoch_loss = 0
  for step,(image,mask, boxes) in enumerate(tqdm(train_dl)):
      val, counts = np.unique(mask, return_counts=True)
      if(1 - (counts[0]/counts.sum())) > 0.01:
      # inputs, labels = batch['image'], batch['label']
      # sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
      # trans_image = sam_trans.apply_image_torch(image)
        image = image.float()
        image = image.to(device)

        mask = mask.to(device)

        box_np = boxes.numpy()
        sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
        box = sam_trans.apply_boxes(box_np, (mask.shape[-2], mask.shape[-1]))
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)

        optimizer.zero_grad()


        # Forward pass
        outputs = model(image,box_torch)

        # Calculate the loss
        loss = seg_loss(outputs, mask)

        # Backward pass
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

  epoch_loss /= step
  losses.append(epoch_loss)
  print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
