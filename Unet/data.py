from glob import glob
import os
import numpy as np
import torch
import albumentations as A
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from albumentations.pytorch import ToTensorV2

imgs_dir='/Users/jojo/Downloads/content/BraTs'
all_dirs = glob (f'{imgs_dir}/*')
all_dirs = all_dirs[0:100]
# print(len(all_dirs))
all_dirs.sort()
# a = glob('/Users/jojo/Downloads/content/BraTs')
# print(a[0:10])

#split the data into training and validation data
def shuffle_split (all_dirs, val_pct = 0.15, seed = 99):
    """ shuffling dataset with random state and split to train and valid """
    n_val = int (len (all_dirs) * val_pct)
    np.random.seed (seed)
    idx = np.random.permutation (len (all_dirs))
    all_dirs = np.array (all_dirs) [idx]
    
    return all_dirs [n_val:], all_dirs [:n_val]

train_dirs, valid_dirs = shuffle_split (all_dirs, seed = 1)
print(len(valid_dirs)), print(len(train_dirs))

#data preprocessing class
#@title brats 8
class BratsDataset (Dataset):
    def __init__ (self, img_dirs, modality_types, transform = None):
        self.img_dirs = img_dirs
        self.transform = transform

    def __len__ (self):
        return len (self.img_dirs)

    def __getitem__ (self, index):
        imgs_path = self.img_dirs [index]
        image = self.concat_imgs (imgs_path)
        mask = np.array (Image.open (f'{imgs_path}/seg.jpg'))
        mask = (mask / 255 * 4).round ()
        mask = self.preprocess_mask_labels(mask)
        
        image = image.transpose(2, 1, 0)  # (B, H, W, C) to (B, C, H, W)
        mask = mask.transpose(2, 1, 0)  # (C, H, W) to (B, C, H, W)
        # print(image.shape)
        # print(mask.shape)
        
        
        if self.transform is not None:
            augmented = self.transform(image = image, mask = mask)
            image = augmented ['image']
            mask = augmented ['mask']
        image = image.permute(0,1,2)  # (B, H, W, C) to (B, C, H, W)
        mask = mask.permute(2, 0, 1)  # (C, H, W) to (B, C, H, W)
        # print(image.shape)
        # print(mask.shape)
        bboxes = self.get_bounding_box(mask) 
        

        # return image.astype(float), mask.astype(float),bboxes
        # return image.float(), mask.float(),bboxes
        # return image.astype(np.float), mask.astype(np.float),bboxes
        return image,mask,bboxes
       

    def concat_imgs (self, path: str):
        types = []
        for modality_type in modality_types:
            img = np.array (Image.open (f'{path}/{modality_type}.jpg'))
            img = self.normalize(img)
            types.append (img)
#         cat_img = np.concatenate (types, axis = -1)

        return np.array(types)
    def get_bounding_box(self, mask):
      ground_truth_map = mask[0]# Extract the first channel of the mask

      # Find the indices of non-zero elements in the ground truth map
      y_indices, x_indices = np.where(ground_truth_map > 0)

      # Calculate the minimum and maximum values for x and y coordinates
      try:
        x_min = np.min(x_indices)
        y_min = np.min(y_indices)
        x_max = np.max(x_indices)
        y_max = np.max(y_indices)
        
      except ValueError:
        x_min = 0
        y_min = 0
        x_max = 0
        y_max = 0
      bboxes = np.array([x_min, y_min, x_max, y_max])

      return bboxes
    
    def preprocess_mask_labels(self, mask: np.ndarray):
        mask_WT = np.zeros(mask.shape)
        mask_WT[mask == 2] = 1
       
        mask_TC = np.zeros(mask.shape)
        mask_TC[mask == 1] = 1

        mask_ET = np.zeros(mask.shape)
        mask_ET[mask == 3] = 1
        
        mask_BG = np.zeros(mask.shape)
        mask_BG[mask == 0] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])#, mask_BG
        # mask = np.moveaxis(mask, (0, 1, 2), (0, 2, 1))
        return mask
    
    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        if np.max(data) == 0:
            return data
        if (np.max(data) - data_min) == 0:
            return data / data_min 
        
        return (data - data_min) / (np.max(data) - data_min)

trn_tfms = A.Compose (
[
    A.Resize (height = 256, width = 256),
        A.Rotate (limit = 35, p = 1.0),
        A.HorizontalFlip (p = 0.5),
        A.VerticalFlip (p = 0.1),
        # A.Normalize (mean=0.5, std=0.5, max_pixel_value = 255.0), 
        #     img = (img - mean * max_pixel_value) / (std * max_pixel_value)
        ToTensorV2 ()
            # The numpy HWC image is converted to pytorch CHW tensor. If the image is in HW format (grayscale image), it will be converted to pytorch HW tensor.
])

# val_tfms = A.Compose (
# [
#     A.Resize (height = 256, width = 256),
#     #     A.Normalize (0.5, 0.5, max_pixel_value = 255.0),
#     #     ToTensorV2 ()
# ])

modality_types = ['flair', 't1', 't1ce']#,'t2'

trn_tfms = A.Compose([
    A.Resize(height=256, width=256),
    ToTensorV2(),
])

val_tfms = A.Compose([
    A.Resize(height=256, width=256),
    ToTensorV2(),
])

train_ds = BratsDataset(train_dirs,modality_types,trn_tfms)


train_dl = DataLoader(train_ds, batch_size = 4, shuffle = False,pin_memory = True)

# from segment_anything.utils.transforms import ResizeLongestSide
import matplotlib.patches as patches
#confirm that the data loader is working as expected
for images,masks,bboxes in train_dl:
   print(f"{images.shape=}, {masks.shape=},{bboxes.shape=}")
   val, counts = np.unique(masks, return_counts=True)
   if(1 - (counts[0]/counts.sum())) > 0.01:
    print('yes')
   else:
    print('no')
  
   print(bboxes)
   break








        
                
                



