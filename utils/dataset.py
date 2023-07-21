from glob import glob
import os
import numpy as np
import torch
import albumentations as A
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
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
class BratsDataset(Dataset):
    def __init__(self, img_dirs, modality_types, transform=None):
        self.img_dirs = img_dirs
        self.modality_types = modality_types
        self.transform = transform

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, index):
        imgs_path = self.img_dirs[index]
        # images = self.concat_imgs(imgs_path)
        images = np.array(Image.open(f'{imgs_path}/flair.jpg'))
        mask = np.array(Image.open(f'{imgs_path}/seg.jpg'))
        mask = (mask / 255 * 4).round()
        
        # print(mask.shape)
        # mask = self.preprocess_mask_labels(mask) #TODO unstack the masks
        

        if self.transform is not None:
            augmented = self.transform(image=images, mask=mask)
            images = augmented['image']
            mask = augmented['mask']

            # Reshape images
            # images = torch.unsqueeze(images, dim=1)
            
        mask = np.expand_dims(mask, axis=0)
        # mask = mask[0]
        # print(mask.shape)
        bboxes = self.get_bounding_box(mask)  # Obtain bounding boxes
        output = []
        for image in images:
          # print(image)
          # print('-----------')
        
          items = [image,mask,bboxes]
          output.append(items)


        for i in range(len(output)):
          # for j in range(len(output[i])):
            modality_1 = output[i]



        # Reshape images and masks
        # images = images.transpose(0, 3, 1, 2)  # (B, H, W, C) to (B, C, H, W)
        # mask = mask.transpose(2, 0, 1)  # (C, H, W) to (B, C, H, W)

        return images,mask,bboxes
    def concat_imgs(self, path: str):
        images = []
        for modality_type in self.modality_types:
            img = np.array(Image.open(f'{path}/{modality_type}.jpg'))
            img = self.normalize(img)
            images.append(img)
        # cat_img = np.concatenate(images, axis=-1)


        return np.array(images)

 

    def preprocess_mask_labels(self, mask: np.ndarray):
        mask_WT = np.zeros(mask.shape)
        mask_WT[mask == 2] = 1

        mask_TC = np.zeros(mask.shape)
        mask_TC[mask == 1] = 1

        mask_ET = np.zeros(mask.shape)
        mask_ET[mask == 3] = 1

        mask_BG = np.zeros(mask.shape)
        mask_BG[mask == 0] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET, mask_BG])
        return mask

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        if np.max(data) == 0:
            return data
        if (np.max(data) - data_min) == 0:
            return data / data_min

        return (data - data_min) / (np.max(data) - data_min)

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

trn_tfms = A.Compose([
    A.Resize(height=256, width=256),
    ToTensorV2(),
])

val_tfms = A.Compose([
    A.Resize(height=256, width=64),
    ToTensorV2(),
])









        
                
                



