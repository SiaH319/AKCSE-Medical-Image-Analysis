import glob
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image


MASKED_DATA_PATH = os.path.abspath("../dataset/sagittal/mask")
IMG_DATA_PATH = os.path.abspath("../dataset/sagittal/original")

############################### LOAD DATA FUNCTIONS ###############################
def load_data_brain():
    # load image files from dataset-mask folder
    masked_data_map = []
    img_data_map = []
    for masked_dir in os.listdir(MASKED_DATA_PATH):
        masked_data_map.append(masked_dir)
    for img_dir in os.listdir(IMG_DATA_PATH):
        img_data_map.append(img_dir)

    # Make data frame
    df_no_masks = pd.DataFrame({"path": img_data_map[:]}) # extract all images that does not contain str "masks" in filepath
    df_masks = pd.DataFrame({"path": masked_data_map[:]}) # extract all images that contains str "masks" in filepath
    #print(df_no_masks)
    #print(df_masks)

    # Data sorting
    no_masks = sorted(df_no_masks["path"].values, key=lambda string: int(string.split('.')[0].strip("img")))
    masks = sorted(df_masks["path"].values, key=lambda string: int(string.split('.')[0].strip("_mask").strip("img")))
    #print(no_masks)
    #print(masks)

    # file number mismatch ckeck
    def file_mismatch_check(no_masks, masks):
        for i in range(min(len(no_masks), len(masks))):
            #print(no_masks[i], masks[i])

            # If there's a mismatch while mapping two files from masked and no-mask images
            if no_masks[i].split('.')[0].strip("img") != masks[i].split('.')[0].strip("_mask").strip("img"):
                no_masks_num = int(no_masks[i].split('.')[0].strip("img"))
                no_masks_num_prev = int(no_masks[i-1].split('.')[0].strip("img"))
                masks_num = int(masks[i].split('.')[0].strip("_mask").strip("img"))
                masks_num_prev = int(masks[i-1].split('.')[0].strip("_mask").strip("img"))
                # if has duplicated file
                if no_masks_num - no_masks_num_prev < 1:
                    pop = no_masks.pop(i)
                    print('---> Duplicated no_masks image found: no_masks img', pop, ' has popped from list, '
                                                                                     '\n\t\treplaced with ', no_masks[i])
                elif masks_num - masks_num_prev < 1:
                    pop = masks.pop(i)
                    print('---> Duplicated masks image found: masks masks', pop, ' has popped from list, '
                                                                                 '\n\t\treplaced with ', masks[i])
                # if has skipped file
                elif (no_masks_num - no_masks_num_prev > 1) or (masks_num - masks_num_prev > 1):
                    if no_masks_num > masks_num:
                        pop = masks.pop(i)
                        print('---> No mask img', i+1, ' are skipped, removing corresponding mask image: ', pop)
                    elif masks_num > no_masks_num:
                        no_masks.pop(i)
                        print('---> Mask img', i+1, ' are skipped, removing corresponding no-mask image: ', pop)
                i = i - 1  # now, check again
        return no_masks, masks

    no_masks, masks = file_mismatch_check(no_masks, masks)

    # Sorting check
    i = random.randint(0, len(no_masks)-1)
    print("===> Sorting Check (for debug purpose)")
    print("\tPath to the Image:", no_masks[i], "\n\tPath to the Mask:", masks[i])

    # Print final dataframe
    print('\n<< Output Final Dataframe >> (for debug purpose)\n',
          pd.DataFrame({"no_mask_path": no_masks,
                        "mask_path": masks}))

    return no_masks, masks
###################################################################################



###################### PROCESS DATA(AUGMENTATION) FUNCTIONS #######################
class PyTorchImageDataset(Dataset):
    def __init__(self, image_paths, target_paths, augmentation=False):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.augmentation = augmentation

    def transform(self, image, mask):
        # Convert to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # Resize
        #resize = transforms.Resize(size=(512, 512))
        #image, mask = resize(image), resize(mask)

        # Random horizontal flipping
        if random.random() > 0.5:
            image, mask = F.hflip(image), F.hflip(mask)
        # Random vertical flipping
        if random.random() > 0.5:
            image, mask = F.vflip(image), F.vflip(mask)
        # Random rotating
        if random.random() > 0.5:
            angle = random.choice([-90, -30, -15, 0, 15, 30, 90])
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)

        # Transform to tensor
        image, mask = F.to_tensor(image), F.to_tensor(mask)
        # Normalize tensor
        #image, mask = F.normalize(image, (0.5), (0.5), (0.5)), F.normalize(mask, (0.5), (0.5), (0.5))
        #mask = F.normalize(mask, (0.5), (0.5), (0.5))

        return image, mask

    def original_transform(self, image, mask):
        # Convert to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # Resize
        #resize = transforms.Resize(size=(512, 512))
        #image, mask = resize(image), resize(mask)

        # Transform to tensor
        image, mask = F.to_tensor(image), F.to_tensor(mask)

        # Normalize tensor
        #image, mask = F.normalize(image, (0.5), (0.5), (0.5)), F.normalize(mask, (0.5), (0.5), (0.5))

        return image, mask

    def __getitem__(self, index):
        image = plt.imread(self.image_paths[index])
        image = np.asarray(image*225).astype(np.uint8)

        mask = Image.open(self.target_paths[index])
        mask = np.asarray(mask)

        if (self.augmentation == True):
            x, y = self.transform(image, mask)
        else:
            x, y = self.original_transform(image, mask)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        #print(self.image_paths[index])
        #print(self.target_paths[index])
        return x, y

    def __len__(self):
        return len(self.image_paths)
###################################################################################



################################### MAIN INPUTS ###################################
# load all original and masked data
no_mask_image_list, mask_image_list = load_data_brain()  # returns a list of original(no-mask) image path and masked images path
no_mask_img_abs_path = []
mask_img_abs_path = []
for img in no_mask_image_list:
    no_masked_dir = IMG_DATA_PATH + "/" + img
    no_mask_img_abs_path.append(no_masked_dir)
for img in mask_image_list:
    masked_dir = MASKED_DATA_PATH + "/" + img
    mask_img_abs_path.append(masked_dir)

# let’s initialize the dataset class and prepare the data loader.

# Save images
def save_img(img_dataloader, masked_dataloader, index, augmented=None):

    if augmented:
        result_dir = '../dataset/sagittal/all-augmented'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        path_name_img = result_dir + '/img' + str(index) + "_" + augmented + ".png"
        path_name_masked = result_dir + '/img' + str(index) +  "_mask_" + augmented + ".png"
    else:
        result_dir = '../dataset/sagittal/all-augmented'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        path_name_img = result_dir + '/img' + str(index) + ".png"
        path_name_masked = result_dir + '/img' + str(index) + "_mask.png"

    save_image(img_dataloader, path_name_img)
    save_image(masked_dataloader, path_name_masked)


# Visualizing a Single Batch of Image
def show_img(img):
    plt.figure(figsize=(18, 15))
    # unnormalize
    img = img / 2 + 0.5
    npimg = img.numpy()
    npimg = np.clip(npimg, 0., 1.)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


original_datasets = PyTorchImageDataset(image_paths=no_mask_img_abs_path, target_paths=mask_img_abs_path)
augmented_datasets1 = PyTorchImageDataset(image_paths=no_mask_img_abs_path, target_paths=mask_img_abs_path, augmentation=True)
augmented_datasets2 = PyTorchImageDataset(image_paths=no_mask_img_abs_path, target_paths=mask_img_abs_path, augmentation=True)
augmented_datasets3 = PyTorchImageDataset(image_paths=no_mask_img_abs_path, target_paths=mask_img_abs_path, augmentation=True)
augmented_datasets4 = PyTorchImageDataset(image_paths=no_mask_img_abs_path, target_paths=mask_img_abs_path, augmentation=True)
augmented_datasets5 = PyTorchImageDataset(image_paths=no_mask_img_abs_path, target_paths=mask_img_abs_path, augmentation=True)
augmented_datasets6 = PyTorchImageDataset(image_paths=no_mask_img_abs_path, target_paths=mask_img_abs_path, augmentation=True)
augmented_datasets7 = PyTorchImageDataset(image_paths=no_mask_img_abs_path, target_paths=mask_img_abs_path, augmentation=True)
augmented_datasets8 = PyTorchImageDataset(image_paths=no_mask_img_abs_path, target_paths=mask_img_abs_path, augmentation=True)
augmented_datasets9 = PyTorchImageDataset(image_paths=no_mask_img_abs_path, target_paths=mask_img_abs_path, augmentation=True)
augmented_datasets10 = PyTorchImageDataset(image_paths=no_mask_img_abs_path, target_paths=mask_img_abs_path, augmentation=True)

for i in range(len(original_datasets)):
    # Display Images
    x1, y1 = augmented_datasets1[i]  # extract brain pic and its masked part
    augmented_img_dataloader1 = DataLoader(dataset=x1, batch_size=1)
    augmented_masked_dataloader1 = DataLoader(dataset=y1, batch_size=1)

    x2, y2 = augmented_datasets2[i]
    augmented_img_dataloader2 = DataLoader(dataset=x2, batch_size=1)
    augmented_masked_dataloader2 = DataLoader(dataset=y2, batch_size=1)

    x3, y3 = augmented_datasets3[i]
    augmented_img_dataloader3 = DataLoader(dataset=x3, batch_size=1)
    augmented_masked_dataloader3 = DataLoader(dataset=y3, batch_size=1)

    x4, y4 = augmented_datasets4[i]
    augmented_img_dataloader4 = DataLoader(dataset=x4, batch_size=1)
    augmented_masked_dataloader4 = DataLoader(dataset=y4, batch_size=1)

    x5, y5 = augmented_datasets5[i]
    augmented_img_dataloader5 = DataLoader(dataset=x5, batch_size=1)
    augmented_masked_dataloader5 = DataLoader(dataset=y5, batch_size=1)

    x6, y6 = augmented_datasets6[i]
    augmented_img_dataloader6 = DataLoader(dataset=x6, batch_size=1)
    augmented_masked_dataloader6 = DataLoader(dataset=y6, batch_size=1)

    x7, y7 = augmented_datasets7[i]
    augmented_img_dataloader7 = DataLoader(dataset=x7, batch_size=1)
    augmented_masked_dataloader7 = DataLoader(dataset=y7, batch_size=1)

    x8, y8 = augmented_datasets8[i]
    augmented_img_dataloader8 = DataLoader(dataset=x8, batch_size=1)
    augmented_masked_dataloader8 = DataLoader(dataset=y8, batch_size=1)

    x9, y9 = augmented_datasets9[i]
    augmented_img_dataloader9 = DataLoader(dataset=x9, batch_size=1)
    augmented_masked_dataloader9 = DataLoader(dataset=y9, batch_size=1)

    x10, y10 = augmented_datasets10[i]
    augmented_img_dataloader10 = DataLoader(dataset=x10, batch_size=1)
    augmented_masked_dataloader10 = DataLoader(dataset=y10, batch_size=1)

    xor, yor = original_datasets[i]
    original_img_dataloader = DataLoader(dataset=xor, batch_size=1)
    original_masked_dataloader = DataLoader(dataset=yor, batch_size=1)

    augmented_img1 = iter(augmented_img_dataloader1).next()
    # show_img(make_grid(augmented_img1))
    augmented_masked1 = iter(augmented_masked_dataloader1).next()
    # show_img(make_grid(augmented_masked1))
    augmented_img2 = iter(augmented_img_dataloader2).next()
    # show_img(make_grid(augmented_img2))
    augmented_masked2 = iter(augmented_masked_dataloader2).next()
    # show_img(make_grid(augmented_masked2))
    augmented_img3 = iter(augmented_img_dataloader3).next()
    augmented_masked3 = iter(augmented_masked_dataloader3).next()
    augmented_img4 = iter(augmented_img_dataloader4).next()
    augmented_masked4 = iter(augmented_masked_dataloader4).next()
    augmented_img5 = iter(augmented_img_dataloader5).next()
    augmented_masked5 = iter(augmented_masked_dataloader5).next()
    augmented_img6 = iter(augmented_img_dataloader6).next()
    augmented_masked6 = iter(augmented_masked_dataloader6).next()
    augmented_img7 = iter(augmented_img_dataloader7).next()
    augmented_masked7 = iter(augmented_masked_dataloader7).next()
    augmented_img8 = iter(augmented_img_dataloader8).next()
    augmented_masked8 = iter(augmented_masked_dataloader8).next()
    augmented_img9 = iter(augmented_img_dataloader9).next()
    augmented_masked9 = iter(augmented_masked_dataloader9).next()
    augmented_img10 = iter(augmented_img_dataloader10).next()
    augmented_masked10 = iter(augmented_masked_dataloader10).next()

    original_img = iter(original_img_dataloader).next()
    #show_img(make_grid(original_img))
    original_masked = iter(original_masked_dataloader).next()
    #show_img(make_grid(original_masked))

    index = i + 1
    save_img(original_img, original_masked, index)
    save_img(augmented_img1, augmented_masked1, index, "a1")
    save_img(augmented_img2, augmented_masked2, index, "a2")
    save_img(augmented_img3, augmented_masked3, index, "a3")
    save_img(augmented_img4, augmented_masked4, index, "a4")
    save_img(augmented_img5, augmented_masked5, index, "a5")
    save_img(augmented_img6, augmented_masked6, index, "a6")
    save_img(augmented_img7, augmented_masked7, index, "a7")
    save_img(augmented_img8, augmented_masked8, index, "a8")
    save_img(augmented_img9, augmented_masked9, index, "a9")
    save_img(augmented_img10, augmented_masked10, index, "a10")