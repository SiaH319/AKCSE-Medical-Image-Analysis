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


############################### LOAD DATA FUNCTIONS ###############################
def load_data_brain():
    # load image files from dataset-mask folder
    DATA_PATH = "dataset-mask"
    data_map = []
    for img_dir in glob.iglob(DATA_PATH + "/**/*.png", recursive=True):
        #print(img_dir)
        data_map.append(img_dir)

    # Make data frame
    df = pd.DataFrame({"path": data_map[:]})
    df_no_masks = df[~df['path'].str.contains("masks")] # extract all images that does not contain str "masks" in filepath
    df_masks = df[df['path'].str.contains("masks")] # extract all images that contains str "masks" in filepath
    #print(df_masks)
    #print(df_no_masks)

    # Data sorting
    no_masks = sorted(df_no_masks["path"].values, key=lambda string: int(string.split('/')[2].strip("img")))
    masks = sorted(df_masks["path"].values, key=lambda string: int(string.split('/')[2].strip("img")))

    # file number mismatch ckeck
    def file_mismatch_check(no_masks, masks):
        for i in range(min(len(no_masks), len(masks))):
            #print(no_masks[i], masks[i])

            # If there's a mismatch while mapping two files from masked and no-mask images
            if no_masks[i].split('/')[2].strip("img") != masks[i].split('/')[2].strip("img"):
                no_masks_num = int(no_masks[i].split('/')[2].strip("img"))
                no_masks_num_prev = int(no_masks[i-1].split('/')[2].strip("img"))
                masks_num = int(masks[i].split('/')[2].strip("img"))
                masks_num_prev = int(masks[i-1].split('/')[2].strip("img"))
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

    #print(final_df)
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
        resize = transforms.Resize(size=(512, 512))
        image, mask = resize(image), resize(mask)
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
        image, mask = F.normalize(image, (0.5), (0.5)), F.normalize(mask, (0.5), (0.5))
        return image, mask

    def original_transform(self, image, mask):
        # Convert to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # Resize
        resize = transforms.Resize(size=(512, 512))
        image, mask = resize(image), resize(mask)
        # Transform to tensor
        image, mask = F.to_tensor(image), F.to_tensor(mask)
        # Normalize tensor
        image, mask = F.normalize(image, (0.5), (0.5)), F.normalize(mask, (0.5), (0.5))
        return image, mask

    def __getitem__(self, index):
        image = plt.imread(self.image_paths[index])
        #image = Image.fromarray(image).convert('RGB')
        image = np.asarray(image).astype(np.uint8)
        #image = np.array(Image.fromarray((image * 255).astype(np.uint8)).resize((512, 512)).convert('RGB'))
        mask = plt.imread(self.target_paths[index])
        #mask = Image.fromarray(mask).convert('RGB')
        mask = np.asarray(mask).astype(np.uint8)
        #mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((512, 512)).convert('RGB'))
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

# let’s initialize the dataset class and prepare the data loader.

# Save images
def save_img(img_dataloader, masked_dataloader, index, augmented=None):

    def check_dir_exists(original_dir, masked_dir):
        if not os.path.exists(original_dir):
            os.makedirs(original_dir)
        if not os.path.exists(masked_dir):
            os.makedirs(masked_dir)

    if augmented:
        original_dir = 'dataset-final/img' + str(index) + augmented + '/original'
        masked_dir = 'dataset-final/img' + str(index) + augmented + '/mask'
        check_dir_exists(original_dir, masked_dir)
        path_name_img = original_dir + '/img' + str(index) + augmented + ".png"
        path_name_masked = masked_dir + '/img' + str(index) + augmented + ".png"
    else:
        original_dir = 'dataset-final/img' + str(index) + '/original'
        masked_dir = 'dataset-final/img' + str(index) + '/mask'
        check_dir_exists(original_dir, masked_dir)
        path_name_img = original_dir + '/img' + str(index) + ".png"
        path_name_masked = masked_dir + '/img' + str(index) + ".png"

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


original_datasets = PyTorchImageDataset(image_paths=no_mask_image_list, target_paths=mask_image_list)
augmented_datasets1 = PyTorchImageDataset(image_paths=no_mask_image_list, target_paths=mask_image_list, augmentation=True)
augmented_datasets2 = PyTorchImageDataset(image_paths=no_mask_image_list, target_paths=mask_image_list, augmentation=True)

for i in range(len(original_datasets)):
    # Display Images
    x1, y1 = augmented_datasets1[i]  # extract brain pic and its masked part
    augmented_img_dataloader1 = DataLoader(dataset=x1, batch_size=1)
    augmented_masked_dataloader1 = DataLoader(dataset=y1, batch_size=1)

    x2, y2 = augmented_datasets2[i]
    augmented_img_dataloader2 = DataLoader(dataset=x2, batch_size=1)
    augmented_masked_dataloader2 = DataLoader(dataset=y2, batch_size=1)

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
    original_img = iter(original_img_dataloader).next()
    # show_img(make_grid(original_img))
    original_masked = iter(original_masked_dataloader).next()
    # show_img(make_grid(original_masked))

    index = i + 1
    save_img(original_img, original_masked, index, "original")
    save_img(augmented_img1, augmented_masked1, index, "a1")
    save_img(augmented_img2, augmented_masked2, index, "a2")

'''
for i in range(len(original_datasets)):
    # Save Images
    #augmented_dataloader1 = DataLoader(dataset=augmented_datasets1[i], batch_size=2)
    augmented_no_mask = DataLoader(dataset=next(iter(augmented_datasets1[i])))
    augmented_mask = DataLoader(dataset=next(iter(augmented_datasets1[i])))
    save_img(augmented_no_mask, no_mask_image_list)
    save_img(augmented_mask, mask_image_list)
    #augmented_dataloader2 = DataLoader(dataset=augmented_datasets2[i], batch_size=2)
    #original_dataloader = DataLoader(dataset=original_datasets[i], batch_size=2)
'''
