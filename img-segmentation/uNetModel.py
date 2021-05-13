import tensorflow as tf
import os
import sys
import random
import numpy as np

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

seed = 42
np.random.seed = seed

pretrained_weights = None
input_size = (512,512,1)
IMG_HEIGHT = input_size[0]
IMG_WIDTH = input_size[1]
IMG_CHANNELS = input_size[2]

PATH = os.getcwd().replace("/img-segmentation/unet_rhina3", "") + '/dataset/sagittal/all-augmented'
masks, images = [],[]

for file in os.listdir(PATH):
    if '_mask_a' in file.split('.')[0]:
        masks.append(os.path.join(PATH,file))
    elif '_a' in file.split('.')[0]:
        images.append(os.path.join(PATH,file)) 
    elif '_mask' in file.split('.')[0]:
        masks.append(os.path.join(PATH,file)) 
    elif 'img' in file.split('.')[0]:
        images.append(os.path.join(PATH,file)) 
    else:
        print("File is invalid.")

images = sorted(images, key=lambda string: string.split('/')[-1].strip("img").split('.')[0].strip('_a'))
images = sorted(images, key=lambda string: int(string.split('/')[-1].split('_')[0].strip('img').strip('.png')))
masks = sorted(masks, key=lambda string: string.split('/')[-1].strip("img").split('.')[0].strip('_mask_a'))
masks = sorted(masks, key=lambda string: int(string.split('/')[-1].split('_')[0].strip('img').strip('.png')))


#data = (images, masks)

# splitting to trainset and validation set
TRAIN_PORTION = 0.8
VAL_PORTION = 0.2

length = len(images)
train_length = int(length*TRAIN_PORTION)+1
val_length = int(length*VAL_PORTION)

'''
TRAIN_PATH = "add_train_path_here/" #need to add path
TEST_PATH = "add_test_path_here/" #need to add path
trainset = next(os.walk(TRAIN_PATH))[1] #tuple where first entry is train path
testset = next(os.walk(TEST_PATH))[1] #tuple where first entry is test path
'''

x_train, x_test, y_train, y_test = train_test_split(images, masks, test_size=0.33, random_state=4)
#train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=10,shuffle=True)
#val_loader = torch.utils.data.DataLoader(dataset=valset, batch_size=10)

X_train = np.zeros((len(x_train), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
Y_train = np.zeros((len(y_train), IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)

print("Resizing training images and masks")
for path in tqdm(enumerate(X_train, Y_train), total=len(x_train)):
    img = imread(x_train)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode = "constant", preserve_range = True)
    X_train[n] = img # Fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)

    for mask_file in tqdm(enumerate(X_train, Y_train), total=len(x_train))]:
        mask_ = imread(mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode = "constant", preserve_range = True), axis = -1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

#test images
X_test = np.zeros((len(valset), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print("Resizing test images")
for n, id_ in tqdm(enumerate(valset), total=len(valset)):
    path= TEST_PATH + id_
    img = imread(path + "/images/" + id_ + ".png")[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode = "constant", preserve_range = True)
    X_test[n] = img

print("Done!")


###########

#Just testing to make sure the model is okay

image_x=random.randint(0,len(train_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()

###########



#Build the model
inputs = tf.keras.layers.Input(input_size)
adj_inputs = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(adj_inputs)
conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = tf.keras.layers.Dropout(0.5)(conv4)
pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = tf.keras.layers.Dropout(0.5)(conv5)

up6 = tf.keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
merge6 = tf.keras.layers.concatenate([drop4,up6], axis = 3)
conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

up7 = tf.keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv6))
merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)
conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv7))
merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3)
conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv8))
merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = tf.keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv10 = tf.keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)

model = tf.keras.Model(inputs = inputs, outputs = conv10)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

################################
#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_sagittal_brain_scan', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks) 

####################################

idx = random.randint(0,len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose = 1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose = 1)
preds_test = model.predict(X_test, verbose =1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

#Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

#Perform a sanity check on some random validation samples
ix = random.randint(0,len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()













