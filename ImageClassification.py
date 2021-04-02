import matplotlib.pyplot as plt
import cv2
import PIL
#%matplotlib inline
from keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range = 30, width_shift_range =0.1, height_shift_range = 0.1, rescale = 1/255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = 'nearest')
image_shape = (28, 28, 3)
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape=(28,28,3), activation = 'relu',))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape=(28,28,3), activation = 'relu',))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape=(28,28,3), activation = 'relu',))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(120))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
batch_size = 20
train_image_gen = image_gen.flow_from_directory('Train',target_size=image_shape[:2], batch_size=batch_size, class_mode='binary')
batch_size = 20
test_image_gen = image_gen.flow_from_directory('Test',target_size=image_shape[:2], batch_size=batch_size, class_mode='binary')
train_image_gen.class_indices
import warnings
warnings.filterwarnings('ignore')
results = model.fit_generator(train_image_gen, epochs = 1, steps_per_epoch =1, validation_data=test_image_gen, validation_steps=12)
model.save('MRI_Classifier.hS')
plt.plot(results.history['acc'])
train_image_gen.class_indices
import numpy as np
from keras.preprocessing import image
isit_file = 'file.jpg'
isit_file = image.load_img(isit_file, target_size =(150, 150))
isit_file = image.img_to_array(isit_file)
isit_file=np.expand_dims(isit_file,axis=0)
isit_file = isit_file/255
prediction_prob = model.predict(isit_file)
print(f'Probability that image is a Sagital is: {prediction_prob}')

