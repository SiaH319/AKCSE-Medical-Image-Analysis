from flask import Flask, redirect, render_template, request, send_from_directory
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
import os
import jinja2

curr = os.getcwd()

inputCheckModel = Sequential()

inputCheckModel.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128,128,3)))
inputCheckModel.add(BatchNormalization())
inputCheckModel.add(MaxPooling2D(pool_size=(2, 2)))
inputCheckModel.add(Dropout(0.25))

inputCheckModel.add(Conv2D(64, (3, 3), activation='relu'))
inputCheckModel.add(BatchNormalization())
inputCheckModel.add(MaxPooling2D(pool_size=(2, 2)))
inputCheckModel.add(Dropout(0.25))

inputCheckModel.add(Conv2D(128, (3, 3), activation='relu'))
inputCheckModel.add(BatchNormalization())
inputCheckModel.add(MaxPooling2D(pool_size=(2, 2)))
inputCheckModel.add(Dropout(0.25))

inputCheckModel.add(Flatten())
inputCheckModel.add(Dense(512, activation='relu'))
inputCheckModel.add(BatchNormalization())
inputCheckModel.add(Dropout(0.5))
inputCheckModel.add(Dense(2, activation='softmax'))
inputCheckModel.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

inputCheckModel.load_weights(__location__+'/static/model/cat-and-sagittal.h5')

tumorModel = Sequential()

tumorModel.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128,128,3)))
tumorModel.add(BatchNormalization())
tumorModel.add(MaxPooling2D(pool_size=(2, 2)))
tumorModel.add(Dropout(0.25))

tumorModel.add(Conv2D(64, (3, 3), activation='relu'))
tumorModel.add(BatchNormalization())
tumorModel.add(MaxPooling2D(pool_size=(2, 2)))
tumorModel.add(Dropout(0.25))

tumorModel.add(Conv2D(128, (3, 3), activation='relu'))
tumorModel.add(BatchNormalization())
tumorModel.add(MaxPooling2D(pool_size=(2, 2)))
tumorModel.add(Dropout(0.25))

tumorModel.add(Flatten())
tumorModel.add(Dense(512, activation='relu'))
tumorModel.add(BatchNormalization())
tumorModel.add(Dropout(0.5))
tumorModel.add(Dense(4, activation='softmax'))
tumorModel.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#tumorModel.load_weights(__location__+'/static/model/tumor_final_model.h5')

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def mainPage():
    return render_template('index-main.html')

@app.route('/members')
def members():
    return render_template('index-main.html')

@app.route('/project/description')
def projectDescription():
    return render_template('project-description.html')

@app.route('/project/demo/brain-mapping')
def brainMappingDemo():
    return render_template('project-demo-brain-mapping.html')


@app.route('/project/demo/brain-mapping/result', methods=['POST'])
def brainMappingDemoResult():
    global COUNT
    img = request.files['image']

    img.save(__location__+'/static/{}.jpg'.format(COUNT))
    img_arr = cv2.imread(__location__+'/static/{}.jpg'.format(COUNT))
    img_arr = cv2.resize(img_arr, (128,128))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 128,128,3)
    prediction = inputCheckModel.predict(img_arr)

    x = round(prediction[0,0], 2)
    y = round(prediction[0,1], 2)
    preds = np.array([x,y])
    COUNT += 1
    return render_template('project-demo-brain-mapping-result.html', data=preds)

@app.route('/project/demo/tumor-detection')
def tumorDetectionDemo():
    return render_template('project-demo-tumor-detection.html')

@app.route('/project/demo/tumor-detection/result', methods=['POST'])
def tumorDetectionDemoResult():
    global COUNT
    img = request.files['image']

    img.save(__location__+'/static/{}.jpg'.format(COUNT))
    img_arr = cv2.imread(__location__+'/static/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (128,128))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 128,128,3)
    prediction = tumorModel.predict(img_arr)

    a = round((1-prediction[0,0])*100, 2)
    b = round(prediction[0,1]*100, 2)
    c = round(prediction[0,2]*100, 2)
    d = round(prediction[0,3]*100, 2)


    preds = np.array([a,b,c,d])
    COUNT += 1
    return render_template('project-demo-tumor-detection-result.html', tumorData=preds)

@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory(__location__+'/static', "{}.jpg".format(COUNT-1))

@app.route('/SiaHamGithub')
def SiaHamGithub():
    return redirect("https://github.com/SiaH319")

@app.route('/SiaHamLinkedin')
def SiaHamLinkedin():
    return redirect("https://www.linkedin.com/in/sia-ham")

@app.route('/RhinaKimGithub')
def RhinaKimGithub():
    return redirect("https://github.com/chihiroanihr")


@app.route('/RhinaKimLinkedin')
def RhinaKimLinkedin():
    return redirect("https://www.linkedin.com/in/rhina-kim-568ab3178/")

@app.route('/JamesSungLinkedin')
def JamesSungLinkedin():
    return redirect("https://www.linkedin.com/in/james-sung-4a1b3a193/")

@app.route('/dataset1')
def dataset1():
    return redirect("https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection")

@app.route('/dataset2')
def dataset2():
    return redirect("https://www.robots.ox.ac.uk/~vgg/software/via/")

@app.route('/dataset3')
def dataset3():
    return redirect("https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri")

@app.route('/dataset4')
def dataset4():
    return redirect("https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/")

if __name__ == '__main__':
    app.run(debug=True)
