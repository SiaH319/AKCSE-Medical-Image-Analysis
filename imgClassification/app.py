from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
import os

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

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def mainPage():
    return render_template('index-main.html')


@app.route('/members')
def members():
    return render_template('index-member.html')


@app.route('/project/demo')
def projectDemo():
    return render_template('project-demo.html')


@app.route('/project/description')
def projectDescription():
    return render_template('project-description.html')


@app.route('/project/demo/result', methods=['POST'])
def demoResult():
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
    return render_template('project-demo-result.html', data=preds)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory(__location__+'/static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)