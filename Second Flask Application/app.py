from flask import *
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, MaxPool2D, BatchNormalization, Flatten
from PIL import Image
import cv2
import requests
import time

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(30,30,3)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights('static/model.h5')

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('Saved Test Images/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('Saved Test Images/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (30,30))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 30,30,3)
    prediction = model.predict(img_arr)

    x = round(prediction[0,0], 2)
    y = round(prediction[0,1], 2)
    z = round(prediction[0,2], 2)
    preds = np.array([x,y,z])
    COUNT += 1
    return render_template('prediction.html', data=preds)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        img = request.files['image']

        img.save('Saved Test Images/{}.jpg'.format(COUNT))    
        img_arr = cv2.imread('Saved Test Images/{}.jpg'.format(COUNT))

        img_arr = cv2.resize(img_arr, (30,30))
        img_arr = img_arr / 255.0
        img_arr = img_arr.reshape(1, 30,30,3)
        prediction = model.predict(img_arr)

        x = round(prediction[0,0], 2)
        y = round(prediction[0,1], 2)
        z = round(prediction[0,2], 2)
        preds = np.array([x,y,z])
        return {"real": float(preds[0]), "spoof": float(preds[1]), "nonmeter": float(preds[2])}
    elif request.method == 'GET' and request.args.get('url', default=None):
        url = request.args.get('url', default=None)
        r = requests.get(url, allow_redirects=True)
        filename = str(time.time()) + '.jpg'
        file_path = "Saved Test Images/" + filename
        print(file_path)
        open(file_path, 'wb').write(r.content)
        # Make prediction
        img_arr = cv2.imread(file_path)

        img_arr = cv2.resize(img_arr, (30,30))
        img_arr = img_arr / 255.0
        img_arr = img_arr.reshape(1, 30,30,3)
        prediction = model.predict(img_arr)

        x = round(prediction[0,0], 2)
        y = round(prediction[0,1], 2)
        z = round(prediction[0,2], 2)
        preds = np.array([x,y,z])
        os.remove(file_path)
        return {"real": float(preds[0]), "spoof": float(preds[1]), "nonmeter": float(preds[2])}
    return None


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('Saved Test Images', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)
