from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import time
from flask import *
import os
from shutil import copyfile
import pandas as pd
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, MaxPool2D, BatchNormalization, Flatten
from PIL import Image
import cv2
import json
import subprocess
from pathlib import Path
import random
import string
from yolov5.detect import run
from yolov5.load_models import load_roi_model, load_digitrec_model

roi_values = load_roi_model()
digit_rec_values = load_digitrec_model()

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
g_model = None
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

# Classes of meters
classes = { 0:'Real',
            1:'Spoof', 
            }
def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def image_processing(img):
    global g_model
    if g_model is None:
        g_model = load_model('Knight.h5')
        print("model init")
    data=[]
    image = Image.open(img)
    image = image.resize((100,100))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = g_model.predict_classes(X_test)
    print(Y_pred)
    return Y_pred


def get_random_string(N=15):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))


@app.route('/reading-value')
def reading_value():
    return render_template('reading_value_index.html')


@app.route('/reading-value-display', methods=['POST'])
def reading_value_display():
    p = Path().resolve() # Path(__file__).parents[1]
    # roi_dir = str(p) + "/roi/yolov5"
    # digitrec_dir = str(p) + "/digitrec/yolov5"
    flaskML_dir = str(p) # + "/flaskML"
    filename = get_random_string()
    img = request.files['image']

    img.save(os.path.join('SavedTestImages', "{}.jpg".format(filename)))
    # one way start
    first_model_img_path, first_text_path = run(classify=roi_values[0], pt=roi_values[1], onnx=roi_values[2],
                                                stride=roi_values[3], names=roi_values[4], model=roi_values[5],
                                                modelc=roi_values[6], session=roi_values[7], device=roi_values[8],
                                                save_crop=True,
                                                infile=os.path.join(flaskML_dir, 'SavedTestImages/{}.jpg'.format(filename)))

    img_path, text_path = run(classify=digit_rec_values[0], pt=digit_rec_values[1], onnx=digit_rec_values[2],
                              stride=digit_rec_values[3], names=digit_rec_values[4], model=digit_rec_values[5],
                              modelc=digit_rec_values[6], session=digit_rec_values[7], device=roi_values[8],
                              save_txt=True, save_conf=True, infile=first_model_img_path)

    cropped_filename = filename + "_crop"
    new_strings = []
    if os.path.isfile(img_path):
        copyfile(img_path, os.path.join('SavedTestImages', "{}.jpg".format(cropped_filename)))
    if os.path.isfile(text_path):
        copyfile(text_path, os.path.join('SavedTestImages', "{}.txt".format(cropped_filename)))
        with open(os.path.join('SavedTestImages', "{}.txt".format(cropped_filename))) as f:
            content = f.readlines()
            content = [item.replace(" ", ",") for item in content]
            content = [item.replace("\n", "") for item in content]
            dataframe = pd.DataFrame(content)
            headerlist = ['a']
            dataframe.to_csv(os.path.join('SavedTestImages', "{}.csv".format(cropped_filename)), header=headerlist, index=False)
            df = pd.read_csv(os.path.join('SavedTestImages', "{}.csv".format(cropped_filename)))
            final_df = df.a.str.split(",",expand=True)
            current_max_prob_val = ''
            current_row = ''
            for index, row in final_df.iterrows():
                print(row)
                if int(row[0]) > 10:
                    print("hi", int(row[0]) > 10)
                    if current_max_prob_val:
                        if float(current_max_prob_val) < float(row[5]):
                            final_df = final_df.drop(labels=int(current_row), axis=0)
                            current_max_prob_val = row[5]
                            current_row = index
                    else:
                        current_max_prob_val = row[5]
                        current_row = index
            final_df = final_df.sort_values(by=[1], ascending=True)
            prediction = final_df[0]
            # strings = prediction
            for string in prediction:
                new_string = string.replace("10", ".").replace("11","kwh").replace("12","kw").replace("13","kvah").replace("14","kva").replace("15","pf")
                new_strings.append(new_string)
            # new_strings.append(new_strings.pop(new_strings.index('kwh')))
            # print(new_strings)
    
    # value = ''.join(new_strings)
    # parameter = ''

    params = []
    value = ""
    for st in new_strings:
        if st in ["kw", "kwh", "kvah", "kva", "pf"]:
            params.append(st)
        else:
            value = value + st
    parameter = '/'.join(params)

    # if new_strings[-1] in ["kw", "kwh", "kvah", "kva", "pf"]:
    #     parameter = new_strings[-1]
    #     value = ''.join(new_strings[:-1])
    if request.args.get('isJson', None):
        return {"value": value, "parameter": parameter}
    else:
        return render_template('reading_value_display.html', data=value+" "+parameter, fname=filename+"_crop.jpg")


    # # second way start
    # # changing directory to run the command
    # os.chdir(roi_dir)
    # # to get the image with meter reading location identified
    # p = subprocess.check_output(
    #     ['python3', 'detect.py', '--save-crop', '--infile', os.path.join(flaskML_dir, 'SavedTestImages/{}.jpg'.format(filename))],
    #                      shell=False).splitlines()
    # # changing directory to run the command
    # os.chdir(digitrec_dir)
    # # to get the cropped image of meter reading and meter reading values
    # p = subprocess.check_output(['python3', 'detect.py', '--save-txt','--infile',
    #                              p[-4].decode("utf-8").split(": ")[1]],
    #                             shell=False).splitlines()
    # # changing to original directory
    # os.chdir(flaskML_dir)
    # # copying the cropped meter reading image to this directory
    # cropped_filename = filename + "_crop"
    # copyfile(p[-1].decode("utf-8"), os.path.join('SavedTestImages', "{}.jpg".format(cropped_filename)))
    # copyfile(p[-3].decode("utf-8"), os.path.join('SavedTestImages', "{}.txt".format(cropped_filename)))
    # with open(os.path.join('SavedTestImages', "{}.txt".format(cropped_filename))) as f:
    #     content = f.readlines()
    #     content = [item.replace(" ", ",") for item in content]
    #     content = [item.replace("\n", "") for item in content]
    #     dataframe = pd.DataFrame(content)
    #     headerlist = ['a']
    #     dataframe.to_csv(os.path.join('SavedTestImages', "{}.csv".format(cropped_filename)), header=headerlist, index=False)
    #     df = pd.read_csv(os.path.join('SavedTestImages', "{}.csv".format(cropped_filename)))
    #     final_df = df.a.str.split(",",expand=True)
    #     final_df = final_df.sort_values(by=[1], ascending=True)
    #     prediction = final_df[0]
    #     # strings = prediction
    #     new_strings = []
    #     for string in prediction:
    #         new_string = string.replace("10", ".").replace("11","kwh").replace("12","kw").replace("13","kvah").replace("14","kva").replace("15","pf")
    #         new_strings.append(new_string)
    #     # new_strings.append(new_strings.pop(new_strings.index('kwh')))
    #     # print(new_strings)
    #
    # return render_template('reading_value_display.html', data=' '.join(new_strings), fname=filename+"_crop.jpg")


@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    filename = get_random_string()
    img = request.files['image']

    img.save('SavedTestImages/{}.jpg'.format(filename))
    img_arr = cv2.imread('SavedTestImages/{}.jpg'.format(filename))

    img_arr = cv2.resize(img_arr, (30,30))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 30,30,3)
    prediction = model.predict(img_arr)

    x = round(prediction[0,0], 2)
    y = round(prediction[0,1], 2)
    z = round(prediction[0,2], 2)
    preds = np.array([x,y,z])
    # COUNT += 1
    return render_template('prediction.html', data=preds, fname=filename+".jpg")


@app.route('/real-spoof')
def index():
    return render_template('real_spoof_index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = image_processing(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = classes[a]
        os.remove(file_path)
        return  {"real": float(1-a), "spoof": float(a), "nonmeter": 0}

    elif request.method == 'GET' and request.args.get('url', default=None):
        url = request.args.get('url', default=None)
        r = requests.get(url, allow_redirects=True)
        filename = str(time.time()) + '.jpg'
        print(filename)
        open(filename, 'wb').write(r.content)
        # Make prediction
        result = image_processing(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = classes[a]
        os.remove(file_path)
        return result
    return None


@app.route('/testimage', methods=['POST'])
def testimage():
    
    if request.method == 'POST':
        # global COUNT
        filename = get_random_string()
        img = request.files['file']
        img.save('SavedTestImages/{}.jpg'.format(filename))
        img_arr = cv2.imread('SavedTestImages/{}.jpg'.format(filename))

        img_arr = cv2.resize(img_arr, (30,30))
        img_arr = img_arr / 255.0
        img_arr = img_arr.reshape(1, 30,30,3)
        prediction = model.predict(img_arr)

        x = round(prediction[0,0], 2)
        y = round(prediction[0,1], 2)
        z = round(prediction[0,2], 2)
        preds = np.array([x,y,z])
        # COUNT += 1
        return {"real": float(preds[0]), "spoof": float(preds[1]), "nonmeter": float(preds[2])}

#        if preds[1] > 0.50:
 #           return 'Meter is Spoof'
  #      elif preds[2]  > 0.50:
   #         return 'METER IMAGE IS NON METER'
    #    else:
     #       return 'Meter Image is Real'
    #return None


@app.route('/load_crop_img/<fname>')
def load_crop_img(fname):
    global COUNT
    return send_from_directory('SavedTestImages', "{}".format(fname))


@app.route('/load_img/<fname>')
def load_img(fname):
    global COUNT
    print(fname)
    print('SavedTestImages', "{}".format(fname))
    return send_from_directory('SavedTestImages', "{}".format(fname))

if __name__ == '__main__':
    g_model = load_model('static/Knight.h5')
    app.run(host='0.0.0.0', port=80, debug=True)
