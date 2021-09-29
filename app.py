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
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2
import json
import subprocess
from pathlib import Path
import random
import string
from yolov5.detect import run
from yolov5.load_models import load_roi_model, load_digitrec_model

roi_values_best = load_roi_model()
digit_rec_values_best = load_digitrec_model()

roi_values_light = load_roi_model(weights="yolov5/models_last/slast-roi.pt")
digit_rec_values_light = load_digitrec_model(weights="yolov5/models_last/slast-digitrec.pt")

# Real/ Spoof
ensemble_model1 = load_model('static/90acc_0.24valloss.h5')

# Real/ NonMeter
ensemble_model2 = load_model('static/91acc_0.16valloss.h5')

# Real/ Spoof/ NonMeter
ensemble_model3 = load_model('static/90acc_0.25valloss.h5')

def get_classify_model1():
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
    return model

def get_classify_model2():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu', input_shape=(30,30,3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights('static/91acc15loss.h5')
    return model

# model = get_classify_model2()
COUNT = 0
app = Flask(__name__)
g_model = None
f_model = None
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

# Classes of meters
cl_classes = { 0:'Real',
            1:'Spoof', 
            }
classes = { 0:'real',
            1:'spoof',
            2:'nonmeter'}

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


def save_image(img_data, url=None):
    print(img_data, url)
    filename = None
    if img_data is not None and img_data.filename != '':
        filename = get_random_string()
        img_data.save(os.path.join('SavedTestImages', "{}.jpg".format(filename)))
    elif url is not None:
        filename = download_image(url)
    return filename


def download_image(url):
    filename = url.split('/')[-1]
    filename = filename.replace(".jpg", "")
    if not os.path.exists(os.path.join('SavedTestImages', "{}.jpg".format(filename))):
        img_data = requests.get(url).content
        with open(os.path.join('SavedTestImages', "{}.jpg".format(filename)), 'wb') as handler:
            handler.write(img_data)
    return filename

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
    filename = None
    # print(request.form)
    mtype = "best"
    stype = "upload"
    if request.form != None and request.form.get('mtype') in ["light", "best"]:
        mtype = request.form.get('mtype')

    if mtype == "light":
        roi_values = roi_values_light
        digit_rec_values = digit_rec_values_light
    else:
        roi_values = roi_values_best
        digit_rec_values = digit_rec_values_best

    if request.form != None and request.form.get('stype') in ["upload", "url"]:
        stype = request.form.get('stype')

    img_url = None
    img_data = None
    if request.form and request.form.get('url'):
        img_url = request.form.get('url')
    if request.files and 'image' in request.files:
        img_data = request.files['image']
    filename = save_image(img_data, img_url)

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
        return render_template('reading_value_display.html', data=value+" "+parameter, mtype=mtype, fname=filename+"_crop.jpg")


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
    prediction = ensemble_model3.predict(img_arr)

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
        result = cl_classes[a]
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
        result = cl_classes[a]
        os.remove(file_path)
        return result
    return None


@app.route('/classify-image', methods=['POST'])
def classify_image():
    
    if request.method == 'POST':
        # global COUNT

        img_url = None
        img_data = None
        if request.form and request.form.get('url'):
            img_url = request.form.get('url')
        if request.files and 'image' in request.files:
            img_data = request.files['image']
        filename = save_image(img_data, img_url)

        img_arr = cv2.imread('SavedTestImages/{}.jpg'.format(filename))

        # img_arr = cv2.resize(img_arr, (30,30))
        # img_arr = img_arr / 255.0
        # img_arr = img_arr.reshape(1, 30,30,3)

        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        img_arr = cv2.resize(img_arr,(128,128))
        img_arr = img_arr.astype('float')/ 255.0
        img_arr = img_to_array(img_arr)
        img_arr = np.expand_dims(img_arr,axis=0)

        results1 = list(ensemble_model1.predict_proba(img_arr)[0])
        print(results1)

        results2 = list(ensemble_model2.predict_proba(img_arr)[0])
        print(results2)

        results3 = list(ensemble_model3.predict_proba(img_arr)[0])
        print(results3)

        final_class = ""  # Default
        spoof_res = ""
        nonmeter_res = ""
        if (np.max(results1) - np.min(results1) > 0.5): # and np.max(results1) > 0.6
            spoof_res = classes[0] if results1[0] > results1[1] else classes[1]
            print("res1", spoof_res)
        if (np.max(results2) - np.min(results2) > 0.5): # and np.max(results2) > 0.6
            nonmeter_res = classes[0] if results2[0] > results2[2] else classes[2]
            print("res2", nonmeter_res)

        results3 = [results3[0], max(results3[1], results1[1]), max(results3[2], results2[2])]
        print(results3)
        if spoof_res == "":
            if nonmeter_res == "":
                if (results3[0] > results3[1]):
                    final_class = classes[2] if results3[2] > results3[0] else classes[0]
                else:
                    final_class = classes[2] if results3[2] > results3[1] else classes[1]
                return results3
            elif nonmeter_res == "real":
                final_class = classes[1] if results3[1] > results3[0] else classes[0]
            else:
                final_class = nonmeter_res
        elif spoof_res == "real":
            if nonmeter_res == "":
                final_class = classes[2] if results3[2] > results3[0] else classes[0]
            else:
                final_class = nonmeter_res
        else:
            final_class = spoof_res
        # return final_class

        # prediction = model.predict(img_arr)

        # x = round(prediction[0,0], 2)
        # y = round(prediction[0,1], 2)
        # z = round(prediction[0,2], 2)
        # preds = np.array([x,y,z])
        # COUNT += 1
        return_val = {"real": 0, "spoof": 0, "nonmeter": 0}
        return_val[final_class] = 1

        if request.args.get('isJson', None):
            return return_val
        else:
            return render_template('prediction.html', data=list(return_val.values()), class_label=final_class, fname=filename+".jpg")
#        if preds[1] > 0.50:
 #           return 'Meter is Spoof'
  #      elif preds[2]  > 0.50:
   #         return 'METER IMAGE IS NON METER'
    #    else:
     #       return 'Meter Image is Real'
    #return None

@app.route('/clear-blur')
def clearblur():
    return render_template('clear_blur_index.html')


@app.route('/clear-blur-prediction', methods=['POST'])
def clear_blur_prediction():
    global f_model
    if f_model is None:
        f_model = load_model('static/92acc33loss.h5')
        print("model init")

    img_url = None
    img_data = None
    if request.form and request.form.get('url'):
        img_url = request.form.get('url')
    if request.files and 'image' in request.files:
        img_data = request.files['image']
    filename = save_image(img_data, img_url)

    img_arr = cv2.imread('SavedTestImages/{}.jpg'.format(filename))

    img_arr = cv2.resize(img_arr, (100,100))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 100,100,3)
    prediction = f_model.predict(img_arr)

    x = round(prediction[0,0], 2)
    y = round(prediction[0,1], 2)
    preds = np.array([x,y])

    if request.args.get('isJson', None):
        return {"blur": float(preds[0]), "clear": float(preds[1])}
    else:
        return render_template('clear_blur_prediction.html', data=preds, fname=filename+".jpg")


@app.route('/load_clear_blur_img')
def load_clear_blur_img(fname):
    global COUNT
    return send_from_directory('Saved Test Images', "{}.jpg".format(fname))

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
    app.run(host='0.0.0.0', port=80, debug=False)
