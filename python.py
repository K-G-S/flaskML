from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import time
from flask import *
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Classes of meters
classes = { 0:'Real',
            1:'Spoof', 
            }
def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def image_processing(img):
    parser = argparse.ArgumentParser()
    parser.add_argument(
       '-i',
       '--image',
       default='/tmp/grace_hopper.bmp',
       help='image to be classified')
    parser.add_argument(
       '-m',
       '--model_file',
       default='Lightmodel.tflite',
       help='.tflite model to be executed')
    parser.add_argument(
       '-l',
       '--label_file',
       default='label.txt',
       help='name of file containing labels')
    parser.add_argument(
       '--input_mean',
       default=0., type=float,
       help='input_mean')
       parser.add_argument(
       '--input_std',
       default=255., type=float,
       help='input standard deviation')
    parser.add_argument(
       '--num_threads', default=None, type=int, help='number of threads')
    args = parser.parse_args()
    interpreter = tf.lite.Interpreter(
      model_path=args.model_file, num_threads=args.num_threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(args.image).resize((width, height))

    # add N dim
    input_data = np.expand_dims(img, axis=0)    
    if floating_model:    
      input_data = (np.float32(input_data) - args.input_mean) / args.input_std   
    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    #top_k = results.argsort()[-5:][::-1]
    labels = load_labels(args.label_file)
    if floating_model:
        return results,labels
    else:
        return (results/255.0),labels
            
    model = load_model('Knight.h5')
    data=[]
    image = Image.open(img)
    image = image.resize((100,100))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = model.predict_classes(X_test)
    return Y_pred

@app.route('/')
def index():
    return render_template('index.html')

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
        result = "The Meter Image is : " +classes[a]
        os.remove(file_path)
        return result
    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80,debug=True)
