from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
import tensorflow as tf
from predict_image import detect_and_predict_mask
import cv2

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

path = r"D:\Mask_detection\mask_detector_own\saved_models\Augmented_CNN.model"
maskNet = tf.keras.models.load_model(path)

print('Model loaded. Check http://127.0.0.1:5000/')

def predict_(locs,preds):
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (withoutMask, mask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        return label
    return "None"
    
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Make sure to create a uploads folder in current working directory
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        locs,preds = detect_and_predict_mask(cv2.imread(file_path), maskNet)

        label = predict_(locs,preds)
        return label
    return None


if __name__ == '__main__':
    app.run(debug=True)

