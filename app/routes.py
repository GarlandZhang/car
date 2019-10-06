from app import *
import urllib.request
from face_verification import triplet_loss, verify, who_is_it
from face_detection import get_face
from flask import Flask, request, redirect, url_for, render_template, jsonify
from tensorflow import keras
from werkzeug.utils import secure_filename
from PIL import Image
import io
import cv2
import numpy as np
from fr_utils import *
from inception_blocks_v2 import *
from keras import backend as K
import sys
K.set_image_data_format('channels_first')

def to_image(file):
      image_str = file.read()
      array = np.fromstring(image_str, np.uint8)
      return cv2.imdecode(array, cv2.IMREAD_COLOR)

@app.route('/', methods = ['GET'])
def upload():
    return render_template('upload.html')

@app.route('/add_person_image', methods = ['POST'])
def add_person_image():
  if request.method == 'POST':
    with graph.as_default():
      set_session(sess)
      name = request.form['name']
      file = request.files['file']
      if file.filename == '':
        return redirect('/')
      image = to_image(file)
      image = get_face(image, detection_model)
      if not image:
        return jsonify({ "error": "No face found." })
      database[name] = img_to_encoding(image, model)
      result = {
        "image": database[name].tolist() 
      }
      return jsonify(result)
    
@app.route('/predict', methods = ['POST'])
def predict():
  if request.method == 'POST':
    with graph.as_default():
      set_session(sess)
      file = request.files['file']
      image = to_image(file)
      image = get_face(image, detection_model)
      if not image:
        return jsonify({ "error": "No face found." })
      min_dist, identity = who_is_it(image, database, model)
      result = {
        "min_dist": str(min_dist),
        "identity": identity,
      }
  return jsonify(result)


  
  