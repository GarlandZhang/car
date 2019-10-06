from flask import Flask
from face_detection import get_boxes
from face_verification import triplet_loss, verify, who_is_it
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from keras.models import load_model
from tensorflow.python.keras.backend import set_session
import sys     

app = Flask(__name__)

def init():
  global model, graph, database, sess, detection_model
  # model = faceRecoModel(input_shape=(3,96,96))
  # graph = tf.get_default_graph()
  # model.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
  # load_weights_from_FaceNet(model)
  # model.save('my_model.h5')
  sess = tf.Session()
  graph = tf.get_default_graph()

  set_session(sess)
  model = load_model('my_model.h5')
  model.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])

  detection_model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel') 

  database = {}
  database["dad"] = img_to_encoding("images/dad-in-san-fran-just-face.jpg", model)
  print("Ready!")

UPLOAD_FOLDER = '/uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

np.set_printoptions(threshold=sys.maxsize)

print("* Loading model and Flask server")
init()

from app import routes