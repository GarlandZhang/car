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
import tensorflow as tf
from keras.models import load_model
from tensorflow.python.keras.backend import set_session
from fr_utils import *
from face_verification import triplet_loss
from .models import FaceEncoding
from . import db
import copy
import json

def import_settings():
  K.set_image_data_format('channels_first')
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

  if not db.session.query(FaceEncoding).filter(FaceEncoding.name == 'garland').count():
    encoding = img_to_encoding('images/me.jpg', model)
    payload = { "encoding": encoding.tolist() }
    db.session.add(FaceEncoding(name='garland', encoding=json.dumps(payload)))
    db.session.commit()

  database = {}
  database["dad"] = img_to_encoding("images/me.jpg", model)
  print("Ready!")