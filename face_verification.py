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
import sys

def triplet_loss(y_true, y_pred, alpha = 0.2):
  anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
  pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
  neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
  basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
  loss = tf.maximum(tf.reduce_mean(basic_loss), 0)

  return loss

def verify(image_path, identity, database, model):
  encoding = img_to_encoding(image_path, model)
  target_encoding = database[identity]
  distance = np.linalg.norm(encoding - target_encoding)
  
  door_open = distance < 0.7
  if door_open:
    print("It's " + str(identity) + ", welcome home!")
  else:
    print("It's not " + str(identity) + ", please go away")

  return distance, door_open

def who_is_it(image_path, database, model):
  image_path = format_properly(image_path)
  
  encoding = img_to_encoding(image_path, model)
  min_dist = 100
  for (name, db_encoding) in database.items():
    distance = np.linalg.norm(encoding - db_encoding)
    if distance < min_dist:
      min_dist = distance
      identity = name

  if min_dist < 1:
    print("It's " + str(identity) + ", the distance is " + str(min_dist))
  else:
    print("Not in the database.")
  
  return min_dist, identity