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
import sys     

np.set_printoptions(threshold=sys.maxsize)

FRmodel = faceRecoModel(input_shape=(3,96,96))

with tf.Session() as test:
  tf.set_random_seed(1)
  y_true = (None, None, None)
  y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
            tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
            tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
  loss = triplet_loss(y_true, y_pred)
  print("loss = " + str(loss.eval()))

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

database = {}
# database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
# database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
# database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
# database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
# database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
# database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
# database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
# database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
# database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
# database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
# database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
# database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)

# need to insert the actual cropped image (ground truth)
database["dad"] = img_to_encoding("images/dad-in-san-fran-just-face.jpg", FRmodel)

file_name = 'dad-in-china.jpg'
image_path = 'images/' + file_name
model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')
(boxes, confidences) = get_boxes(image_path, model)

for i, box in enumerate(boxes):
  image = cv2.imread(image_path)
  startX, startY, endX, endY = box
  crop = image[int(startY):int(endY), int(startX):int(endX)]
  
  # save image
  output_path = 'Output/' + file_name + "_cropped" + str(i) + ".jpg"
  cv2.imwrite(output_path, crop)
  who_is_it(output_path, database, FRmodel)

