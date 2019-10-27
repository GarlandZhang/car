from flask import Flask
from face_detection import get_boxes
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
from inception_blocks_v2 import *
import psycopg2
import sys      
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
global db
db = SQLAlchemy(app)

from .models import FaceEncoding
from .settings import import_settings

def init():  
  app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
  app.config['UPLOAD_FOLDER'] = '/uploads'
  np.set_printoptions(threshold=sys.maxsize)
  import_settings()

print("* Loading model and Flask server")
init()

from app import routes