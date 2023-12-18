from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import operator
import functools
import math
import os
import skimage.io as io
import random
from scipy import ndimage
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Activation, Input, Lambda, Concatenate,Average, Permute, concatenate
from tensorflow.keras.datasets import mnist,fashion_mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from scipy import ndimage
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
import sys
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

positions_i = ["justAPlaceholder", "supine_1", "right_0",
               "left_0", "right_30", "right_60",
               "left_30", "left_60", "supine_2",
               "supine_3", "supine_4", "supine_5",
               "supine_6", "right_fetus", "left_fetus",
               "supine_30", "supine_45", "supine_60"]

positions_i_short = ["justAPlaceholder", "supine", "right",
               "left", "right", "right",
               "left", "left", "supine",
               "supine", "supine", "supine",
               "supine", "right", "left",
               "supine", "supine", "supine"]

positions_ii = {
    "B":"supine", "1":"supine", "C":"right",
    "D":"left", "E1":"right", "E2":"right",
    "E3":"left", "E4":"left", "E5":"right",
    "E6":"left", "F":"supine", "G1":"supine",
    "G2":"right", "G3":"left"
}

class_positions = ['supine', 'left', 'right', 'left_fetus', 'right_fetus']

# We also want the classes to be encoded as numbers so we can work easier when
# modeling. This function achieves so. Since left_fetus and right_fetus are not
# considered as classes in the evaluation of the original paper and since they
# are not considered in the "Experiment I", we encode them also as left and right
# positions.

def token_position_short(x):
  return {
      'supine': 0,
      'left': 1,
      'right': 2,
      'left_fetus': 1,
      'right_fetus': 2
  }[x]

def token_position(x):
  return {
      "supine_1":0, 
      "right_0":1,
      "left_0":2, 
      "right_30":3, 
      "right_60":4,
      "left_30":5, 
      "left_60":6, 
      "supine_2":7,
      "supine_3":8, 
      "supine_4":9, 
      "supine_5":10,
      "supine_6":11, 
      "right_fetus":12, 
      "left_fetus":13,
      "supine_30":14, 
      "supine_45":15, 
      "supine_60":16
  }[x]

def token_position_new(x):
  return {
      "supine_1":0, 
      "supine_2":1,
      "supine_3":2, 
      "supine_4":3, 
      "supine_5":4,
      "supine_6":5, 
      "supine_30":6, 
      "supine_45":7, 
      "supine_60":8, 
      "left_0":9, 
      "left_30":10, 
      "left_60":11,
      "left_fetus":12, 
      "right_0":13,
      "right_30":14, 
      "right_60":15,
      "right_fetus":16, 
  }[x]
list_supine = ["1.txt","8.txt","9.txt","10.txt","11.txt","12.txt","15.txt","16.txt","17.txt"]

list_supine_norm_1 = ["1.txt","8.txt","9.txt"]

list_supine_norm_2 = ["10.txt","11.txt","12.txt"]

list_supine_incl = ["15.txt","16.txt","17.txt"]

list_left = ["3.txt","6.txt","7.txt","14.txt"]

list_right = ["2.txt","4.txt","5.txt","13.txt"]

def token_position_supine(x):
  return {
      "supine_1":0, 
      "supine_2":1,
      "supine_3":2, 
      "supine_4":3, 
      "supine_5":4,
      "supine_6":5, 
      "supine_30":6, 
      "supine_45":7, 
      "supine_60":8
    }[x]

def token_position_supine_norm_1(x):
  return {
      "supine_1":0, 
      "supine_2":1,
      "supine_3":2, 
    }[x]

def token_position_supine_norm_2(x):
  return {
      "supine_4":0, 
      "supine_5":1,
      "supine_6":2, 
    }[x]

def token_position_supine_incl(x):
  return {
      "supine_30":0, 
      "supine_45":1, 
      "supine_60":2
    }[x]

def token_position_left(x):
  return {
      "left_0":0, 
      "left_30":1, 
      "left_60":2,
      "left_fetus":3,
  }[x]

def token_position_right(x):
  return {
      "right_0":0,
      "right_30":1, 
      "right_60":2,
      "right_fetus":3, 
  }[x]


def load_exp_i(path,preprocess=True):
  """
  Creates a numpy array for the data and labels.
  params:
  ------
  path    -- Data path.
  returns:
  -------
  A numpy array (data, labels).
  """

  dataset = {}

  for _, dirs, _ in os.walk(path):
    for directory in dirs:
      # each directory is a subject
      subject = directory
      data = None
      labels = None
      max_val = []
      for _, _, files in os.walk(os.path.join(path, directory)):
        # print(files)
        for file in files:
          # print(file)
          file_path = os.path.join(path, directory, file)
          with open(file_path, 'r') as f:
            lines = f.read().splitlines()[2:]
            for i in range(3, len(lines) - 3):
                              
              raw_data = np.fromstring(lines[i], dtype=float, sep='\t').reshape(64, 32)
              
              if preprocess is True:
                past_image = np.fromstring(lines[i-1], dtype=float, sep='\t').reshape(64, 32)
                future_image = np.fromstring(lines[i+1], dtype=float, sep='\t').reshape(64, 32)
                
                # Spatio-temporal median filter 3x3x3
                raw_data = ndimage.median_filter(raw_data, 3)
                past_image = ndimage.median_filter(past_image, 3)
                future_image = ndimage.median_filter(future_image, 3)
                raw_data = np.concatenate((raw_data[np.newaxis, :, :], past_image[np.newaxis, :, :], future_image[np.newaxis, :, :]), axis=0)
                raw_data = np.median(raw_data, axis=0)
            
            # with open(file_path, 'r') as f:
            #   # Start from second recording, as the first two are corrupted
            #   lines = f.read().splitlines()[2:]
            #   for line in f.read().splitlines()[2:]:
            #     # print(line)
            #     raw_data = np.fromstring(line, dtype=float, sep='\t')
                # Change the range from [0-1000] to [0-255].
                  # max_val.append(np.amax(raw_data))
              file_data = np.round(raw_data*255/1000).astype(np.uint8)
              # file_data = np.round(raw_data).astype(np.uint8)
              
              file_data = file_data.reshape((1,64,32))
              # print(positions_i[int(file[:-4])])
              file_label = token_position(positions_i[int(file[:-4])])
              # print("directory: ",directory,"file_name: " ,file,"file_label: ",file_label)
              file_label = np.array([file_label])

              if data is None:
                data = file_data
              else:
                data = np.concatenate((data, file_data), axis=0)
              if labels is None:
                labels = file_label
              else:
                labels = np.concatenate((labels, file_label), axis=0)
      
      # max_over_all = max(max_val)
      # print(max_over_all)

      # data = np.round(data * 255/1000).astype(np.uint8)
      dataset[subject] = (data, labels)

  return dataset

def load_exp_i_short(path,preprocess=True):
  """
  Creates a numpy array for the data and labels.
  params:
  ------
  path    -- Data path.
  returns:
  -------
  A numpy array (data, labels).
  """

  dataset = {}

  for _, dirs, _ in os.walk(path):
    for directory in dirs:
      # each directory is a subject
      subject = directory
      data = None
      labels = None
      max_val = []
      for _, _, files in os.walk(os.path.join(path, directory)):
        # print(files)
        for file in files:
          # print(file)
          file_path = os.path.join(path, directory, file)
          with open(file_path, 'r') as f:
            lines = f.read().splitlines()[2:]
            for i in range(3, len(lines) - 3):
                              
              raw_data = np.fromstring(lines[i], dtype=float, sep='\t').reshape(64, 32)
              
              if preprocess is True:
                past_image = np.fromstring(lines[i-1], dtype=float, sep='\t').reshape(64, 32)
                future_image = np.fromstring(lines[i+1], dtype=float, sep='\t').reshape(64, 32)
                
                # Spatio-temporal median filter 3x3x3
                raw_data = ndimage.median_filter(raw_data, 3)
                past_image = ndimage.median_filter(past_image, 3)
                future_image = ndimage.median_filter(future_image, 3)
                raw_data = np.concatenate((raw_data[np.newaxis, :, :], past_image[np.newaxis, :, :], future_image[np.newaxis, :, :]), axis=0)
                raw_data = np.median(raw_data, axis=0)
              file_data = np.round(raw_data*255/1000).astype(np.uint8)
              # file_data = np.round(raw_data).astype(np.uint8)
              
              file_data = file_data.reshape((1,64,32))
              # print(positions_i[int(file[:-4])])
              file_label = token_position_short(positions_i_short[int(file[:-4])])
              # print("directory: ",directory,"file_name: " ,file,"file_label: ",file_label)
              file_label = np.array([file_label])

              if data is None:
                data = file_data
              else:
                data = np.concatenate((data, file_data), axis=0)
              if labels is None:
                labels = file_label
              else:
                labels = np.concatenate((labels, file_label), axis=0)

      dataset[subject] = (data, labels)
  return dataset

def load_exp_i_supine(path,preprocess=True):
  """
  Creates a numpy array for the data and labels.
  params:
  ------
  path    -- Data path.
  returns:
  -------
  A numpy array (data, labels).
  """

  dataset = {}

  for _, dirs, _ in os.walk(path):
    for directory in dirs:
      # each directory is a subject
      subject = directory
      data = None
      labels = None
      for _, _, files in os.walk(os.path.join(path, directory)):
        files = list_supine
        # print(files)
        for file in files:
          # print(file)
          file_path = os.path.join(path, directory, file)
          with open(file_path, 'r') as f:
            # Start from second recording, as the first two are corrupted
            lines = f.read().splitlines()[2:]
            for i in range(5, len(lines) - 5):
                            
              raw_data = np.fromstring(lines[i], dtype=float, sep='\t').reshape(64, 32)
              
              if preprocess is True:
                past_image_1 = np.fromstring(lines[i-1], dtype=float, sep='\t').reshape(64, 32)
                future_image_1 = np.fromstring(lines[i+1], dtype=float, sep='\t').reshape(64, 32)
                past_image_2 = np.fromstring(lines[i-2], dtype=float, sep='\t').reshape(64, 32)
                future_image_2 = np.fromstring(lines[i+2], dtype=float, sep='\t').reshape(64, 32)
              
                # Spatio-temporal median filter 5x5x5
              
                raw_data = ndimage.median_filter(raw_data, 3)
                
                past_image_1 = ndimage.median_filter(past_image_1, 3)
                future_image_1 = ndimage.median_filter(future_image_1, 3)
                past_image_2 = ndimage.median_filter(past_image_2, 3)
                future_image_2 = ndimage.median_filter(future_image_2, 3)

                raw_data = np.concatenate((past_image_2[np.newaxis, :, :],past_image_1[np.newaxis, :, :] ,raw_data[np.newaxis, :, :], \
                future_image_1[np.newaxis, :, :],future_image_2[np.newaxis, :, :]), axis=0)
                raw_data = np.median(raw_data, axis=0)
              
              # a=np.amax(raw_data)

              file_data = np.round(raw_data*255/1000).astype(np.uint8)
              
              # file_data = np.round(raw_data).astype(np.uint8)
              
              file_data = file_data.reshape((1,64,32))

              file_label = token_position_supine(positions_i[int(file[:-4])])
              
              file_label = np.array([file_label])

              if data is None:
                data = file_data
              else:
                data = np.concatenate((data, file_data), axis=0)
              if labels is None:
                labels = file_label
              else:
                labels = np.concatenate((labels, file_label), axis=0)

      dataset[subject] = (data, labels)
  return dataset

def load_exp_i_supine_norm_1(path):
  """
  Creates a numpy array for the data and labels.
  params:
  ------
  path    -- Data path.
  returns:
  -------
  A numpy array (data, labels).
  """

  dataset = {}

  for _, dirs, _ in os.walk(path):
    for directory in dirs:
      # each directory is a subject
      subject = directory
      data = None
      labels = None
      for _, _, files in os.walk(os.path.join(path, directory)):
        files = list_supine_norm_1
        # print(files)
        for file in files:
          # print(file)
          file_path = os.path.join(path, directory, file)
          with open(file_path, 'r') as f:
            # Start from second recording, as the first two are corrupted
            for line in f.read().splitlines()[2:]:
              # print(line)
              raw_data = np.fromstring(line, dtype=float, sep='\t')
              # Change the range from [0-1000] to [0-255].
              max_val = np.amax(raw_data)
              file_data = np.round(raw_data*255/max_val).astype(np.uint8)
              
              # file_data = np.round(raw_data).astype(float)
              
              file_data = file_data.reshape((1,64,32))

              file_label = token_position_supine_norm_1(positions_i[int(file[:-4])])
              # print("directory: ",directory,"file_name: " ,file,"file_label: ",file_label)
              file_label = np.array([file_label])

              if data is None:
                data = file_data
              else:
                data = np.concatenate((data, file_data), axis=0)
              if labels is None:
                labels = file_label
              else:
                labels = np.concatenate((labels, file_label), axis=0)

      dataset[subject] = (data, labels)
  return dataset

def load_exp_i_supine_norm_2(path):
  """
  Creates a numpy array for the data and labels.
  params:
  ------
  path    -- Data path.
  returns:
  -------
  A numpy array (data, labels).
  """

  dataset = {}

  for _, dirs, _ in os.walk(path):
    for directory in dirs:
      # each directory is a subject
      subject = directory
      data = None
      labels = None
      for _, _, files in os.walk(os.path.join(path, directory)):
        files = list_supine_norm_2
        # print(files)
        for file in files:
          # print(file)
          file_path = os.path.join(path, directory, file)
          with open(file_path, 'r') as f:
            # Start from second recording, as the first two are corrupted
            for line in f.read().splitlines()[2:]:
              # print(line)
              raw_data = np.fromstring(line, dtype=float, sep='\t')
              # Change the range from [0-1000] to [0-255].
              max_val = np.amax(raw_data)
              file_data = np.round(raw_data*255/max_val).astype(np.uint8)
              
              # file_data = np.round(raw_data).astype(float)
              
              file_data = file_data.reshape((1,64,32))

              file_label = token_position_supine_norm_2(positions_i[int(file[:-4])])
              # print("directory: ",directory,"file_name: " ,file,"file_label: ",file_label)
              file_label = np.array([file_label])

              if data is None:
                data = file_data
              else:
                data = np.concatenate((data, file_data), axis=0)
              if labels is None:
                labels = file_label
              else:
                labels = np.concatenate((labels, file_label), axis=0)

      dataset[subject] = (data, labels)
  return dataset

def load_exp_i_supine_incl(path,preprocess=True):
  """
  Creates a numpy array for the data and labels.
  params:
  ------
  path    -- Data path.
  returns:
  -------
  A numpy array (data, labels).
  """

  dataset = {}

  for _, dirs, _ in os.walk(path):
    for directory in dirs:
      # each directory is a subject
      subject = directory
      data = None
      labels = None
      for _, _, files in os.walk(os.path.join(path, directory)):
        files = list_supine_incl
        # print(files)
        for file in files:
          # print(file)
          file_path = os.path.join(path, directory, file)
          with open(file_path, 'r') as f:
            # Start from second recording, as the first two are corrupted
            # with open(file_path, 'r') as f:
            lines = f.read().splitlines()[2:]
            for i in range(3, len(lines) - 3):

              raw_data = np.fromstring(lines[i], dtype=float, sep='\t').reshape(64, 32)
              
              if preprocess is True:
                past_image = np.fromstring(lines[i-1], dtype=float, sep='\t').reshape(64, 32)
                future_image = np.fromstring(lines[i+1], dtype=float, sep='\t').reshape(64, 32)
                
                # Spatio-temporal median filter 3x3x3
                raw_data = ndimage.median_filter(raw_data, 3)
                past_image = ndimage.median_filter(past_image, 3)
                future_image = ndimage.median_filter(future_image, 3)
                raw_data = np.concatenate((raw_data[np.newaxis, :, :], past_image[np.newaxis, :, :], future_image[np.newaxis, :, :]), axis=0)
                raw_data = np.median(raw_data, axis=0)

              # Change the range from [0-1000] to [0-255].
              # max_vol = np.amax(raw_data)
              file_data = np.round(raw_data ).astype(np.uint8)

              # file_data = np.round(raw_data).astype(np.uint8)
              file_data = file_data.reshape(1, 64, 32)

              file_label = token_position_supine_incl(positions_i[int(file[:-4])])
              # print("directory: ",directory,"file_name: " ,file,"file_label: ",file_label)
              file_label = np.array([file_label])

              if data is None:
                data = file_data
              else:
                data = np.concatenate((data, file_data), axis=0)
              if labels is None:
                labels = file_label
              else:
                labels = np.concatenate((labels, file_label), axis=0)

      dataset[subject] = (data, labels)
  return dataset

def load_exp_i_left(path,preprocess=True):
  """
  Creates a numpy array for the data and labels.
  params:
  ------
  path    -- Data path.
  returns:
  -------
  A numpy array (data, labels).
  """

  dataset = {}

  for _, dirs, _ in os.walk(path):
    for directory in dirs:
      # each directory is a subject
      subject = directory
      data = None
      labels = None
      for _, _, files in os.walk(os.path.join(path, directory)):
        files = list_left
        # print(files)
        for file in files:
          # print(file)
          file_path = os.path.join(path, directory, file)
          with open(file_path, 'r') as f:
            # Start from second recording, as the first two are corrupted
            lines = f.read().splitlines()[2:]
            for i in range(5, len(lines) - 5):
                            
              raw_data = np.fromstring(lines[i], dtype=float, sep='\t').reshape(64, 32)
              
              if preprocess is True:
                past_image_1 = np.fromstring(lines[i-1], dtype=float, sep='\t').reshape(64, 32)
                future_image_1 = np.fromstring(lines[i+1], dtype=float, sep='\t').reshape(64, 32)
                past_image_2 = np.fromstring(lines[i-2], dtype=float, sep='\t').reshape(64, 32)
                future_image_2 = np.fromstring(lines[i+2], dtype=float, sep='\t').reshape(64, 32)
              
                # Spatio-temporal median filter 5x5x5
              
                raw_data = ndimage.median_filter(raw_data, 3)
                
                past_image_1 = ndimage.median_filter(past_image_1, 3)
                future_image_1 = ndimage.median_filter(future_image_1, 3)
                past_image_2 = ndimage.median_filter(past_image_2, 3)
                future_image_2 = ndimage.median_filter(future_image_2, 3)

                raw_data = np.concatenate((past_image_2[np.newaxis, :, :],past_image_1[np.newaxis, :, :] ,raw_data[np.newaxis, :, :], \
                future_image_1[np.newaxis, :, :],future_image_2[np.newaxis, :, :]), axis=0)
                raw_data = np.median(raw_data, axis=0)
          # with open(file_path, 'r') as f:
          #   # Start from second recording, as the first two are corrupted
          #   for line in f.read().splitlines()[2:]:
          #     # print(line)
          #     raw_data = np.fromstring(line, dtype=float, sep='\t')
          #     # Change the range from [0-1000] to [0-255].

              file_data = np.round(raw_data*255/1000).astype(np.uint8)
              

              file_data = file_data.reshape((1,64,32))

              file_label = token_position_left(positions_i[int(file[:-4])])
              # print("directory: ",directory,"file_name: " ,file,"file_label: ",file_label)
              file_label = np.array([file_label])

              if data is None:
                data = file_data
              else:
                data = np.concatenate((data, file_data), axis=0)
              if labels is None:
                labels = file_label
              else:
                labels = np.concatenate((labels, file_label), axis=0)

      dataset[subject] = (data, labels)
  return dataset

def load_exp_i_right(path,preprocess=True):
  """
  Creates a numpy array for the data and labels.
  params:
  ------
  path    -- Data path.
  returns:
  -------
  A numpy array (data, labels).
  """

  dataset = {}

  for _, dirs, _ in os.walk(path):
    for directory in dirs:
      # each directory is a subject
      subject = directory
      data = None
      labels = None
      for _, _, files in os.walk(os.path.join(path, directory)):
        files = list_right
        # print(files)
        for file in files:
          # print(file)
          file_path = os.path.join(path, directory, file)
          with open(file_path, 'r') as f:
            # Start from second recording, as the first two are corrupted
            lines = f.read().splitlines()[2:]
            for i in range(5, len(lines) - 5):
                            
              raw_data = np.fromstring(lines[i], dtype=float, sep='\t').reshape(64, 32)
              
              if preprocess is True:
                past_image_1 = np.fromstring(lines[i-1], dtype=float, sep='\t').reshape(64, 32)
                future_image_1 = np.fromstring(lines[i+1], dtype=float, sep='\t').reshape(64, 32)
                past_image_2 = np.fromstring(lines[i-2], dtype=float, sep='\t').reshape(64, 32)
                future_image_2 = np.fromstring(lines[i+2], dtype=float, sep='\t').reshape(64, 32)
              
                # Spatio-temporal median filter 5x5x5
              
                raw_data = ndimage.median_filter(raw_data, 3)
                
                past_image_1 = ndimage.median_filter(past_image_1, 3)
                future_image_1 = ndimage.median_filter(future_image_1, 3)
                past_image_2 = ndimage.median_filter(past_image_2, 3)
                future_image_2 = ndimage.median_filter(future_image_2, 3)

                raw_data = np.concatenate((past_image_2[np.newaxis, :, :],past_image_1[np.newaxis, :, :] ,raw_data[np.newaxis, :, :], \
                future_image_1[np.newaxis, :, :],future_image_2[np.newaxis, :, :]), axis=0)
                raw_data = np.median(raw_data, axis=0)
              # Change the range from [0-1000] to [0-255].
              file_data = np.round(raw_data*255/1000).astype(np.uint8)
              file_data = file_data.reshape((1,64,32))

              file_label = token_position_right(positions_i[int(file[:-4])])
              # print("directory: ",directory,"file_name: " ,file,"file_label: ",file_label)
              file_label = np.array([file_label])

              if data is None:
                data = file_data
              else:
                data = np.concatenate((data, file_data), axis=0)
              if labels is None:
                labels = file_label
              else:
                labels = np.concatenate((labels, file_label), axis=0)

      dataset[subject] = (data, labels)
  return dataset

def load_exp_i_new(path,preprocess=True):
  """
  Creates a numpy array for the data and labels.
  params:
  ------
  path    -- Data path.
  returns:
  -------
  A numpy array (data, labels).
  """

  dataset = {}

  for _, dirs, _ in os.walk(path):
    for directory in dirs:
      # each directory is a subject
      subject = directory
      data = None
      labels = None
      max_val = []
      for _, _, files in os.walk(os.path.join(path, directory)):
        # print(files)
        for file in files:
          # print(file)
          file_path = os.path.join(path, directory, file)
          with open(file_path, 'r') as f:
            lines = f.read().splitlines()[2:]
            for i in range(3, len(lines) - 3):
                              
              raw_data = np.fromstring(lines[i], dtype=float, sep='\t').reshape(64, 32)
              
              if preprocess is True:
                past_image = np.fromstring(lines[i-1], dtype=float, sep='\t').reshape(64, 32)
                future_image = np.fromstring(lines[i+1], dtype=float, sep='\t').reshape(64, 32)
                
                # Spatio-temporal median filter 3x3x3
                raw_data = ndimage.median_filter(raw_data, 3)
                past_image = ndimage.median_filter(past_image, 3)
                future_image = ndimage.median_filter(future_image, 3)
                raw_data = np.concatenate((raw_data[np.newaxis, :, :], past_image[np.newaxis, :, :], future_image[np.newaxis, :, :]), axis=0)
                raw_data = np.median(raw_data, axis=0)
            
            # with open(file_path, 'r') as f:
            #   # Start from second recording, as the first two are corrupted
            #   lines = f.read().splitlines()[2:]
            #   for line in f.read().splitlines()[2:]:
            #     # print(line)
            #     raw_data = np.fromstring(line, dtype=float, sep='\t')
                # Change the range from [0-1000] to [0-255].
                  # max_val.append(np.amax(raw_data))
              file_data = np.round(raw_data*255/1000).astype(np.uint8)
              # file_data = np.round(raw_data).astype(np.uint8)
              
              file_data = file_data.reshape((1,64,32))
              # print(positions_i[int(file[:-4])])
              file_label = token_position_new(positions_i[int(file[:-4])])
              # print("directory: ",directory,"file_name: " ,file,"file_label: ",file_label)
              file_label = np.array([file_label])

              if data is None:
                data = file_data
              else:
                data = np.concatenate((data, file_data), axis=0)
              if labels is None:
                labels = file_label
              else:
                labels = np.concatenate((labels, file_label), axis=0)
      
      # max_over_all = max(max_val)
      # print(max_over_all)

      # data = np.round(data * 255/1000).astype(np.uint8)
      dataset[subject] = (data, labels)

  return dataset

def load_exp_ii(path):

  exp_ii_data_air = {}
  exp_ii_data_spo = {}

  # each directory is a subject
  for _, subject_dirs, _ in os.walk(path):
    for subject in subject_dirs:
      data = None
      labels = None

      # each directory is a matresss
      for _, mat_dirs, _ in os.walk(os.path.join(path, subject)):
        for mat in mat_dirs:
          for _, _, files in os.walk(os.path.join(path, subject, mat)):
            for file in files:
              file_path = os.path.join(path, subject, mat, file)
              raw_data = np.loadtxt(file_path)
              # Change the range from [0-500] to [0-255].
              file_data = np.round(raw_data*255/500).astype(np.uint8)
              
              file_data = resize_and_rotate(file_data)
              
              file_data = file_data.view(1, 64, 32)

              if file[-6] == "E" or file[-6] == "G":
                file_label = positions_ii[file[-6:-4]]
              else:
                file_label = positions_ii[file[-6]]

              file_label = token_position(file_label)
              file_label = np.array([file_label])

              if data is None:
                data = file_data
              else:
                data = np.concatenate((data, file_data), axis=0)

              if labels is None:
                labels = file_label
              else:
                labels = np.concatenate((labels, file_label), axis=0)

          if mat == "Air_Mat":
            exp_ii_data_air[subject] = (data, labels)
          else:
            exp_ii_data_spo[subject] = (data, labels)

          data = None
          labels = None

    return exp_ii_data_air, exp_ii_data_spo

class Mat_Dataset():
  def __init__(self,datasets, mats, Subject_IDs):

    self.samples = []
    self.labels = []

    for mat in mats:
      data = datasets[mat]
      self.samples.append(np.vstack([data.get(key)[0] for key in Subject_IDs]))
      self.labels.append(np.hstack([data.get(key)[1] for key in Subject_IDs]))

    self.samples = np.vstack(self.samples)
    self.labels = np.hstack(self.labels)

  def __len__(self):
    return self.samples.shape[0]

  def __getitem__(self, idx):
    return self.samples[idx], self.labels[idx]