import argparse
from datetime import datetime
import csv
import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D 
from keras.layers.normalization import BatchNormalization
import h5py
from keras import __version__ as keras_version

def train(data_root_path, model_file):
  captures = []
  with open(data_root_path + 'driving_log.csv') as csv_file:
      reader = csv.reader(csv_file)
      for line in reader:
        captures.append(line)
        #if len(captures) > 300:
        #  break

  images = []
  steering_angles = []
  #for capture in captures:
  #  panoramic = None
  #  for view in range(3):
  #    source_path = capture[view]
  #    filename = source_path.split('\\')[-1]
  #    image_path = data_root_path + 'IMG/' + filename
  #    image = cv2.imread(image_path)
  #    if panoramic is None:
  #      panoramic = image
  #    else:
  #      panoramic = np.concatenate((panoramic, image), axis = 1)
  #  # TODO: Augment (mirror) data
  #  images.append(panoramic)
  #  steering_angles.append(float(capture[3]))
  for capture in captures:
    source_path = capture[0]
    filename = source_path.split('\\')[-1]
    image_path = data_root_path + 'IMG/' + filename
    image = cv2.imread(image_path)
    #images.append(image)
    #steering_angles.append(float(capture[3]))
    # TODO: Use left and right images with correction factor for angle
    # TODO: Augment (mirror) data
    images.append(np.fliplr(image))
    steering_angles.append(-float(capture[3]))

  X_train = np.array(images)
  y_train = np.array(steering_angles)

  # TODO: Load a model if provided, to finetune
  if (model_file == ''):
    model = Sequential()
    model.add(BatchNormalization(input_shape = (160, 320, 3)))
    model.add(Convolution2D(32, 1, 1, activation = 'relu', bias = True))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 1, 1, activation = 'relu', bias = True))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((5, 5)))
    model.add(Convolution2D(128, 1, 1, activation = 'relu', bias = True))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 1, 1, activation = 'relu', bias = True))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((5, 5)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    #TODO: Might want to attach to an RNN since they are good with sequences

    model.compile(loss = 'mse', optimizer = 'adam')

  else:
    model = load_model(model_file)

  model.fit(X_train, y_train, validation_split = 0.2, shuffle = True,
            nb_epoch = 20)

  model.save(datetime.now().strftime('%Y%m%d%H%M') + '_model.hd5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving Training')
    parser.add_argument(
        'data_root_path',
        type=str,
        help='Path to the directory where the driving data was saved. It should contain a driving_log.csv file and IMG folder.'
    )
    parser.add_argument(
        '--base_model',
        type=str,
#        nargs='?',
#        default='',
        required = False,
        help='Path to a model file that will be fine-tuned.'
    )
    args = parser.parse_args()
    data_path = args.data_root_path  # TODO: Verify it is a valid path

    if (args.base_model is not None):
      # check that model Keras version is same as local Keras version
      f = h5py.File(args.base_model, mode='r')
      model_version = f.attrs.get('keras_version')
      keras_version = str(keras_version).encode('utf8')

      if model_version != keras_version:
          print('You are using Keras version ', keras_version,
                ', but the model was built using ', model_version)

    train(data_path, args.base_model)
