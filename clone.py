
import argparse
from datetime import datetime
import csv
import cv2
import numpy as np
from keras import __version__ as keras_version
from keras.layers import Flatten, Dense, Activation, Dropout, merge, Input, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, load_model
import math
import h5py


def load_image(base_path, image_path_from_capture):
    images_path = base_path + 'IMG/'
    filename = image_path_from_capture.split('\\')[-1]
    return cv2.cvtColor(cv2.imread(images_path + filename), cv2.COLOR_BGR2RGB)

def correct_angle(angle, offset):
    # =ATAN(Y/((Y/TAN(B*PI()/180))+W))*180/PI()  # Beware TAN(0)!
    b = 25 * angle  # The max angle (as displayed in the sim) times the proportional steer
    y = 75  # The distance to the horizon
    y_tan = y / math.tan(b * math.pi / 180) if abs(angle) > 0.00001 else 200000000
    return math.atan(y / (y_tan + offset)) * 180 / math.pi / 25


def train(data_root_path, model_file):
    captures = []

    print('Preprocessing  . . .')
    with open(data_root_path + 'driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            captures.append(line)
    
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(captures, test_size=0.2)

    import sklearn

    def batch_len(array):
        """Shortcut to calculating the length of a batch from the generator based
        on the augmentations performed"""
        return len(array)*6  # (Center + Left + Right) * 2 (mirrored)

    def generator(samples, batch_size=32):
        def mirror_and_append(image, steering_angle):
            images.append(image)
            steering_angles.append(float(steering_angle))
            images.append(np.fliplr(image))
            steering_angles.append(-float(steering_angle))
        num_samples = len(samples)
        while 1:
            sklearn.utils.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]
                images = []
                steering_angles = []
                for batch_sample in batch_samples:
                    image = load_image(data_root_path, batch_sample[0])
                    mirror_and_append(image, batch_sample[3])
                    image = load_image(data_root_path, batch_sample[1]) # Left
                    mirror_and_append(image, correct_angle(float(batch_sample[3]), -35))
                    image = load_image(data_root_path, batch_sample[2]) # Right
                    mirror_and_append(image, correct_angle(float(batch_sample[3]), 35))

                X_data = np.array(images)
                y_data = np.array(steering_angles)
                yield sklearn.utils.shuffle(X_data, y_data)
    
    train_generator = generator(train_samples, batch_size=4)
    validation_generator = generator(validation_samples, batch_size=4)

    def _res_block(activation):
        return merge((activation, Dropout(0.2)(
            # TODO: Get number of features from a parameter
            Convolution2D(64, 1, 1, activation='relu', bias=True)(
                Dropout(0.2)(
                    Convolution2D(64, 1, 1,
                                  activation='relu', bias=True)(activation))))), mode='sum')

    if (model_file is None):
        inputs = Input(shape=(160, 320, 3))
        x = Cropping2D(cropping=((60, 25), (0,0)))(inputs)
        x = BatchNormalization()(x)
        # Initial convolution to fit the residual block
        x = Convolution2D(64, 1, 1, activation='relu', bias=True)(x)
        x = _res_block(x)
        x = MaxPooling2D((4, 4))(x)
        x = _res_block(x)
        x = MaxPooling2D((4, 4))(x)
        x = _res_block(x)
        x = MaxPooling2D((4, 4))(x)
        x = Flatten()(x)
        #x = Dropout(0.5)(x)  # Dropout right after maxpooling?
        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1)(x)

        model = Model(input=inputs, output=predictions)

        #TODO: Might want to attach to an RNN since they are good
        #      with sequences

        model.compile(loss='mse', optimizer='adam')

    else:
        model = load_model(model_file)

    #history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True,
    #          nb_epoch=20)
    history = model.fit_generator(train_generator, samples_per_epoch=
            batch_len(train_samples), validation_data=validation_generator,
            nb_val_samples=batch_len(validation_samples), nb_epoch=7)

    #print(history.keys())

    model.save(datetime.now().strftime('%Y%m%d%H%M') + '_model2.hd5')

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
        required=False,
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

