"""Trains or fine-tunes a Keras NN model to predict a steering angle from camera images"""
import argparse
import csv
import math
import cv2
import numpy as np
from keras import __version__ as keras_version
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Activation, Dropout, merge, Input, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import h5py


def load_image(base_path, image_path_from_capture):
    """Returns an RGB image from the CarND simulator.

    Arguments:
    base_path -- The simulation data folder in the current machine.
    image_path_from_capture -- The full path from the drive_log.csv file.
    """
    images_path = base_path + 'IMG/'
    filename = image_path_from_capture.split('\\')[-1]
    return cv2.cvtColor(cv2.imread(images_path + filename), cv2.COLOR_BGR2RGB)

def correct_angle(angle, offset):
    """Calculates an adjusted steering angle using the distnce between cameras.

    Arguments:
    angle -- The steering angle as measured by the central camera.
    offset -- The distance from the central camera to the secondary camera.
    """
    # =ATAN(Y/((Y/TAN(B*PI()/180))+W))*180/PI()
    b = 25 * angle  # The max angle (as displayed in the sim) times the proportional steer
    y = 75  # The distance to the horizon
    y_tan = y / math.tan(b * math.pi / 180) if abs(angle) > 0.00001 else 200000000
    return math.atan(y / (y_tan + offset)) * 180 / math.pi / 25


def train(data_root_path, model_target_name, process_one_in_x_images,
          source_model_file):
    """Trains a steering-angle prediction neural network from data in the CarND
    simulation output format.

    Arguments:
    data_root_path -- The path to the data to be used for training. Must have a
                      drive_log.csv file and an IMG folder with the images.
    model_target_name -- Base name of the target model. The file extension will
                         be 'hd5'. Intermediate model files may be created when
                         validation loss improves during training.
    process_one_in_x_images -- Value used to exclude examples from the training/
                               validation set. It speeds up the process to test
                               modifications to the model.
    source_model_file -- A model or weights file in h5 format that will be
                         used to get initial weights for the model. It is useful
                         to resume training, especially after new examples are
                         added.
    """
    captures = []

    print('Preprocessing  . . .')
    with open(data_root_path + 'driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for index, line in enumerate(reader):
            if (index % process_one_in_x_images) == 0:
                captures.append(line)

    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(captures, test_size=0.2)

    import sklearn

    def batch_len(array):
        """Shortcut to calculating the length of a batch from the generator based
        on the augmentations performed"""
        return len(array)*6  # (Center + Left + Right) * 2 (mirrored)

    def generator(samples, batch_size=32):
        """Generator function to return a number of example instances for
        training.

        Arguments:
        samples -- The full array of samples (X data and y result).
        batch_size -- The number of samples (before augmentation) that will be
                      returned by the generator. Samples are shuffled before
                      each new batch is generated.
        """
        def mirror_and_append(image, steering_angle):
            """Appends to the internal images array a normal and mirrored image
            and steering angle."""
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
                    # Center image
                    image = load_image(data_root_path, batch_sample[0])
                    mirror_and_append(image, batch_sample[3])
                    # Left
                    image = load_image(data_root_path, batch_sample[1])
                    mirror_and_append(image, correct_angle(float(batch_sample[3]), -65))
                    # Right
                    image = load_image(data_root_path, batch_sample[2])
                    mirror_and_append(image, correct_angle(float(batch_sample[3]), 65))

                X_data = np.array(images)
                y_data = np.array(steering_angles)
                yield sklearn.utils.shuffle(X_data, y_data)

    train_generator = generator(train_samples, batch_size=3)
    validation_generator = generator(validation_samples, batch_size=3)

    def _res_block(activation):
        """A residual network block with 32 filters in a bottleneck
        configuration, with 20% dropout on the input for a Keras model."""
        return merge((activation,
                      Convolution2D(32, 1, 1, activation='relu', bias=True)(
                          Convolution2D(16, 1, 3, activation='relu', bias=True, border_mode='same')(
                              Convolution2D(16, 3, 1, activation='relu', bias=True,
                                            border_mode='same')(
                                                Convolution2D(16, 1, 1, activation='relu',
                                                              bias=True)(
                                                                  Dropout(0.2)(activation)
                                                              )
                                            )
                          )
                      )
                     ), mode='sum')

    inputs = Input(shape=(160, 320, 3))
    x = Cropping2D(cropping=((60, 25), (0,0)))(inputs)
    x = BatchNormalization()(x)
    x = Convolution2D(32, 1, 1, activation='relu', bias=True)(x)
    x = _res_block(x)
    x = MaxPooling2D((2, 2))(x)
    x = _res_block(x)
    x = MaxPooling2D((2, 2))(x)
    x = _res_block(x)
    x = MaxPooling2D((2, 2))(x)
    x = _res_block(x)
    x = MaxPooling2D((2, 2))(x)
    x = _res_block(x)
    x = MaxPooling2D((2, 2))(x)
    x = _res_block(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    predictions = Dense(1)(x)

    model = Model(input=inputs, output=predictions)

    model.compile(loss='mse', optimizer='adam')

    if not source_model_file is None:
        model.load_weights(source_model_file)

    checkpoint = ModelCheckpoint(
        model_target_name + '_{val_loss:.4f}.h5',
        monitor='val_loss',
        verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    history = model.fit_generator(train_generator,
                                  samples_per_epoch=40000,
                                  validation_data=validation_generator,
                                  nb_val_samples=batch_len(validation_samples),
                                  nb_epoch=30,
                                  callbacks=callbacks_list)

    #print(history.history.keys())

    model.save(model_target_name + '.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving Training')
    parser.add_argument(
        'data_root_path',
        type=str,
        help='Path to the directory where the driving data was saved. It should contain a driving_log.csv file and IMG folder.'
    )
    parser.add_argument(
        'model_name',
        type=str,
        help='Base name of the model file to save to.'
    )
    parser.add_argument(
        '--keep_one_in',
        type=int,
        default=1,
        required=False,
        help='Factor used to drop some of the entries for learning.'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        required=False,
        help='Path to a model file that will be fine-tuned.'
    )
    args = parser.parse_args()
    data_path = args.data_root_path  # TODO: Verify it is a valid path
    model_name = args.model_name

    if args.base_model is not None:
        # Check that model Keras version is same as local Keras version
        f = h5py.File(args.base_model, mode='r')
        model_version = f.attrs.get('keras_version')
        keras_version = str(keras_version).encode('utf8')

        if model_version != keras_version:
            print('You are using Keras version ', keras_version,
                  ', but the model was built using ', model_version)

    train(data_path, model_name, args.keep_one_in, args.base_model)
