import keras

from keras import utils as np_utils

import numpy as np
import tensorflow as tf
from keras.models import Model
from functools import partial
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Concatenate, Activation, Input, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

def _generate_layer_name(name, branch_idx=None, prefix=None):
    if prefix is None:
        return None
    if branch_idx is None:
        return '_'.join((prefix, name))
    return '_'.join((prefix, 'Branch', str(branch_idx), name))

def load_data(file_name):
    attack_image = np.loadtxt("./CAN_DATA/"+file_name+"_attack_image.csv", delimiter=",")
    attack_free_image = np.loadtxt(file_name+"_attack_free_image.csv", delimiter=",")
    attack_image = attack_image.reshape(
        attack_image.shape[0], 29, 29, 1
    )
    attack_free_image = attack_free_image.reshape(
        attack_free_image.shape[0], 29, 29, 1
    )
    attack_y = np.array([[1]] * attack_image.shape[0])
    attack_free_y = np.array([[0]] * attack_free_image.shape[0])

    x_data = np.concatenate((attack_image, attack_free_image))
    y_data = np.concatenate((attack_y, attack_free_y))

    train_x = x_data[:int(x_data.shape[0] * 0.7)]
    test_x = x_data[int(x_data.shape[0] * 0.7):]
    train_y = y_data[:int(x_data.shape[0] * 0.7)]
    test_y = y_data[int(x_data.shape[0] * 0.7):]

    return (train_x, train_y), (test_x, test_y)


class Inception_Resnet_V1:
    def __init__(self, epochs=10, batch_size=128, load_weights=True):
        self.name = 'Inception_Resnet_V1'
        self.model_filename = 'networks/models/Inception_Resnet_V1.h5'
        self.num_classes = 2
        self.input_shape = 29, 29, 1
        self.batch_size = batch_size
        self.epochs = epochs
        self.iterations = 100
        self.weight_decay = 0.0005
        self.log_filepath = r'networks/models/Inception_Resnet_V1/'

        if load_weights:
            try:
                self._model = load_model(self.model_filename)
                print('Successfully loaded', self.name)
            except (ImportError, ValueError, OSError):
                print('Failed to load', self.name)

    def count_params(self):
        return self._model.count_params()

    def color_preprocessing(self, x_train, x_test):
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
            x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
        return x_train, x_test

    def scheduler(self, epoch):
        if epoch < 80:
            return 0.1
        if epoch < 150:
            return 0.01
        return 0.001

    def build_model(self, img_input):

        # Stem block

        net = Conv2D(32, 3, strides=1, padding='SAME',
                     name='Conv2d_1a_3x3',
                     kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(img_input)
        net = Conv2D(32, 3, strides=1, padding='VALID',
                     name='Conv2d_2a_3x3',
                     kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(net)

        net = MaxPooling2D(3, strides=2, padding='VALID',
                           name='MaxPool_3a_3x3')(net)
        net = Conv2D(64, 1, strides=1, padding='VALID',
                     name='Conv2d_3b_1x1',
                     kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(net)
        net = Conv2D(128, 3, strides=1, padding='SAME',
                     name='Conv2d_4a_3x3',
                     kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(net)
        net = Conv2D(128, 3, strides=1, padding='SAME',
                     name='Conv2d_4b_3x3',
                     kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(net)

        # Inception_Resnet_A

        name_fmt = partial(_generate_layer_name, prefix="Inception_Resnet_A")

        branch_0 = Conv2D(32, 1, strides=1, padding='SAME',
                          name=name_fmt("Conv2d_1x1", 0),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(net)

        branch_1 = Conv2D(32, 1, strides=1, padding='SAME',
                          name=name_fmt("Conv2d_0a_1x1", 1),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(net)
        branch_1 = Conv2D(32, 3, strides=1, padding='SAME',
                          name=name_fmt("Conv2d_0b_3x3", 1),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(branch_1)

        branch_2 = Conv2D(32, 1, strides=1, padding='SAME',
                          name=name_fmt("Conv2d_1a_1x1", 2),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(net)
        branch_2 = Conv2D(32, 3, strides=1, padding='SAME',
                          name=name_fmt("Conv2d_1b_3x3", 2),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='SAME',
                          name=name_fmt("Conv2d_1c_3x3", 2),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(branch_2)

        branches = [branch_0, branch_1, branch_2]

        mixed = Concatenate(axis=3, name=name_fmt('Concatenate'))(branches)

        up = Conv2D(128, 1, strides=1, padding='SAME', activation=None, use_bias=True,
                    kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay),
                    name=name_fmt('mixed'))(mixed)
        net += 0.17 * up  # scale
        net = Activation('relu', name=name_fmt('Activation'))(net)

        # Reduction_A
        name_fmt = partial(_generate_layer_name, prefix='Reduction_A')

        branch_0 = Conv2D(192, 3, strides=2, padding='VALID',
                          name=name_fmt('Conv2d_0a_3x3', 0),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(net)

        branch_1 = Conv2D(96, 1, strides=1, padding='SAME',
                          name=name_fmt("Conv2d_1a_1x1", 1),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(net)
        branch_1 = Conv2D(96, 3, strides=1, padding='SAME',
                          name=name_fmt("Conv2d_1b_3x3", 1),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(branch_1)
        branch_1 = Conv2D(128, 3, strides=2, padding='VALID',
                          name=name_fmt("Conv2d_1c_3x3", 1),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(branch_1)
        branch_2 = MaxPooling2D(3, strides=2, padding='VALID',
                                name=name_fmt("Maxpool_1a_3x3", 2))(net)

        branches = [branch_0, branch_1, branch_2]
        net = Concatenate(axis=3, name=name_fmt('Concatenate'))(branches)

        # Inception_Resnet_B
        name_fmt = partial(_generate_layer_name, prefix='Inception_Resnet_B')

        branch_0 = Conv2D(64, 1, strides=1, padding='SAME',
                          name=name_fmt("Conv2d_1x1", 0),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(net)

        branch_1 = Conv2D(64, 1, strides=1, padding='SAME',
                          name=name_fmt('Conv2d_0a_1x1', 1),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(net)
        branch_1 = Conv2D(64, [1, 3], strides=1, padding='SAME',
                          name=name_fmt('Conv2d_0b_1x7', 1),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(branch_1)
        branch_1 = Conv2D(64, [3, 1], strides=1, padding='SAME',
                          name=name_fmt('Conv2d_0c_7x1', 1),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(branch_1)

        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name=name_fmt('Concatenate'))(branches)

        up = Conv2D(448, 1, activation=None, use_bias=True,
                    strides=1, padding='SAME', name=name_fmt('mixed'),
                    kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(mixed)

        net += 0.10 * up  # scale
        net = Activation('relu', name=name_fmt('Activation'))(net)

        # Reduction_B
        name_fmt = partial(_generate_layer_name, prefix='Reduction_B')

        branch_0 = Conv2D(128, 1, strides=1, padding='SAME',
                          name=name_fmt("Conv2d_0a_1x1", 0),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(net)
        branch_0 = Conv2D(192, 3, strides=1, padding='VALID',
                          name=name_fmt("Conv2d_1a_3x3", 0),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(branch_0)

        branch_1 = Conv2D(128, 1, strides=1, padding='SAME',
                          name=name_fmt("Conv2d_1a_1x1", 1),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(net)
        branch_1 = Conv2D(128, 3, strides=1, padding='VALID',
                          name=name_fmt('Conv2d_1b_3x3', 1),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(branch_1)

        branch_2 = Conv2D(128, 1, strides=1, padding='SAME',
                          name=name_fmt("Conv2d_2a_1x1", 2),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(net)
        branch_2 = Conv2D(128, 3, strides=1, padding='SAME',
                          name=name_fmt("Conv2d_2b_3x3", 2),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(branch_2)
        branch_2 = Conv2D(128, 3, strides=1, padding='VALID',
                          name=name_fmt("Conv2d_2c_3x3", 2),
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(branch_2)

        branch_3 = MaxPooling2D(3, strides=1, padding='VALID',
                                name=name_fmt('MaxPool_1a_3x3'))(net)

        branches = [branch_0, branch_1, branch_2, branch_3]
        net = Concatenate(axis=3, name=name_fmt('Concatenate'))(branches)

        # Logits
        x = GlobalAveragePooling2D(name='AvgPool')(net)
        x = Dropout(0.2, name='Dropout')(x)
        x = Dense(self.num_classes, name='Logits')(x)
        x = Activation('softmax', name='Predictions')(x)

        return x

    def train(self, x_train, y_train, x_test, y_test):
        y_train = keras.utils.np_utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.np_utils.to_categorical(y_test, self.num_classes)

        img_input = Input(shape=(self.input_shape))
        output = self.build_model(img_input)
        Inception_Resnet_V1 = Model(img_input, output)
        Inception_Resnet_V1.summary()

        #         sgd = optimizers.SGD(lr = 0.1, momentum = 0.9, nesterov = True)
        #         Inception_Resnet_V1.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

        Inception_Resnet_V1.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        #         tb_cb = TensorBoard(log_dir=self.log_filepath, histogram_freq=0)
        #         change_lr = LearningRateScheduler(self.scheduler)
        #         checkpoint = ModelCheckpoint(self.model_filename,
        #                 monitor='val_loss', verbose=0, save_best_only= True, mode='auto')
        #         plot_callback = PlotLearning()
        #         cbks = [change_lr,tb_cb,checkpoint,plot_callback]
        filepath = 'Inception_Resnet_V1.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',
                                     verbose=1, save_best_only=True, mode='max')
        early = EarlyStopping(monitor='val_loss', mode='min', patience=4, restore_best_weights=True)
        callbacks_list = [checkpoint, early]
        # set data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=True,
                                     width_shift_range=0.125,
                                     height_shift_range=0.125,
                                     fill_mode='constant', cval=0.)

        # datagen.fit(x_train)

        Inception_Resnet_V1.fit_generator(datagen.flow(x_train, y_train, batch_size=self.batch_size),
                                          steps_per_epoch=self.iterations,
                                          epochs=self.epochs,
                                          callbacks=callbacks_list,
                                          validation_data=(x_test, y_test))
        Inception_Resnet_V1.save(self.model_filename)

        self._model = Inception_Resnet_V1
        self.param_count = self._model.count_params()

    def predict(self, img):

        return self._model.predict(img, batch_size=self.batch_size)

    def predict_one(self, img):
        return self.predict(img)[0]

    def accuracy(self):
        (x_train, y_train), (x_test, y_test) = load_data("DoS")
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        return self._model.evaluate(x_test, y_test, verbose=0)[1]
    
    
    def retrain(self, x_train, y_train, x_test, y_test):
        y_train = keras.utils.np_utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.np_utils.to_categorical(y_test, self.num_classes)

        img_input = Input(shape=(self.input_shape))
        output = self.build_model(img_input)
        Inception_Resnet_V1 = Model(img_input, output)
        Inception_Resnet_V1.summary()

        #         sgd = optimizers.SGD(lr = 0.1, momentum = 0.9, nesterov = True)
        #         Inception_Resnet_V1.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

        Inception_Resnet_V1.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        #         tb_cb = TensorBoard(log_dir=self.log_filepath, histogram_freq=0)
        #         change_lr = LearningRateScheduler(self.scheduler)
        #         checkpoint = ModelCheckpoint(self.model_filename,
        #                 monitor='val_loss', verbose=0, save_best_only= True, mode='auto')
        #         plot_callback = PlotLearning()
        #         cbks = [change_lr,tb_cb,checkpoint,plot_callback]
        filepath = 'Inception_Resnet_V1_retrain.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',
                                     verbose=1, save_best_only=True, mode='max')
        early = EarlyStopping(monitor='val_loss', mode='min', patience=4, restore_best_weights=True)
        callbacks_list = [checkpoint, early]
        # set data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=True,
                                     width_shift_range=0.125,
                                     height_shift_range=0.125,
                                     fill_mode='constant', cval=0.)

        # datagen.fit(x_train)

        Inception_Resnet_V1.fit_generator(datagen.flow(x_train, y_train, batch_size=self.batch_size),
                                          steps_per_epoch=self.iterations,
                                          epochs=self.epochs,
                                          callbacks=callbacks_list,
                                          validation_data=(x_test, y_test))
        Inception_Resnet_V1.save(self.model_filename)

        self._model = Inception_Resnet_V1
        self.param_count = self._model.count_params()
