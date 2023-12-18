from modules.Tea import Tea, AdditivePooling
from keras.layers import Dropout, Activation, Flatten
from tensorflow.keras.layers import Layer, Lambda, Input, concatenate, Concatenate
import tensorflow as tf
from tensorflow_addons.layers import SpatialPyramidPooling2D as SPP

from tensorflow.keras.layers import Input,Conv2D, BatchNormalization, MaxPool2D, ReLU
from tensorflow.keras import Model

class TeaBlock(Layer):
    def __init__(self,
                 num_cls, dropout=False):
        """Initializes a new TeaLayer.

        Arguments:
            units -- The number of neurons to use for this layer."""
        super().__init__()
        self.dropout = dropout
        self.num_cls = num_cls

        self.core_1_01 = Tea(64)
        self.core_1_02 = Tea(64)
        self.core_1_03 = Tea(64)
        self.core_1_04 = Tea(64)
        self.core_1_05 = Tea(64)
        self.core_1_06 = Tea(64)
        self.core_1_07 = Tea(64)
        self.core_1_08 = Tea(64)
        self.core_1_09 = Tea(64)
        self.core_1_10 = Tea(64)
        self.core_1_11 = Tea(64)
        self.core_1_12 = Tea(64)
        self.core_1_13 = Tea(64)
        self.core_1_14 = Tea(64)
        self.core_1_15 = Tea(64)
        self.core_1_16 = Tea(64)

        self.core_2_01 = Tea(64)
        self.core_2_02 = Tea(64)
        self.core_2_03 = Tea(64)
        self.core_2_04 = Tea(64)

        num_out_node = round(256/num_cls)*num_cls

        self.core_3_01 = Tea(num_out_node)

        self.AdditivePooling = AdditivePooling(num_classes=self.num_cls)
    
    def get_config(self):

      config = super().get_config()
      config.update({
          "num_cls"   : self.num_cls,
          "dropout_rance": self.dropout,
          "core_1_01" : self.core_1_01,
          "core_1_02" : self.core_1_02,
          "core_1_03" : self.core_1_03,
          "core_1_04" : self.core_1_04,
          "core_1_05" : self.core_1_05,
          "core_1_06" : self.core_1_06,
          "core_1_07" : self.core_1_07,
          "core_1_08" : self.core_1_08,
          "core_1_09" : self.core_1_09,
          "core_1_10" : self.core_1_10,
          "core_1_11" : self.core_1_11,
          "core_1_12" : self.core_1_12,
          "core_1_13" : self.core_1_13,
          "core_1_14" : self.core_1_14,
          "core_1_15" : self.core_1_15,
          "core_1_16" : self.core_1_16,

          "core_2_01" : self.core_2_01,
          "core_2_02" : self.core_2_02,
          "core_2_03" : self.core_2_03,
          "core_2_04" : self.core_2_04,

          "core_3_01" : self.core_3_01,

          "AdditivePooling": self.AdditivePooling

      })
      return config

    def call(self, x):
        flattened_inputs = Flatten()(x)

        flattened_inputs_1 = Lambda(lambda x: x[:, : 2176])(flattened_inputs)

        x1_1 = Lambda(lambda x: x[:, :256])(flattened_inputs_1)
        x2_1 = Lambda(lambda x: x[:, 128: 384])(flattened_inputs_1)
        x3_1 = Lambda(lambda x: x[:, 256: 512])(flattened_inputs_1)
        x4_1 = Lambda(lambda x: x[:, 384: 640])(flattened_inputs_1)
        x5_1 = Lambda(lambda x: x[:, 512: 768])(flattened_inputs_1)
        x6_1 = Lambda(lambda x: x[:, 640: 896])(flattened_inputs_1)
        x7_1 = Lambda(lambda x: x[:, 768: 1024])(flattened_inputs_1)
        x8_1 = Lambda(lambda x: x[:, 896: 1152])(flattened_inputs_1)
        x9_1 = Lambda(lambda x: x[:, 1024: 1280])(flattened_inputs_1)
        x10_1 = Lambda(lambda x: x[:, 1152: 1408])(flattened_inputs_1)
        x11_1 = Lambda(lambda x: x[:, 1280: 1536])(flattened_inputs_1)
        x12_1 = Lambda(lambda x: x[:, 1408: 1664])(flattened_inputs_1)
        x13_1 = Lambda(lambda x: x[:, 1536: 1792])(flattened_inputs_1)
        x14_1 = Lambda(lambda x: x[:, 1664: 1920])(flattened_inputs_1)
        x15_1 = Lambda(lambda x: x[:, 1792: 2048])(flattened_inputs_1)
        x16_1 = Lambda(lambda x: x[:, 1820: 2176])(flattened_inputs_1)

        x1_1 = self.core_1_01(x1_1)
        x2_1 = self.core_1_02(x2_1)
        x3_1 = self.core_1_03(x3_1)
        x4_1 = self.core_1_04(x4_1)
        x5_1 = self.core_1_05(x5_1)
        x6_1 = self.core_1_06(x6_1)
        x7_1 = self.core_1_07(x7_1)
        x8_1 = self.core_1_08(x8_1)
        x9_1 = self.core_1_09(x9_1)
        x10_1 = self.core_1_10(x10_1)
        x11_1 = self.core_1_11(x11_1)
        x12_1 = self.core_1_12(x12_1)
        x13_1 = self.core_1_13(x13_1)
        x14_1 = self.core_1_14(x14_1)
        x15_1 = self.core_1_15(x15_1)
        x16_1 = self.core_1_16(x16_1)

        x1_1 = concatenate(([x1_1, x2_1, x3_1, x4_1]), axis=1)
        x2_1 = concatenate(([x5_1, x6_1, x7_1, x8_1]), axis=1)
        x3_1 = concatenate(([x9_1, x10_1, x11_1, x12_1]), axis=1)
        x4_1 = concatenate(([x13_1, x14_1, x15_1, x16_1]), axis=1)

        if self.dropout:
            x1_1 = Dropout(self.dropout)(x1_1)
            x2_1 = Dropout(self.dropout)(x2_1)
            x3_1 = Dropout(self.dropout)(x3_1)
            x4_1 = Dropout(self.dropout)(x4_1)

        x1_1 = self.core_2_01(x1_1)
        x2_1 = self.core_2_02(x2_1)
        x3_1 = self.core_2_03(x3_1)
        x4_1 = self.core_2_04(x4_1)

        x_out_1 = Concatenate(axis=1)([x1_1, x2_1, x3_1, x4_1])

        if self.dropout:
            x_out_1 = Dropout(self.dropout)(x_out_1)

        x_out_1 = self.core_3_01(x_out_1)

        x_out = self.AdditivePooling(x_out_1)

        predictions = Activation('softmax')(x_out)

        return predictions


def baseline_spp(input_shape=(64, 32, 1), num_cls=17, dropout_ranc=False):
    inputs = Input(shape=input_shape)

    x = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding="same")(inputs)
    x = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding="same")(x)
    x = BatchNormalization(axis=3)(x)
    x = tf.keras.activations.relu(x)

    x = MaxPool2D((2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=128, kernel_size=3, strides=(2, 2), padding="same")(x)
    x = tf.keras.activations.sigmoid(x)

    x = SPP(bins=[1, 4], data_format='channels_last')(x)

    outputs = TeaBlock(num_cls=num_cls, dropout=dropout_ranc)(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def baseline(input_shape=(64, 32, 1), num_cls=17, dropout_ranc=False):
    inputs = Input(shape=input_shape)

    x = Conv2D(filters=8, kernel_size=3, strides=(1, 1), padding="same")(inputs)
    x = Conv2D(filters=16, kernel_size=3, strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.activations.sigmoid(x)
    x = tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same')(x)

    outputs = TeaBlock(num_cls=num_cls, dropout=dropout_ranc)(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

