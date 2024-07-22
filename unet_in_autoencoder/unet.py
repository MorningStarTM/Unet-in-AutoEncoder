import tensorflow as tf
from keras.models import Model
import keras.layers as L
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.optimizers import Adam


def conv_block(input, num_filters):
    x = L.Conv2D(num_filters, 3, padding='same')(input)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)

    return x


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = L.MaxPool2D((2,2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    x = L.Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(input)
    x = L.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x



