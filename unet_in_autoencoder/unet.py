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



def build_unet(input_shape):
    inputs = L.Input(input_shape)

    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    b1 = conv_block(p4, 512)


    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    outputs = L.Conv2D(3, (1,1), padding='same', activation='sigmoid')(d4)

    model = Model(inputs, outputs, name='U-Net')
    return model