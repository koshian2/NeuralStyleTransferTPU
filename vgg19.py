"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556)
"""

# Create your own VGG-19 to avoid layer name conflicts

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import layers
from tensorflow.keras import utils as keras_utils
from tensorflow.keras import models
import os
import numpy as np


WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')


def VGG19(input_shape, cnt,
          **kwargs):
    prefix = str(cnt)+"_"

    img_input = layers.Input(input_shape)
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=prefix+'block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=prefix+'block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block3_conv3')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=prefix+'block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block4_conv3')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=prefix+'block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block5_conv3')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv4')(x)                      
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=prefix+'block5_pool')(x)

    # Create model.
    model = models.Model(img_input, x, name='vgg19')

    # Don't train all layers
    for layer in model.layers:
        layer.trainable = False

    # Load weights.
    weights_path = keras_utils.get_file(
        'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
        WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        file_hash='253f8cb515780f3b799900260a226db6')
    model.load_weights(weights_path, by_name=True)

    return model

def extract_vgg_features(input_tensor, input_shape, cnt):
    # Create a vgg-19 with a new tensor
    model = VGG19(input_shape, cnt)
    # Intermediate layers of using loss functions
    content_loss = ["block5_conv2"]
    style_loss = ["block1_conv1", "block2_conv1", "block3_conv1",
                  "block4_conv1", "block5_conv1"]
    # Remap the graphs
    x = input_tensor
    contents, styles = [], []
    for i, l in enumerate(model.layers):
        if i == 0: continue
        l.trainable = False
        x = l(x)
        if any([x in l.name for x in content_loss]):
            contents.append(x)
        if any([x in l.name for x in style_loss]):
            styles.append(x)
    return [*contents, *styles]