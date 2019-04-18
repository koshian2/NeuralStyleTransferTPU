import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.contrib.tpu.python.tpu import keras_support

from PIL import Image
from vgg19 import extract_vgg_features
from keras.applications.vgg19 import preprocess_input
import os
import numpy as np

# Savng layer of a generated image
class GeneratedImages(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.image_params = self.add_weight(name="image_params",
                                shape=(int(input_shape[1]), int(input_shape[2]), int(input_shape[3])),
                                initializer="uniform", trainable=True)
        super().build(input_shape)

    def call(self, x):
        # Apply sigmoid function to the params
        x = K.sigmoid(self.image_params) * K.sign(K.abs(x)+1)
        # [0,1] -> [0, 255]
        x = 255*x
        # RGB -> BGR
        x = x[:, :, :, ::-1]
        # Convert to caffe color scale
        mean = K.variable(np.array([103.939, 116.779, 123.68], np.float32).reshape(1,1,1,3))
        return x - mean
        #return self.image_params * K.sign(K.abs(x)+1)


    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()

# Layer for calculating losses
class LossLayer(layers.Layer):
    def __init__(self, content_weight=0.025, style_weight=1.0, tv_weight=0.01, **kwargs):
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        """
        6 tensors for each images
        [content_features, style_feartures1, style_features2, ..., style_features5]
        3 images for inputs
        [content_image, style_image, generated_images]
        1 additional tensor for tv loss
        generated image raw
        """
        assert len(inputs) == 19 # 6 tensors x 3 images + 1 generated 
        content_features = inputs[0:6]
        style_features = inputs[6:12]
        generated_features = inputs[12:18]
        generated_image = inputs[18]
        # content loss
        loss_content = self.content_loss(content_features[0], generated_features[0])
        # style loss
        loss_style = self.style_loss(style_features[1:], generated_features[1:])
        # total_variation_loss
        loss_tv = self.total_variation_loss(generated_image)
        # total loss
        total_loss = self.content_weight * loss_content
        total_loss += self.style_weight * loss_style
        total_loss += self.tv_weight * loss_tv
        total_loss = K.reshape(total_loss, (-1, 1, 1, 1))

        # add last channel to the generated image(generated + loss)
        ones = K.sign(K.abs(generated_image)+1)
        ones = K.expand_dims(K.mean(ones, axis=-1), axis=-1)
        x = ones * total_loss
        x = K.concatenate([generated_image, x], axis=-1)
        print(x)
        return x

    def compute_output_shape(self, input_shapes):
        generated_shape = input_shapes[18]
        return (generated_shape[0], generated_shape[1], generated_shape[2], generated_shape[3]+1)

    def get_config(self):
        base_config = super().get_config()
        output = {
            **base_config,
            "content_weight": self.content_weight,
            "style_weight": self.style_weight,
            "tv_weight": self.tv_weight
        }
        return output

    @staticmethod
    def content_loss(img1, img2):
        assert K.int_shape(img1) == K.int_shape(img2)
        assert K.ndim(img1) == 4
        batch, height, width, channels = K.int_shape(img1)
        return K.sum((img1-img2)**2, axis=(1,2,3)) / height / width / channels / 2

    @staticmethod
    def gram_matrix(input):
        assert K.ndim(input) == 4
        # [batch, H, W, C] -> [batch, C, H, W]
        batch, height, width, channel = K.int_shape(input)
        x = K.permute_dimensions(input, (0, 3, 1, 2))
        # [batch, C, H*W]
        x = K.reshape(x, (-1, channel, height*width))
        # [batch, H*W, C]
        y = K.permute_dimensions(x, (0, 2, 1))
        # gram matrix [batch, C, C]
        return K.batch_dot(x, y)

    @staticmethod
    def style_loss(features1, features2):
        assert len(features1) == len(features2)
        assert len(features1) == 5
        # scaling factor of each layers
        w = 1.0 / len(features1)
        losses = 0
        for x1, x2 in zip(features1, features2):
            gram1 = LossLayer.gram_matrix(x1)
            gram2 = LossLayer.gram_matrix(x2)
            batch, height, width, channels = K.int_shape(x1)
            # style loss for a single layer
            sloss = K.sum((gram1-gram2)**2, axis=(1,2)) / 4.0 / height **2 / width ** 2 / channels ** 2
            losses += w * sloss
        return losses

    @staticmethod
    def total_variation_loss(img):
        assert K.ndim(img) == 4
        batch, height, width, channel = K.int_shape(img)
        a = K.square(img[:, 1:, :, :] - img[:, :-1, :, :])
        b = K.square(img[:, :, 1:, :] - img[:, :, :-1, :])
        x = K.sum(a, axis=(1,2,3)) + K.sum(b, axis=(1,2,3))
        return x / height / width / channel / 2
    
# Train model
def create_neural_style_transfer_train(img_width, img_height, img_channels=3):
    # Content image, Style image
    image_shape = (img_height, img_width, img_channels)
    input_content = layers.Input(image_shape)
    input_style = layers.Input(image_shape)
    # generated images
    generated = GeneratedImages(name="generated")(input_content) #この入力はダミー
    # VGG-19 features
    content_features = extract_vgg_features(input_content, image_shape, 0)
    style_features = extract_vgg_features(input_style, image_shape, 1)
    generated_features = extract_vgg_features(generated, image_shape, 2)
    # Loss funcation layer
    loss_inputs = [*content_features, *style_features, *generated_features, generated]
    x = LossLayer()(loss_inputs)

    model = Model(inputs=[input_content, input_style], outputs=x)
    return model

# identity loss
def identity_loss(dummy, joint):
    loss_values = joint[:, :, :, 3]
    return K.mean(loss_values)

def deprocess_image(input):
    assert input.ndim == 4
    x = input.copy()
    # Remove zero-center by mean pixel
    x[:, :, :, 0] += 103.939
    x[:, :, :, 1] += 116.779
    x[:, :, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

class SamplingCallback(Callback):
    def __init__(self, model, content_batch, style_batch, original_width, original_height):
        self.model = model
        self.content_batch = content_batch
        self.style_batch = style_batch
        self.original_size = (original_width, original_height)

    def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0:
            generated = self.model.predict([self.content_batch, self.style_batch])[:, :, :, :3]
            deprocess = deprocess_image(generated)
            if not os.path.exists("sampling"):
                os.mkdir("sampling")
            with Image.fromarray(deprocess[0]) as img:
                img_original_size = img.resize(self.original_size, Image.LANCZOS)
                img_original_size.save(f"sampling/epoch_{epoch:04}.png")

def train(content_path, style_path):
    # Store the original image size
    input_size = 256
    with Image.open(content_path) as content:
        original_width, original_height = content.size
        content_numpy = np.asarray(content.convert("RGB").resize(
                            (input_size, input_size), Image.LANCZOS), np.uint8)
    # Loading the style image
    with Image.open(style_path) as style:
        style_numpy = np.asarray(style.convert("RGB").resize(
                            (input_size, input_size), Image.LANCZOS), np.uint8)
    # To Batch
    batch_size = 8
    content_batch = np.expand_dims(content_numpy, axis=0) * np.ones((batch_size, 1, 1, 3), np.uint8)
    style_batch = np.expand_dims(style_numpy, axis=0) * np.ones((batch_size, 1, 1, 3), np.uint8)
    dummy_y = np.zeros((batch_size, input_size, input_size, 3), np.float32)
    # preprocessing
    content_batch = preprocess_input(content_batch)
    style_batch = preprocess_input(style_batch)

    # create model
    model = create_neural_style_transfer_train(input_size, input_size)
    # initialize generating layer with content image
    # [0,1] -> logit
    content_r = content_numpy.astype(np.float32) / 255.0
    content_logit = np.log(content_r / (1 - content_r + 1e-8))
    model.get_layer("generated").set_weights([content_logit])
    # compile
    model.compile(tf.train.AdamOptimizer(5e-2), identity_loss)

    # convert to tpu model
    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    # train
    cb = SamplingCallback(model, content_batch, style_batch, original_width, original_height)

    model.fit([content_batch, style_batch], dummy_y, callbacks=[cb], epochs=3000, batch_size=batch_size)
    model.save_weights("style_trainsfer_train.hdf5")

if __name__ == "__main__":
    K.clear_session()
    train("images/kiyomizu.jpg", "images/nue.jpg") # Content, Style

