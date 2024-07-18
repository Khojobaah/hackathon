import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class ImageGenerator(Sequence):
    def __init__(self, images, batch_size, img_size, noise_factor=0.5, **kwargs):
        self.images = images
        self.batch_size = batch_size
        self.img_size = img_size
        self.noise_factor = noise_factor
        self.on_epoch_end()
        super().__init__(**kwargs)

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        resized_images = [cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC) for img in batch_images]
        resized_images = np.array(resized_images).astype('float32') / 255.0
        resized_images = np.expand_dims(resized_images, axis=-1)
        
        noisy_images = resized_images + self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=resized_images.shape)
        noisy_images = np.clip(noisy_images, 0., 1.)
        
        return noisy_images, resized_images

    def on_epoch_end(self):
        np.random.shuffle(self.images)

# Loading and Preprocessing the Dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Define the new size for resizing
new_size = (200, 200)

# Creating Image Generators
batch_size = 128
train_generator = ImageGenerator(x_train, batch_size, new_size)
test_generator = ImageGenerator(x_test, batch_size, new_size)

# Creating the Autoencoder Model #
input_shape = (200, 200, 1)
latent_dim = 128

# Encoder
inputs = Input(shape=input_shape)
x = Conv2D(32, kernel_size=3, strides=2, activation='relu', padding='same')(inputs)
x = Conv2D(64, kernel_size=3, strides=2, activation='relu', padding='same')(x)
x = Flatten()(x)
latent_repr = Dense(latent_dim)(x)

# Decoder
x = Dense(50 * 50 * 64)(latent_repr)
x = Reshape((50, 50, 64))(x)
x = Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same')(x)
decoded = Conv2DTranspose(1, kernel_size=3, strides=2, activation='sigmoid', padding='same')(x)

# Autoencoder model
autoencoder = Model(inputs, decoded)

# Compiling the Autoencoder Model
autoencoder.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')

# Adding Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training the Autoencoder
epochs = 20

history = autoencoder.fit(train_generator, validation_data=test_generator,
                          epochs=epochs, callbacks=[early_stopping])

autoencoder.save('denoising_autoencoder_200.keras')
