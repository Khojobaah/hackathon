import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence


    
class ImageDataGenerator(Sequence):
    def __init__(self, folder, batch_size, img_size, **kwargs):
        self.folder = folder
        self.batch_size = batch_size
        self.img_size = img_size
        self.image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.on_epoch_end()
        super().__init__(**kwargs)  # Call the base class constructor

    def __len__(self):
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.image_files[index * self.batch_size:(index + 1) * self.batch_size]
        images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in batch_files]
        images = [cv2.resize(img, self.img_size) for img in images]
        images = np.array(images).astype('float32') / 255.0
        images = np.expand_dims(images, axis=-1)
        return images, images

    def on_epoch_end(self):
        np.random.shuffle(self.image_files)

# Loading and Preprocessing the Dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Resize the images
new_size = (200, 200)
x_train = x_train
x_test = x_test


# Preprocessing the Dataset
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)


# Adding Random Noise to the Training Set
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


#  Creating the Autoencoder Model #
input_shape = (200, 200, 1)
latent_dim = 128

# Encoder
inputs = Input(shape=input_shape)
x = Conv2D(32, kernel_size=3, strides=2, activation='relu', padding='same')(inputs)
x = Conv2D(64, kernel_size=3, strides=2, activation='relu', padding='same')(x)
x = Flatten()(x)
latent_repr = Dense(latent_dim)(x)

# Decoder
x = Dense(7 * 7 * 64)(latent_repr)
x = Reshape((7, 7, 64))(x)
x = Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same')(x)
decoded = Conv2DTranspose(1, kernel_size=3, strides=2, activation='sigmoid', padding='same')(x)

# Autoencoder model
autoencoder = Model(inputs, decoded)

# ----------------- #

# Compiling the Autoencoder Model
autoencoder.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')

# Adding Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training the Autoencoder
epochs = 20
batch_size = 128

history = autoencoder.fit(x_train_noisy, x_train, validation_data=(x_test_noisy, x_test),
                          epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

autoencoder.save('denoising_autoencoder_200.keras')
# keras.saving.save_model(autoencoder, 'denoising_autoencoder_01_.keras')

