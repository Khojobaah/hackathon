import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# Load your resized images
train_dir = '../images/train'
test_dir = '../images/test'

x_train = np.array(load_images_from_folder(train_dir))
x_test = np.array(load_images_from_folder(test_dir))

# Preprocess the dataset
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

# Creating the Autoencoder Model
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
batch_size = 128

history = autoencoder.fit(x_train_noisy, x_train, validation_data=(x_test_noisy, x_test),
                          epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

autoencoder.save('denoising_autoencoder_200.keras')
