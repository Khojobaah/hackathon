import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

    

# Function to resize a batch of images
def resize_images(images, new_size):
    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        resized_images.append(resized_image)
    return np.array(resized_images)


# Loading and Preprocessing the Dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Resize the images
new_size = (200, 200)
x_train_resized = resize_images(x_train, new_size)
x_test_resized = resize_images(x_test, new_size)


# Preprocessing the Dataset
x_train_resized = x_train_resized.astype('float32') / 255.0
x_test_resized = x_test_resized.astype('float32') / 255.0
x_train = np.expand_dims(x_train_resized, axis=-1)
x_test = np.expand_dims(x_test_resized, axis=-1)


# Adding Random Noise to the Training Set
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


#  Creating the Autoencoder Model #
input_shape = (None, None, 1)
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

