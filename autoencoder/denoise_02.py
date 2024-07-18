
import os
import numpy as np
import cv2
from keras.utils import Sequence
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


class CustomImageGenerator(Sequence):
    def __init__(self, folder, batch_size, is_training=True, **kwargs):
        self.folder = folder
        self.batch_size = batch_size
        self.is_training = is_training
        self.image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.on_epoch_end()
        super().__init__(**kwargs)

    def __len__(self):
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.image_files[index * self.batch_size:(index + 1) * self.batch_size]
        images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in batch_files]
        images = np.array(images).astype('float32') / 255.0
        images = np.expand_dims(images, axis=-1)
        if self.is_training:
            noise_factor = 0.5
            images_noisy = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
            images_noisy = np.clip(images_noisy, 0., 1.)
            return images_noisy, images
        else:
            return images, images

    def on_epoch_end(self):
        np.random.shuffle(self.image_files)

# Define the directories for train and test datasets
train_folder = '../images/train'
test_folder = '../images/test'

# Define image size and batch size
img_size = (200, 200)
batch_size = 256

# Create custom generators
train_generator = CustomImageGenerator(train_folder, batch_size, is_training=True)
test_generator = CustomImageGenerator(test_folder, batch_size, is_training=False)

# Define the autoencoder model
input_shape = (200, 200, 1)
latent_dim = 128

# Encoder
inputs = Input(shape=input_shape)
x = Conv2D(32, kernel_size=3, strides=2, activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = Conv2D(64, kernel_size=3, strides=2, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Flatten()(x)
latent_repr = Dense(latent_dim)(x)

# Decoder
x = Dense(50 * 50 * 64)(latent_repr)
x = Reshape((50, 50, 64))(x)
x = Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
decoded = Conv2DTranspose(1, kernel_size=3, strides=2, activation='sigmoid', padding='same')(x)

# Autoencoder model
autoencoder = Model(inputs, decoded)

# Compile the autoencoder model
autoencoder.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')

# Adding early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training the autoencoder
epochs = 50

history = autoencoder.fit(train_generator, validation_data=test_generator, epochs=epochs, callbacks=[early_stopping])

# Save the trained model
autoencoder.save('denoising_autoencoder_200.keras')

