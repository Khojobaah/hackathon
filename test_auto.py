import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load your trained autoencoder model
autoencoder = load_model('autoencoder/denoising_autoencoder_200.keras')

# Function to load and preprocess image
def load_and_preprocess_image(image_path):
    # Load image using cv2
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

    # Normalize image to range [0, 1]
    image_normalized = image.astype('float32') / 255.0

    # Expand dimensions to match model input shape (add batch dimension)
    image_input = np.expand_dims(image_normalized, axis=0)
    image_input = np.expand_dims(image_input, axis=-1)  # Add channel dimension for grayscale

    return image_input

# Example usage: Replace 'path_to_your_image.png' with your image path
image_path = 'noise.png'

# Load and preprocess the image
input_image = load_and_preprocess_image(image_path)

# Predict using the autoencoder
predicted_image = autoencoder.predict(input_image)

# Post-process the predicted image if necessary (e.g., reshape, denormalize, etc.)
predicted_image = np.squeeze(predicted_image)  # Remove batch and channel dimensions
predicted_image = (predicted_image * 255).astype(np.uint8)  # Convert back to uint8 range [0, 255]

# Save the predicted image using cv2
cv2.imwrite('predicted_image.png', predicted_image)