import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load and preprocess MNIST data
(x_train, _), (x_test, _) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)

# Define the stacked autoencoder
input_dim = x_train.shape[1]
encoding_dim = 128

input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(512, activation='relu')(input_layer)
encoded = layers.Dense(256, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
decoded = layers.Dense(256, activation='relu')(encoded)
decoded = layers.Dense(512, activation='relu')(decoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256)

# Get encoded images
encoded_imgs = encoder.predict(x_test)
# Visualize encoded images

n = 10  # Number of samples to visualize
plt.figure(figsize=(20, 4))
for i in range(n):
    plt.subplot(1, n, i + 1)
    plt.imshow(encoded_imgs[i].reshape((16, 8)), cmap="gray")  # Adjust based on encoding_dim
    plt.axis('off')
    plt.title(f"Encoded {i + 1}")
plt.show()


# Reconstruct images
decoded_imgs = autoencoder.predict(x_test)
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    for j, img in enumerate([x_test, decoded_imgs]):
        plt.subplot(2, n, i + 1 + j * n)
        plt.imshow(img[i].reshape(28, 28), cmap="gray")
        plt.axis('off')
        plt.title("Original" if j == 0 else "Reconstructed")
plt.show()

# Calculate and print reconstruction accuracy
threshold = 0.5
accurate_pixels = np.mean(np.abs(x_test - decoded_imgs) < threshold)
accuracy = accurate_pixels * 100
print(f"Reconstruction Accuracy: {accuracy:.2f}%")
