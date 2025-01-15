import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

(x_train_10, y_train_10), (x_test_10, y_test_10) = cifar10.load_data()
x_train_10, x_test_10 = x_train_10 / 255.0, x_test_10 / 255.0
y_train_10 = to_categorical(y_train_10, 10)
y_test_10 = to_categorical(y_test_10, 10)

(x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data()
x_train_100, x_test_100 = x_train_100 / 255.0, x_test_100 / 255.0
y_train_100 = to_categorical(y_train_100, 100)
y_test_100 = to_categorical(y_test_100, 100)

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = x_train_10.shape[1:]
num_classes_10 = 10

model_10 = build_cnn_model(input_shape, num_classes_10)
history_10 = model_10.fit(x_train_10, y_train_10, epochs=20, batch_size=64, validation_data=(x_test_10, y_test_10))
test_loss_10, test_acc_10 = model_10.evaluate(x_test_10, y_test_10, verbose=2)
print(f"CIFAR-10 Test accuracy: {test_acc_10:.4f}")

num_classes_100 = 100
model_100 = build_cnn_model(input_shape, num_classes_100)
history_100 = model_100.fit(x_train_100, y_train_100, epochs=20, batch_size=64, validation_data=(x_test_100, y_test_100))
test_loss_100, test_acc_100 = model_100.evaluate(x_test_100, y_test_100, verbose=2)
print(f"CIFAR-100 Test accuracy: {test_acc_100:.4f}")

plt.plot(history_10.history['accuracy'], label='CIFAR-10 Training Accuracy')
plt.plot(history_10.history['val_accuracy'], label='CIFAR-10 Validation Accuracy')
plt.title('CIFAR-10 Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history_100.history['accuracy'], label='CIFAR-100 Training Accuracy')
plt.plot(history_100.history['val_accuracy'], label='CIFAR-100 Validation Accuracy')
plt.title('CIFAR-100 Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()











import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to the range 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
# Build and compile the CNN
cnn = build_cnn((32, 32, 3), 10)
cnn.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Train the CNN on CIFAR-10
history = cnn.fit(x_train, y_train, epochs=15, batch_size=64, validation_data=(x_test, y_test))
# Evaluate on the test data
test_loss, test_acc = cnn.evaluate(x_test, y_test)
print(f"CIFAR-10 Test Accuracy: {test_acc:.2f}")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Load CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# Normalize data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Build the model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:]),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(100, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
batch_size = 64
epochs = 100
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    validation_data=(x_test, y_test),
    epochs=epochs,
    verbose=1
)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")








