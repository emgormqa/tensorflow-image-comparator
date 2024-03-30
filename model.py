import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# load training data
train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')

# normailize data
train_images = train_images / 255.0
test_images = test_images / 255.0

# create neural network
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# train model
model.fit(train_images, train_labels, epochs=20, batch_size=32, validation_data=(test_images, test_labels))

# save model
model.save('x_classifier.h5')
