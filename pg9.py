import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Reshape data to include the channel dimension (28, 28, 1) for the Conv2D layer
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# 2. Build Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 3. Train
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

# A. Plot Accuracy & Loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()

# B. Show Predictions for first 5 test images
plt.figure(figsize=(10, 3))
for i in range(5):
    img = x_test[i]
    # Model predicts
    pred = np.argmax(model.predict(img.reshape(1, 28, 28, 1), verbose=0))

    plt.subplot(1, 5, i+1)
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {pred}")
    plt.axis('off')
plt.tight_layout()
plt.show()
