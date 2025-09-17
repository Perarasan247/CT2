import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.dataset import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy


# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize images (0–255 → 0–1)
x_train = x_train.reshape(-1,28,28,1).astype("float32") / 255.0
x_test = x_test.reshape(-1,28,28,1).astype("float32") / 255.0

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")  # 10 classes for digits
])

model.compile(optimizer="adam",
              loss=SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5, validation_split=0.1)

loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# Pick one test image
img = x_test[4]
plt.imshow(img.squeeze(), cmap="gray")
plt.title("Input Digit")
plt.show()

# Predict
pred = model.predict(np.expand_dims(img, axis=0))
print("Predicted digit:", pred.argmax())
