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

---------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist  # Fixed import path
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize images (0–255 → 0–1)
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")

# Data augmentation for better generalization
datagen = ImageDataGenerator(
    rotation_range=10,      
    width_shift_range=0.1,   
    height_shift_range=0.1,   
    zoom_range=0.1,          
    shear_range=0.1,          
    fill_mode='nearest'      
)

# Improved model architecture
model = Sequential([
    # First convolutional block
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Second convolutional block
    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Third convolutional block
    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    Dropout(0.25),
    
    # Dense layers
    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")  # 10 classes for digits
])

# Compile model with better optimizer settings
model.compile(
    optimizer="adam",
    loss=SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Print model summary
model.summary()

# Callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.0001,
        verbose=1
    )
]

# Train model with data augmentation
print("\nTraining model...")
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    steps_per_epoch=len(x_train) // 32,
    epochs=20,  # Increased epochs
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Evaluate model
print("\nEvaluating model...")
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# Detailed evaluation
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Test prediction on multiple samples
def test_predictions(model, x_test, y_test, num_samples=10):
    """Test predictions on random samples"""
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        img = x_test[idx]
        true_label = y_test[idx]
        
        # Predict
        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
        pred_label = pred.argmax()
        confidence = pred.max()
        
        plt.subplot(2, 5, i + 1)
        plt.imshow(img.squeeze(), cmap="gray")
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.3f}', 
                 color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Test on random samples
print("\nTesting predictions on random samples:")
test_predictions(model, x_test, y_test)
