# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from keras import layers, models 


# Step 1: Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Step 2: Preprocess the data
# Normalize the images to be between 0 and 1 by dividing by 255
train_images, test_images = train_images / 255.0, test_images / 255.0

# Step 3: Build a Sequential Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images to 1D vector of 784 elements
    layers.Dense(128, activation='relu'),  # Dense layer with 128 neurons and ReLU activation
    layers.Dropout(0.2),  # Dropout layer to reduce overfitting
    layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (for the 10 digits) and softmax activation
])

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # For multi-class classification
              metrics=['accuracy'])

# Step 5: Train the model
model.fit(train_images, train_labels, epochs=5)

# Step 6: Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")
