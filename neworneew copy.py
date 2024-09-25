import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Split Dataset

# Define paths
dataset_path = r"C:\Users\parma\Downloads\Image Dataset of Indian Coins\Image Dataset of Indian Coins\Indian Coins Image Dataset\Indian Coins Image Dataset"
train_path = r"C:\Users\parma\Downloads\Image Dataset of Indian Coins\train"
val_path = r"C:\Users\parma\Downloads\Image Dataset of Indian Coins\val"
test_path = r"C:\Users\parma\Downloads\Image Dataset of Indian Coins\test"

# Create directories
for folder in ['train', 'val', 'test']:
    os.makedirs(os.path.join(train_path), exist_ok=True)
    os.makedirs(os.path.join(val_path), exist_ok=True)
    os.makedirs(os.path.join(test_path), exist_ok=True)

# Split dataset
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        random.shuffle(images)

        # Calculate splits
        train_count = int(len(images) * 0.7)
        val_count = int(len(images) * 0.15)

        # Create class directories
        os.makedirs(os.path.join(train_path, class_folder), exist_ok=True)
        os.makedirs(os.path.join(val_path, class_folder), exist_ok=True)
        os.makedirs(os.path.join(test_path, class_folder), exist_ok=True)

        for i, image in enumerate(images):
            src = os.path.join(class_path, image)
            if i < train_count:
                dst = os.path.join(train_path, class_folder, image)
            elif i < train_count + val_count:
                dst = os.path.join(val_path, class_folder, image)
            else:
                dst = os.path.join(test_path, class_folder, image)
            shutil.copy(src, dst)

print("Data split into train, validation, and test sets.")

# Step 2: Set Up Image Data Generators

# Define image dimensions and paths
img_height, img_width = 150, 150  # Adjust as needed
batch_size = 32

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse'  # Change if you use one-hot encoding
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse'
)

# Step 3: Build the CNN Model

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')  # Adjust for number of classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the Model

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10  # You can increase the number of epochs
)

# Step 5: Evaluate the Model

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')

# Step 6: Save the Model

model.save('coin_classification_model.h5')

# Step 7: Visualize Training History

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Model Accuracy')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')
plt.show()

# Step 8: Load the Model and Predict on a Sample Image

# Load the trained model
model = tf.keras.models.load_model('coin_classification_model.h5')

# Function to predict a single image
def predict_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return predicted_class

# Path to your sample image
sample_image_path = r"C:\Users\parma\Downloads\Image Dataset of Indian Coins\Image Dataset of Indian Coins\Indian Coins Image Dataset\50 (2).jpeg"  # Update with your sample image path

predicted_class = predict_image(sample_image_path)
print(f'Predicted class index: {predicted_class}')
