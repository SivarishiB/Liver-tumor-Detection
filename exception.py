# Import Data Science Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import Xception
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")
# Metrics
from sklearn.metrics import classification_report, confusion_matrix

# Define constants
BATCH_SIZE = 32
IMAGE_SIZE = (320, 320)
dataset = "data/"
image_dir = Path(dataset)

# Get filepaths and labels
filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate filepaths and labels into DataFrame
image_df = pd.concat([filepaths, labels], axis=1)

# Display sample images with labels
random_index = np.random.randint(0, len(image_df), 16)
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10), subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    image = Image.open(image_df.Filepath[random_index[i]])
    ax.imshow(image)
    ax.set_title(image_df.Label[random_index[i]])
plt.tight_layout()
plt.show()

# Split data into train and test sets
train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)

# Image data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.xception.preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

validation_test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input)

# Adjust the data generators
train_images = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = validation_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Save class indices for future use
class_indices = train_images.class_indices
print("Class Indices:", class_indices)
with open('class_indices.txt', 'w') as file:
    for class_name, class_index in class_indices.items():
        file.write(f"{class_index}: {class_name}\n")

# Load the pre-trained Xception model
pretrained_model = Xception(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
pretrained_model.trainable = False  # Freeze the layers of Xception

# Build the model
inputs = pretrained_model.input
x = pretrained_model.output
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(len(class_indices), activation='softmax')(x)  # Dynamic output layer based on classes

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(
    optimizer=Adam(0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Calculate steps per epoch
steps_per_epoch = train_images.samples // BATCH_SIZE
validation_steps = val_images.samples // BATCH_SIZE

# Train the model with correctly set steps
history = model.fit(
    train_images,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_images,
    validation_steps=validation_steps,
    epochs=5
)

# Evaluate the model
results = model.evaluate(test_images, verbose=0)
print("Test Loss, Test Accuracy:", results)
model.save('model.h5')

# Predict and map the predictions to labels
pred = model.predict(test_images)
pred = np.argmax(pred, axis=1)
labels = dict((v, k) for k, v in train_images.class_indices.items())
pred_labels = [labels[k] for k in pred]

# Display random test images with true and predicted labels
random_index = np.random.randint(0, len(test_df), 15)
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(25, 15), subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    image = Image.open(test_df.Filepath.iloc[random_index[i]])
    ax.imshow(image)
    color = "green" if test_df.Label.iloc[random_index[i]] == pred_labels[random_index[i]] else "red"
    ax.set_title(f"True: {test_df.Label.iloc[random_index[i]]}\nPredicted: {pred_labels[random_index[i]]}", color=color)
plt.tight_layout()
plt.show()

# Plot accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()