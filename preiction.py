import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the trained model
model = load_model('model_.h5')

# List of file paths for the new images
new_image_paths = [
    'data/normal/1_png.rf.18855f70ffc87f95ae31ecf208fa58be.jpg',
    'data/tumor/IMG-0004-00022IMG-0004-00021_png_jpg.rf.cdf6f688ac08c1592b12fc2c8ac24fbe.jpg',
    # Add more paths as needed
]

# Load and preprocess the new images
new_images = [image.load_img(img_path, target_size=(224, 224)) for img_path in new_image_paths]
new_images = [image.img_to_array(img) for img in new_images]
new_images = np.stack(new_images)
new_images = preprocess_input(new_images)

# Make predictions
predictions = model.predict(new_images)

# Decode the predictions
class_indices = {0: 'normal', 1: 'tumor'}  # Change this based on your actual class indices
decoded_predictions = [class_indices[np.argmax(pred)] for pred in predictions]

# Display the results
for img_path, pred in zip(new_image_paths, decoded_predictions):
    print(f"Image: {img_path} - Prediction: {pred}")