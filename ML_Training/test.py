import os
import numpy as np
import tensorflow as tf
# Modern Keras method to load images (Error-free)
from tensorflow.keras.utils import load_img, img_to_array

# 1. Load the Model
model_path = 'animal_classification_model.keras'
print("Loading model...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# 2. Image Preprocessing (Backend Contract)
img_path = 'test_image.jpg' # Tera local test image

# Resize to 224x224
img = load_img(img_path, target_size=(224, 224))

# Convert to Array
img_array = img_to_array(img)

# Normalize (0 to 1)
img_array = img_array / 255.0

# Add Batch Dimension -> Shape becomes (1, 224, 224, 3)
img_batch = np.expand_dims(img_array, axis=0)

# 3. Model Prediction
predictions = model.predict(img_batch)
predicted_index = np.argmax(predictions[0])
confidence = np.max(predictions[0])

# 4. Classes & Translation Logic
# Kaggle wali 10 classes
classes = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

# Tera translation dictionary
translate = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant", 
    "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", 
    "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", 
    "ragno": "spider"
}

# Final Output Mapping
predicted_italian = classes[predicted_index]
predicted_english = translate.get(predicted_italian, predicted_italian)

print("\n--- TEST RESULTS ---")
print(f"Italian Label: {predicted_italian}")
print(f"English Label: {predicted_english}")
print(f"Confidence: {confidence:.2f}")