from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from io import BytesIO
from PIL import Image

app = FastAPI(title="Animal Classifier API")

# 1. CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the Actual Trained ML Model
MODEL_PATH = "animal_classification_model.keras"
print("Loading trained model into Backend...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

# 3. Translation Dictionary (Locked via Main Chat)[cite: 1]
TRANSLATE_DICT = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant", 
    "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", 
    "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", 
    "ragno": "spider"
}

# Ordered list of classes matching the Kaggle training dataset indices[cite: 1]
CLASSES = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

def preprocess_image(image_bytes):
    """
    Real Preprocessing Pipeline (Matching the Main Chat Contract):
    - Load image from bytes
    - Resize: 224x224 pixels[cite: 1]
    - Convert to array
    - Normalization: Divide by 255.0 (0-1 range)[cite: 1]
    - Shape expansion: (1, 224, 224, 3)[cite: 1]
    """
    # Open image using PIL from byte stream[cite: 1]
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # Resize to model target size[cite: 1]
    img = img.resize((224, 224))
    
    # Convert to numpy array[cite: 1]
    img_array = img_to_array(img)
    
    # Normalize pixel values[cite: 1]
    img_array = img_array / 255.0
    
    # Expand dimensions to match model input shape (Batch size = 1)[cite: 1]
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch

# 4. The API Endpoint[cite: 1]
@app.post("/predict")
async def predict_animal(file: UploadFile = File(...)):
    print(f"Received file for prediction: {file.filename}")
    
    try:
        # Read uploaded file bytes[cite: 1]
        image_bytes = await file.read()
        
        # Step A: Preprocess the image according to contract[cite: 1]
        processed_image = preprocess_image(image_bytes)
        
        # Step B: Model Prediction[cite: 1]
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Get Italian label[cite: 1]
        predicted_italian = CLASSES[predicted_index]
        
        # Step C: Translate to English[cite: 1]
        predicted_english = TRANSLATE_DICT.get(predicted_italian, "unknown")
        
        # Returning the exact JSON contract locked in the Main Chat[cite: 1]
        response_data = {
            "success": True,
            "prediction": {
                "class_english": predicted_english,
                "class_italian": predicted_italian,
                "confidence": round(confidence, 2),
                "message": "Animal detected successfully!"
            }
        }
        return response_data

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {
            "success": False,
            "prediction": {
                "class_english": "error",
                "class_italian": "errore",
                "confidence": 0.0,
                "message": str(e)
            }
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)