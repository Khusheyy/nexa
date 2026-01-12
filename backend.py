from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model_path = 'digit_model.h5'
if not os.path.exists(model_path):
    print(f"Warning: Model file '{model_path}' not found. Please train the model first.")
    model = None
else:
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Handle preflight request
        return '', 200
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        # Get Base64 image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Grayscale
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize (0-1)
        image_array = np.array(image).astype('float32') / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
        
        # Predict
        predictions = model.predict(image_array, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        return jsonify({'digit': int(predicted_digit), 'confidence': confidence})
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error in predict: {error_msg}")
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 400

if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000")
    print("Make sure the backend is running before using the frontend!")
    app.run(host='127.0.0.1', port=5000, debug=True)