from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__, static_folder='.',
            static_url_path='')
CORS(app)

model_path = 'digit_model.h5'
if not os.path.exists(model_path):
    print(
        f"Warning: Model file '{model_path}' not found. Please train the model first.")
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
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })


@app.route('/', methods=['GET'])
def serve_index():
    return send_from_directory('.', 'index.html')


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200

    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data = data['image'].split(
            ',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy array
        image_array = np.array(image)

        # CRITICAL FIX: Invert the image
        # Your canvas: BLACK digits on WHITE background
        # MNIST trained: WHITE digits on BLACK background
        # This line flips black <-> white to match training data
        image_array = 255 - image_array

        # Normalize to 0-1 range
        image_array = image_array / 255.0

        # Reshape for model input
        image_array = image_array.reshape(1, 28, 28, 1)

        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        probabilities = [float(p) for p in predictions[0]]

        return jsonify({
            'digit': int(predicted_digit),
            'confidence': confidence,
            'probabilities': probabilities
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error in predict: {error_msg}")
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 400


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'False') == 'True'

    print(f"Starting Flask server on http://{host}:{port}")
    print("Make sure the backend is running before using the frontend!")
    app.run(host=host, port=port, debug=debug)
