from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
import cv2

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


def preprocess_image(image):
    """
    Preprocess canvas image to match MNIST format exactly

    Args:
        image: PIL Image object (grayscale)

    Returns:
        Preprocessed numpy array ready for model
    """
    # Convert to numpy array
    img_array = np.array(image)

    # Invert: black on white -> white on black (like MNIST)
    img_array = 255 - img_array

    # Find the bounding box of non-zero pixels (the digit)
    coords = cv2.findNonZero(img_array)

    if coords is None:
        # Empty image - return blank 28x28
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)

    # Add small padding around the digit
    padding = 4
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_array.shape[1] - x, w + 2 * padding)
    h = min(img_array.shape[0] - y, h + 2 * padding)

    # Crop to bounding box
    cropped = img_array[y:y+h, x:x+w]

    # Calculate aspect ratio to fit in 20x20 (leaving 4px border for 28x28)
    aspect_ratio = w / h

    if aspect_ratio > 1:
        # Wider than tall
        new_width = 20
        new_height = max(1, int(20 / aspect_ratio))
    else:
        # Taller than wide
        new_height = 20
        new_width = max(1, int(20 * aspect_ratio))

    # Resize the cropped digit
    resized = cv2.resize(cropped, (new_width, new_height),
                         interpolation=cv2.INTER_AREA)

    # Create 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    # Calculate position to center the digit
    offset_x = (28 - new_width) // 2
    offset_y = (28 - new_height) // 2

    # Paste the digit in center
    canvas[offset_y:offset_y+new_height, offset_x:offset_x+new_width] = resized

    # Normalize to 0-1 range
    canvas = canvas.astype(np.float32) / 255.0

    # Reshape for model input
    canvas = canvas.reshape(1, 28, 28, 1)

    return canvas


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

        # Load and convert to grayscale
        image = Image.open(io.BytesIO(image_bytes)).convert('L')

        # Apply proper preprocessing (centering, padding, etc.)
        image_array = preprocess_image(image)

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
