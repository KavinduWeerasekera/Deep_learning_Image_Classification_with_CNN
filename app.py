# app.py - Flask Backend
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import os
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# CIFAR-10 class names and emojis
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CLASS_EMOJIS = ['✈️', '🚗', '🐦', '🐱', '🦌', '🐕', '🐸', '🐴', '🚢', '🚛']

# Global variable to store the loaded model
model = None

def load_trained_model():
    """Load the trained model from .keras file"""
    global model
    try:
        # Try different possible model file names
        model_files = ['cifar10_model.keras', 'best_cifar10_model.keras', 'model.keras']
        
        for model_file in model_files:
            if os.path.exists(model_file):
                model = load_model(model_file)
                print(f"✅ Model loaded successfully from {model_file}!")
                print(f"Model has {model.count_params():,} parameters")
                return model
        
        print("❌ No trained model found. Please save your model first!")
        print("Use: model.save('cifar10_model.keras') after training")
        return None
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def preprocess_image(image_data):
    """Preprocess uploaded image for CIFAR-10 prediction with better handling"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get original size for logging
        orig_size = image.size
        print(f"Original image size: {orig_size}")
        
        # Better preprocessing for real-world images
        # 1. Center crop to square if needed
        width, height = image.size
        if width != height:
            # Crop to square (center crop)
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            image = image.crop((left, top, right, bottom))
        
        # 2. Resize to 32x32 with better resampling
        image = image.resize((32, 32), Image.Resampling.LANCZOS)
        
        # 3. Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # 4. Normalize to [0, 1] (same as CIFAR-10 training)
        img_array = img_array / 255.0
        
        # 5. Optional: Apply slight Gaussian blur to reduce high-freq noise
        # This helps with the domain gap between real photos and CIFAR-10
        from PIL import ImageFilter
        pil_img = Image.fromarray((img_array * 255).astype(np.uint8))
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        img_array = np.array(pil_img, dtype=np.float32) / 255.0
        
        # 6. Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"Preprocessed shape: {img_array.shape}")
        print(f"Value range: [{img_array.min():.3f}, {img_array.max():.3f}]")
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image classification requests with improved confidence handling"""
    global model
    
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please ensure cifar10_model.keras exists.'
        }), 500
    
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        # Preprocess the image
        processed_image = preprocess_image(data['image'])
        
        if processed_image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to process image'
            }), 400
        
        # Make prediction using your trained model
        predictions = model.predict(processed_image, verbose=0)
        raw_predictions = predictions[0]
        
        print(f"Raw predictions: {raw_predictions}")
        print(f"Max confidence: {np.max(raw_predictions):.3f}")
        print(f"Predicted class: {CLASS_NAMES[np.argmax(raw_predictions)]}")
        
        # Check if model is confident (max prob > threshold)
        max_confidence = np.max(raw_predictions)
        confidence_threshold = 0.3  # Adjust this based on your model's behavior
        
        # Format results
        results = []
        for i, confidence in enumerate(raw_predictions):
            results.append({
                'class': CLASS_NAMES[i],
                'emoji': CLASS_EMOJIS[i],
                'confidence': float(confidence)
            })
        
        # Sort by confidence (descending)
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Add uncertainty warning for low confidence predictions
        warning = None
        if max_confidence < confidence_threshold:
            warning = f"Low confidence prediction ({max_confidence:.1%}). This image might not match CIFAR-10 training data well."
        
        response_data = {
            'success': True,
            'predictions': results,
            'top_prediction': results[0],
            'max_confidence': float(max_confidence),
            'warning': warning
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/model_info')
def model_info():
    """Get model information"""
    global model
    
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    try:
        return jsonify({
            'architecture': 'CNN with 4 Conv2D layers + 2 Dense layers',
            'total_parameters': int(model.count_params()),
            'input_shape': [32, 32, 3],
            'output_classes': len(CLASS_NAMES),
            'classes': CLASS_NAMES,
            'model_loaded': True
        })
    except Exception as e:
        return jsonify({
            'error': f'Error getting model info: {str(e)}'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("🚀 Starting CIFAR-10 Flask App...")
    print("📁 Loading trained model...")
    
    # Load the model on startup
    load_trained_model()
    
    if model is None:
        print("\n⚠️  WARNING: No trained model loaded!")
        print("📝 To fix this:")
        print("   1. Train your model using your existing code")
        print("   2. Save it: model.save('cifar10_model.keras')")
        print("   3. Restart this Flask app")
        print("   4. The app will still run but predictions won't work\n")
    
    print("🌐 Starting server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)