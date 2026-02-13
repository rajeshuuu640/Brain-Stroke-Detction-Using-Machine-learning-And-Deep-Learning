import os
from flask import Flask, request, jsonify
import tempfile
import sys
from werkzeug.utils import secure_filename

# --- CRITICAL PATH FIX ---
# Since the API runs independently, we need to manually add the project root to path
# This assumes the script is run from the project root OR via 'python src/api/rest_service.py'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    # Import core prediction logic from the inference module
    from src.inference.predictor import load_models, predict_and_explain
except ModuleNotFoundError as e:
    print(f"ERROR: Could not import core modules. Details: {e}")
    # Fallback exit if core prediction modules are missing
    sys.exit(1)

# --- FLASK APPLICATION SETUP ---
app = Flask(__name__)

# Define the allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load models once when the API starts (uses predictor.py's safe loading logic)
CNN_MODEL, ENHANCEMENT_MODEL = load_models(model_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models')))

if CNN_MODEL is True:
    print("API initialized with MOCK model. Ready to accept requests.")
else:
    print("API initialized with real model. Ready to accept requests.")

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict_stroke', methods=['POST'])
def predict_stroke_api():
    """
    Main API endpoint for stroke diagnosis.
    Expects a file named 'image' in the POST request.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part in the request'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400
        
    if file and allowed_file(file.filename):
        
        # Save the uploaded file temporarily to disk
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + filename.rsplit('.', 1)[1]) as tmp:
            file.save(tmp.name)
            temp_file_path = tmp.name

        try:
            # 1. Run core prediction logic
            predicted_label, confidence, _ = predict_and_explain(
                temp_file_path, 
                CNN_MODEL, 
                ENHANCEMENT_MODEL
            )

            # 2. Return JSON response for IoT/Mobile system
            response = {
                'status': 'success',
                'diagnosis': predicted_label,
                'confidence': round(float(confidence), 4),
                'message': 'Diagnosis complete. Seek medical advice.'
            }
            return jsonify(response)

        except Exception as e:
            print(f"API Prediction Error: {e}")
            return jsonify({'error': f'Internal analysis error: {e}'}), 500
        
        finally:
            # Clean up the temporary file
            os.remove(temp_file_path)
    
    return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG are allowed.'}), 400

# --- RUNNING THE SERVICE ---
if __name__ == '__main__':
    # Running on port 5000 is standard for Flask APIs
    print("--- Starting Stroke Detection API Service ---")
    print("Accessible via POST request to: http://127.0.0.1:5000/predict_stroke")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)