import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and scalers
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
MINMAX_PATH = os.path.join(BASE_DIR, 'minmaxscaler.pkl')
STAND_PATH = os.path.join(BASE_DIR, 'standscaler.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(MINMAX_PATH, 'rb') as f:
    ms = pickle.load(f)

with open(STAND_PATH, 'rb') as f:
    sc = pickle.load(f)

# Crop dictionary mapping class number to name and details
CROP_DETAILS = {
    1: {'name': 'Rice', 'emoji': '🌾', 'desc': 'Requires high rainfall, warm temperature, and clayey or loamy soil.'},
    2: {'name': 'Maize', 'emoji': '🌽', 'desc': 'Grows best in well-drained fertile soil with moderate rainfall.'},
    3: {'name': 'Jute', 'emoji': '🌿', 'desc': 'Requires warm and humid climate with alluvial soil.'},
    4: {'name': 'Cotton', 'emoji': '☁️', 'desc': 'Grows well in black soil with moderate rainfall and plenty of sunshine.'},
    5: {'name': 'Coconut', 'emoji': '🥥', 'desc': 'Thrives in tropical coastal areas with sandy loam soil.'},
    6: {'name': 'Papaya', 'emoji': '🍈', 'desc': 'Needs warm climate, well-drained soil, and moderate moisture.'},
    7: {'name': 'Orange', 'emoji': '🍊', 'desc': 'Prefers subtropical climate with well-drained slightly acidic to neutral soil.'},
    8: {'name': 'Apple', 'emoji': '🍎', 'desc': 'Grows best in cool temperate zones with loamy soil rich in organic matter.'},
    9: {'name': 'Muskmelon', 'emoji': '🍈', 'desc': 'Thrives in warm, dry weather with sandy loam fertile soil.'},
    10: {'name': 'Watermelon', 'emoji': '🍉', 'desc': 'Requires warm sunny weather with sandy, well-aerated soil.'},
    11: {'name': 'Grapes', 'emoji': '🍇', 'desc': 'Needs dry and warm weather during flowering and fruiting.'},
    12: {'name': 'Mango', 'emoji': '🥭', 'desc': 'Grows well in tropical and subtropical regions with well-drained soil.'},
    13: {'name': 'Banana', 'emoji': '🍌', 'desc': 'Requires warm, humid, tropical conditions with rich loamy soil.'},
    14: {'name': 'Pomegranate', 'emoji': '🍎', 'desc': 'Suitable for semi-arid areas with good drainage.'},
    15: {'name': 'Lentil', 'emoji': '🍲', 'desc': 'Cold-tolerant pulse crop that thrives in light loams.'},
    16: {'name': 'Blackgram', 'emoji': '🥣', 'desc': 'Grows in warm humid climates; tolerates various soil types.'},
    17: {'name': 'Mungbean', 'emoji': '🫘', 'desc': 'Warm season crop needing well-drained loam or sandy loam.'},
    18: {'name': 'Mothbeans', 'emoji': '🫘', 'desc': 'Extremely drought-resistant crop suitable for arid regions.'},
    19: {'name': 'Pigeonpeas', 'emoji': '🫘', 'desc': 'Drought-tolerant legume well-suited for tropical climates.'},
    20: {'name': 'Kidneybeans', 'emoji': '🫘', 'desc': 'Prefers cool to moderate weather with well-drained fertile soil.'},
    21: {'name': 'Chickpea', 'emoji': '🌰', 'desc': 'Cool-season legume requiring well-drained, sandy-clay loam.'},
    22: {'name': 'Coffee', 'emoji': '☕', 'desc': 'Grows on shaded hill slopes with high rainfall and rich volcanic or loamy soil.'}
}

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    # Prepare DataFrame matching training feature columns
    input_df = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]], columns=FEATURE_COLS)
    
    # Scale features using the trained scalers
    scaled_features = sc.transform(ms.transform(input_df))
    
    # Predict
    pred_class = int(model.predict(scaled_features)[0])
    
    # Probability scores
    probabilities = model.predict_proba(scaled_features)[0]
    
    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_recommendations = []
    for idx in top_indices:
        class_num = model.classes_[idx]
        conf = float(probabilities[idx]) * 100
        crop_info = CROP_DETAILS.get(class_num, {'name': f'Crop #{class_num}', 'emoji': '🌱', 'desc': ''})
        top_recommendations.append({
            'name': crop_info['name'],
            'emoji': crop_info['emoji'],
            'confidence': round(conf, 1),
            'desc': crop_info['desc']
        })
        
    recommended_crop = CROP_DETAILS.get(pred_class, {'name': 'Unknown', 'emoji': '🌱', 'desc': 'No description available'})
    return recommended_crop, top_recommendations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if request is JSON or form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        n = float(data.get('N', 0))
        p = float(data.get('P', 0))
        k = float(data.get('K', 0))
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        ph = float(data.get('ph', 0))
        rainfall = float(data.get('rainfall', 0))

        crop_info, top_crops = predict_crop(n, p, k, temperature, humidity, ph, rainfall)

        if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': True,
                'crop': crop_info['name'],
                'emoji': crop_info['emoji'],
                'description': crop_info['desc'],
                'top_recommendations': top_crops
            })

        return render_template('index.html',
                               prediction=crop_info,
                               top_recommendations=top_crops,
                               inputs={'N': n, 'P': p, 'K': k, 'temperature': temperature,
                                       'humidity': humidity, 'ph': ph, 'rainfall': rainfall})

    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'error': error_msg}), 400
        return render_template('index.html', error=error_msg)

if __name__ == '__main__':
    # Using port 5001 as 5000 is commonly used on macOS AirPlay receiver
    print("🌾 Crop Recommendation System is running at: http://127.0.0.1:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
