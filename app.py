from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'placement_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'profile_encoder.pkl')

# Load the model and encoder
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    print("Model and Encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model or encoder: {e}")
    model = None
    encoder = None

@app.route('/')
def home():
    profiles = ['Creative', 'Finance', 'Management', 'Marketing', 'Technical']
    return render_template('index.html', profiles=profiles, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    profiles = ['Creative', 'Finance', 'Management', 'Marketing', 'Technical']
    
    if not model or not encoder:
        return render_template('index.html', 
                               profiles=profiles, 
                               prediction="Error", 
                               message="Machine learning models are not initialized.")
    
    try:
        # Get inputs
        cgpa = float(request.form.get('cgpa'))
        iq = int(request.form.get('iq'))
        profile_val = request.form.get('profile')
        
        # Validation
        if not (5.0 <= cgpa <= 10.0):
            raise ValueError("CGPA must be between 5.0 and 10.0")
        if not (70 <= iq <= 150):
            raise ValueError("IQ must be between 70 and 150")
            
        # Encode profile
        # Use transform on single label
        encoded_profile = encoder.transform([profile_val])[0]
        
        # Predict
        features = np.array([[cgpa, iq, encoded_profile]])
        prediction_val = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        
        # Calculate confidence
        if prediction_val == 1:
            confidence = round(probabilities[1] * 100, 2)
            result = "Placed"
        else:
            confidence = round(probabilities[0] * 100, 2)
            result = "Not Placed"
            
        return render_template('index.html',
                               profiles=profiles,
                               prediction=result,
                               confidence=confidence,
                               cgpa=cgpa,
                               iq=iq,
                               selected_profile=profile_val)
                               
    except Exception as e:
        return render_template('index.html',
                               profiles=profiles,
                               prediction="Error",
                               message=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
