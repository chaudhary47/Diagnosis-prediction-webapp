from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open('diagnosis_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    
scaler = model_data['scaler']
model = model_data['model']

feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [float(request.form[field]) for field in feature_names]
    
    # Preprocess
    features_array = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features_array)
    
    # Predict
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)[:, 1][0]
    
    result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"
    
    return render_template('result.html', 
                           prediction=result,
                           probability=f"{probability:.2%}")

if __name__ == '__main__':
    app.run(debug=True)