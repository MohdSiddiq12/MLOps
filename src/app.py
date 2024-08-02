from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the model
model_path = ('models\RandomForest.pkl')
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from request
            data = request.json

            # Create DataFrame with expected columns
            columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                       'total_bedrooms', 'population', 'households', 'median_income']
            
            df = pd.DataFrame([data], columns=columns)

            # Ensure all necessary columns are present
            for col in columns:
                if col not in df.columns:
                    return jsonify({"error": f"Missing feature: {col}"}), 400

            # Make prediction
            prediction = model.predict(df)

            # Return prediction
            return jsonify({"prediction": prediction[0]})

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    # If it's a GET request, render the prediction form
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)