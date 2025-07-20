from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and preprocessing tools
model = tf.keras.models.load_model('regression_model.h5')

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from form
        data = request.form
        geography = data['geography']
        gender = data['gender']
        age = int(data['age'])
        balance = float(data['balance'])
        credit_score = int(data['credit_score'])
        exited = int(data['exited'])
        tenure = int(data['tenure'])
        num_of_products = int(data['num_of_products'])
        has_cr_card = int(data['has_cr_card'])
        is_active_member = int(data['is_active_member'])

        # Preprocess
        input_df = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'Exited': [exited]
        })

        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

        input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

        for col in scaler.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[scaler.feature_names_in_]
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0][0]
        return jsonify({'predicted_salary': float(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
