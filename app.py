from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and the scaler
model = joblib.load('./gender_model.pkl')
scaler = joblib.load('./scaler.pkl')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and prepare the features from the request
        json_ = request.json
        if isinstance(json_, dict):  # If a single JSON, convert to list
            json_ = [json_]
        query_df = pd.DataFrame(json_)

        # One-hot encode the categorical columns as during the training
        categorical_cols = ['Occupation', 'Education Level', 'Marital Status', 'Favorite Color']
        query_df = pd.get_dummies(query_df, columns=categorical_cols)

        # Ensure the input has the same columns as the training set, fill missing with zeros
        expected_features = joblib.load('./feature_names.pkl')  # Make sure to create this file during model training
        for col in expected_features:
            if col not in query_df.columns:
                query_df[col] = 0
        query_df = query_df.reindex(columns=expected_features)

        # Scale the features
        query = scaler.transform(query_df)

        # Make predictions
        prediction = model.predict(query)

        # Return the prediction
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
