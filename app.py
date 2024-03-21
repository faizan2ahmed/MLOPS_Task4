from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('./gender_model.pkl')
scaler = joblib.load('./scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        if isinstance(json_, dict):
            json_ = [json_]
        query_df = pd.DataFrame(json_)

        categorical_cols = ['Occupation', 'Education Level', 'Marital Status', 'Favorite Color']
        query_df = pd.get_dummies(query_df, columns=categorical_cols)

        expected_features = joblib.load('./feature_names.pkl')
        for col in expected_features:
            if col not in query_df.columns:
                query_df[col] = 0
        query_df = query_df.reindex(columns=expected_features)

        query = scaler.transform(query_df)

        prediction = model.predict(query)

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
