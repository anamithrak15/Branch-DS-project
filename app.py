from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import psycopg2
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
CORS(app)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
model_features = model.feature_names_in_

DB_HOST = 'branchhomeworkdb.cv8nj4hg6yra.ap-south-1.rds.amazonaws.com'
DB_PORT = '5432'
DB_USER = 'datascientist'
DB_PASSWORD = '47eyYBLT0laW5j9U24Uuy8gLcrN'
DB_NAME = 'branchdsprojectgps'

def create_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

def retrieve_user_data(user_id):
    conn = create_db_connection()
    query = f"SELECT * FROM user_attributes WHERE user_id = '{user_id}';"
    user_data = pd.read_sql(query, conn)
    conn.close()
    
    if user_data.empty:
        return None
    
    return user_data.iloc[0]

def preprocess_data(data):
    data = data.to_frame().T
    data = data.reindex(columns=model_features, fill_value=0)
    
    for col in data.select_dtypes(include=['object']).columns:
        if col in encoder.classes_:
            data[col] = encoder.transform(data[col])
        else:
            data[col] = 0

    data = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)

    return data

@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.json
    user_id = request_data.get("user_id")
    raw_data = request_data.get("data")

    if not user_id and not raw_data:
        return jsonify({"error": "No user_id or data provided"}), 400
    if user_id and raw_data:
        return jsonify({"error": "Provide either user_id or raw data, not both"}), 400

    if user_id:
        user_data = retrieve_user_data(user_id)
        if user_data is None:
            return jsonify({"error": "User not found"}), 404
        user_data = user_data.dropna()
    else:
        user_data = pd.Series(raw_data)

    processed_data = preprocess_data(user_data)
    prediction = model.predict(processed_data)
    outcome = "repaid" if prediction[0] == 1 else "defaulted"

    return jsonify({"user_id": user_id if user_id else "raw_data", "prediction": outcome}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
