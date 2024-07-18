from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Define blood type encoding
blood_type_encoding = {
    'A+': 0,
    'A-': 1,
    'B+': 2,
    'B-': 3,
    'AB+': 4,
    'AB-': 5,
    'O+': 6,
    'O-': 7
}

app = Flask(__name__)  # Corrected here


@app.route('/')
def home():
    return "Hello, world!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = request.form.get('age')
        months = request.form.get('months')
        weight = request.form.get('weight')
        blood_type = request.form.get('Blood type')

        # Ensure all required parameters are provided
        if not all([age, months, weight, blood_type]):
            return jsonify({'error': 'Missing data'}), 400

        # Ensure inputs are converted to the appropriate data type
        age = float(age)
        months = float(months)
        weight = float(weight)

        if blood_type not in blood_type_encoding:
            return jsonify({'error': 'Invalid blood type'}), 400

        blood_type_encoded = blood_type_encoding[blood_type]

        # Prepare the input vector for prediction
        input_query = np.array([[age, months, weight, blood_type_encoded]])

        # Make a prediction using the loaded model
        result = model.predict(input_query)[0]

        return jsonify({'Eligible': bool(result)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)