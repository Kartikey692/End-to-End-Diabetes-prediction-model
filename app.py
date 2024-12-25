from flask import Flask, request, render_template
import pickle
import pandas as pd

# Load the trained model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form
    gender = data['gender']
    age = float(data['age'])
    hypertension = int(data['hypertension'])
    heart_disease = int(data['heart_disease'])
    smoking_history = data['smoking_history']
    bmi = float(data['bmi'])
    hba1c_level = float(data['HbA1c_level'])
    blood_glucose_level = float(data['blood_glucose_level'])

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'smoking_history': [smoking_history],
        'bmi': [bmi],
        'HbA1c_level': [hba1c_level],
        'blood_glucose_level': [blood_glucose_level]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]
    result = 'Diabetic' if prediction == 1 else 'Non-Diabetic'

    # Return result
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
