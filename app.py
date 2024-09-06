from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the machine learning model from the pickle file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        nitrogen = float(request.form['Nitrogen'])
        phosphorus = float(request.form['Phosphorus'])
        potassium = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph_value = float(request.form['Ph_value'])
        rainfall = float(request.form['Rainfall'])

        # Create an input array for the model
        input_features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]])

        # Predict the crop
        prediction = model.predict(input_features)[0]

        return render_template('index.html', prediction_text=f'The predicted crop is: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)
