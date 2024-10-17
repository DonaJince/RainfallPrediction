from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib  # Ensure you have joblib installed
from sklearn.preprocessing import MinMaxScaler
import logging

app = Flask(__name__)

# Enable logging to debug issues
logging.basicConfig(level=logging.DEBUG)

# Load your model and scaler
try:
    model = joblib.load('model.pkl')  # Adjust the path as necessary
    scaler = joblib.load('minmax_scaler.pkl')  # Ensure you have saved your scaler after fitting
except Exception as e:
    app.logger.error(f"Error loading model or scaler: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form inputs
            sunshine = float(request.form['Sunshine'])
            wind_gust_speed = float(request.form['WindGustSpeed'])
            humidity9am = float(request.form['Humidity9am'])
            humidity3pm = float(request.form['Humidity3pm'])
            pressure9am = float(request.form['Pressure9am'])
            pressure3pm = float(request.form['Pressure3pm'])
            cloud9am = float(request.form['Cloud9am'])
            cloud3pm = float(request.form['Cloud3pm'])
            temp3pm = float(request.form['Temp3pm'])
            rain_today = float(request.form['RainToday'])

            # Prepare the input data for prediction
            input_data = np.array([[sunshine, wind_gust_speed, humidity9am, humidity3pm,
                                     pressure9am, pressure3pm, cloud9am, cloud3pm, 
                                     temp3pm, rain_today]])

            # Log the input data for debugging
            app.logger.debug(f"Input data: {input_data}")

            # Scale the input data
            input_data_scaled = scaler.transform(input_data)

            # Make the prediction
            prediction = model.predict(input_data_scaled)

            # Log the prediction for debugging
            app.logger.debug(f"{prediction}")

            # Prepare the prediction result
            prediction_result = "It will rain tomorrow" if prediction[0] == 1 else "It will not rain tomorrow"

            # Redirect to the result page with the prediction
            return redirect(url_for('result', prediction=prediction_result))
        except Exception as e:
            app.logger.error(f"Error in prediction: {e}")

    return render_template('index.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
