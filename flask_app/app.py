from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib
import logging

app = Flask(__name__)

# Enable logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Load the model and scaler
try:
    model = joblib.load('model.pkl')  # Adjust the path if needed
    scaler = joblib.load('minmax_scaler.pkl')
except Exception as e:
    app.logger.error(f"Error loading model or scaler: {e}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Extract input data from form
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

            # Prepare data for prediction
            input_data = np.array([[sunshine, wind_gust_speed, humidity9am, humidity3pm,
                                    pressure9am, pressure3pm, cloud9am, cloud3pm, 
                                    temp3pm, rain_today]])

            # Scale the data
            input_data_scaled = scaler.transform(input_data)

            # Make a prediction
            prediction = model.predict(input_data_scaled)

            # Convert prediction to a readable format
            prediction_result = "It will rain tomorrow" if prediction[0] == 1 else "It will not rain tomorrow"
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
