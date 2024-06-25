from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('regression_model_updated.pkl')

# Define a route to render the form
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    features = [float(x) for x in request.form.values()]
    print(features)
    # Perform prediction using the loaded model
    prediction = model.predict([features])
    
    # Prepare and return the response
    output = {'prediction': prediction[0]}
    return render_template('index.html', prediction_text=f'Predicted House Price: ${output["prediction"]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
