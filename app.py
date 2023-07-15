import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify

# Load the data from CSV file
data = pd.read_csv('./data/Data.csv')

# Extract input features (Age and Weight) and target variable (Salary)
X = data[['Age', 'Weight']]
y = data['Salary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Define the Flask app
app = Flask(__name__)

# Route for predicting the salary
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input data from the request
    age = float(request.json['age'])
    weight = float(request.json['weight'])

    # Make the prediction
    input_data = np.array([[age, weight]])
    predicted_salary = model.predict(input_data)[0]

    # Return the predicted salary as JSON response
    return jsonify({'predicted_salary': predicted_salary})
