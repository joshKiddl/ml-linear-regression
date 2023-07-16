import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load the data from CSV file
data = pd.read_csv('./data/Data.csv')

# Extract input features (Age and Weight) and target variable (Salary)
X = data[['Age', 'Weight']]
y = data['Salary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Flask app
app = Flask(__name__)
cors = CORS(app)

# Linear Regression

# Create a linear regression model
model = LinearRegression()

# Fit the linear regression model to the training data
model.fit(X_train, y_train)

# Route for predicting the salary with linear regression
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input data from the request
    age = float(request.json['age'])
    weight = float(request.json['weight'])

    # Make the prediction
    input_data = np.array([[age, weight]])
    predicted_salary = model.predict(input_data)[0]

    # Check if predicted salary is below zero and set it to zero if it is
    predicted_salary = max(predicted_salary, 0)

    # Return the predicted salary as JSON response
    return jsonify({'predicted_salary': predicted_salary})

# Decision Trees

# Create a decision tree regression model
decision_tree_model = DecisionTreeRegressor()

# Fit the decision tree model to the training data
decision_tree_model.fit(X_train, y_train)

# Route for predicting the salary with decision trees
@app.route('/predict-decision-tree', methods=['POST'])
def predict_decision_tree():
    # Retrieve the input data from the request
    age = float(request.json['age'])
    weight = float(request.json['weight'])

    # Make the prediction using the decision tree model
    input_data = np.array([[age, weight]])
    predicted_salary = decision_tree_model.predict(input_data)[0]

    # Check if predicted salary is below zero and set it to zero if it is
    predicted_salary = max(predicted_salary, 0)

    # Return the predicted salary as JSON response
    return jsonify({'predicted_salary': predicted_salary})
