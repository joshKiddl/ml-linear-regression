import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import json
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Get the secret key from the environment variables
secret_key = os.getenv("SECRET_KEY")

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
    return jsonify({'predicted_salary': int(predicted_salary)})

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
    return jsonify({'predicted_salary': int(predicted_salary)})

# Logistic Regression

# Create a logistic regression model
logistic_model = LogisticRegression()

# Fit the logistic regression model to the training data
logistic_model.fit(X_train, y_train)

# Route for predicting the salary with logistic regression
@app.route('/predict-logistic', methods=['POST'])
def predict_logistic():
    # Retrieve the input data from the request
    age = float(request.json['age'])
    weight = float(request.json['weight'])

    # Make the prediction using the logistic regression model
    input_data = np.array([[age, weight]])
    predicted_salary = logistic_model.predict(input_data)[0]

    # Return the predicted salary as JSON response
    return jsonify({'predicted_salary': int(predicted_salary)})

# Route to serve the Data.csv file
@app.route('/data', methods=['GET'])
def get_data():
    return send_from_directory('data', 'Data.csv', as_attachment=True)

# Route for predicting with OpenAI integration
@app.route('/openai-predict', methods=['POST'])
def openai_predict():
    # Retrieve the input data from the request
    input_text = request.json['inputText']

    # Prepare the data for the OpenAI API
    data = {
        'prompt': input_text,
        'max_tokens': 60
    }

    # Make the request to the OpenAI API
    headers = {
        'Authorization': f'Bearer {secret_key}',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post('https://api.openai.com/v1/engines/davinci-codex/completions', headers=headers, data=json.dumps(data))
        response.raise_for_status()  # This will raise an exception if the response contains an HTTP error status code
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")  # or use your logger here
        return jsonify({'error': str(err)})
    except Exception as err:
        print(f"Other error occurred: {err}")  # or use your logger here
        return jsonify({'error': str(err)})

    if response.status_code == 200:
        response_data = response.json()
        if 'choices' in response_data:
            predicted_text = response_data['choices'][0]['text'].strip()
        else:
            # Handle the situation where 'choices' is not in the response
            print("No 'choices' in API response")
            print(response_data)
            return jsonify({'error': "No 'choices' in API response"})
    else:
        # Handle the situation where the API response is not a success
        print(f"API request failed with status code {response.status_code}")
        return jsonify({'error': f"API request failed with status code {response.status_code}"})

    # Return the predicted text as JSON response
    return jsonify({'predicted_text': predicted_text})

# Run the Flask app
if __name__ == '__main__':
    app.run()