import numpy as np
import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import json
from dotenv import load_dotenv
import os
import openai


# Load the .env file
load_dotenv()

# Get the secret key from the environment variables
openai.api_key = os.getenv("SECRET_KEY")

# Define the Flask app
app = Flask(__name__)
cors = CORS(app)

# # Load the data from CSV file
# data = pd.read_csv('./data/Data.csv')

# # Extract input features (Age and Weight) and target variable (Salary)
# X = data[['Age', 'Weight']]
# y = data['Salary']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Linear Regression

# # Create a linear regression model
# model = LinearRegression()

# # Fit the linear regression model to the training data
# model.fit(X_train, y_train)

# # Route for predicting the salary with linear regression
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Retrieve the input data from the request
#     age = float(request.json['age'])
#     weight = float(request.json['weight'])

#     # Make the prediction
#     input_data = np.array([[age, weight]])
#     predicted_salary = model.predict(input_data)[0]

#     # Check if predicted salary is below zero and set it to zero if it is
#     predicted_salary = max(predicted_salary, 0)

#     # Return the predicted salary as JSON response
#     return jsonify({'predicted_salary': int(predicted_salary)})

# # Decision Trees

# # Create a decision tree regression model
# decision_tree_model = DecisionTreeRegressor()

# # Fit the decision tree model to the training data
# decision_tree_model.fit(X_train, y_train)

# # Route for predicting the salary with decision trees
# @app.route('/predict-decision-tree', methods=['POST'])
# def predict_decision_tree():
#     # Retrieve the input data from the request
#     age = float(request.json['age'])
#     weight = float(request.json['weight'])

#     # Make the prediction using the decision tree model
#     input_data = np.array([[age, weight]])
#     predicted_salary = decision_tree_model.predict(input_data)[0]

#     # Check if predicted salary is below zero and set it to zero if it is
#     predicted_salary = max(predicted_salary, 0)

#     # Return the predicted salary as JSON response
#     return jsonify({'predicted_salary': int(predicted_salary)})

# # Logistic Regression

# # Create a logistic regression model
# logistic_model = LogisticRegression()

# # Fit the logistic regression model to the training data
# logistic_model.fit(X_train, y_train)

# # Route for predicting the salary with logistic regression
# @app.route('/predict-logistic', methods=['POST'])
# def predict_logistic():
#     # Retrieve the input data from the request
#     age = float(request.json['age'])
#     weight = float(request.json['weight'])

#     # Make the prediction using the logistic regression model
#     input_data = np.array([[age, weight]])
#     predicted_salary = logistic_model.predict(input_data)[0]

#     # Return the predicted salary as JSON response
#     return jsonify({'predicted_salary': int(predicted_salary)})

# # Route to serve the Data.csv file
# @app.route('/data', methods=['GET'])
# def get_data():
#     return send_from_directory('data', 'Data.csv', as_attachment=True)

# Route for predicting with OpenAI integration
@app.route('/openai-predict', methods=['POST'])
def openai_predict():
    # Retrieve the input data from the request
    input_text = request.json['inputText']

    # Prepend the desired string to the input text
    prompt_string = "List 5 high quality problem statements based on the following problem, in a user story format from the agile software development framework. Make each item in the format As a <something>, I want to <do something>, so that <some outcome>. No line breaks. The problem is:"
    input_text = prompt_string + " " + input_text

    # Make the request to the OpenAI API using the openai.Completion.create() method
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=input_text,
        max_tokens=60
    )

    # Check for errors in the response
    if 'choices' in response and len(response['choices']) > 0:
        # Extract the predicted text from the API response
        predicted_text = response['choices'][0]['text'].strip()

        # Split the predicted text into individual items if it's a list
        predicted_items = predicted_text.split('\n')

        # Return the predicted items as JSON response
        return jsonify({'predicted_items': predicted_items})
    else:
        # Handle the situation where 'choices' is not in the response
        print("No 'choices' in API response")
        print(response)
        return jsonify({'error': "No 'choices' in API response"})

@app.route('/openai-solution', methods=['POST'])
def openai_solution():
    # Retrieve the input data from the request
    input_text = request.json['inputText']

    # Prepend or append the desired string to the problem statement
    prompt_string = "Provide 5 really good solution hypotheses for the following solution hypothesis. Keep each problem statement within 200 characters. No line breaks."
    problem_statement = prompt_string + " " + input_text

    # Make the request to the OpenAI API using the openai.Completion.create() method
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=problem_statement,
        max_tokens=200
    )

    # Check for errors in the response
    if 'choices' in response and len(response['choices']) > 0:
        # Extract the predicted text from the API response
        predicted_text = response['choices'][0]['text'].strip()

        # Split the predicted text into individual solutions if it's a list
        predicted_solutions = predicted_text.split('\n')

        # Return the predicted solutions as JSON response
        return jsonify({'predicted_items': predicted_solutions})
    else:
        # Handle the situation where 'choices' is not in the response
        print("No 'choices' in API response")
        print(response)
        return jsonify({'error': "No 'choices' in API response"})

# Run the Flask app
if __name__ == '__main__':
    app.run()