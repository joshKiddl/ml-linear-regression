import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
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

# NEW USER FLOW

# Route for predicting with OpenAI integration
@app.route('/openai-predict', methods=['POST'])
def openai_predict():
    # Retrieve the input data from the request
    input_text = request.json['inputText']

    # Prepend the desired string to the input text
    prompt_string = "List 5 high quality problem statements based on the following problem, in a user story format from the agile software development framework. Make each item in the format As a <something>, I want to <do something>, so that <some outcome>. No line breaks. The problem is:"
    input_text = prompt_string + " " + input_text

    # Make the request to the OpenAI API using the openai.ChatCompletion.create() method
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": input_text},
        ],
        max_tokens=200
    )

    # Check for errors in the response
    if 'choices' in response and len(response['choices']) > 0:
        # Extract the predicted text from the API response
        predicted_text = response['choices'][0]['message']['content'].strip()

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
    prompt_string = "Given the following user story, generate 5 high-quality acceptance criteria for agile software development. Each criterion should be no more than 100 characters long, in a list format. Only include the list in your response, no other text. The user story is:"
    problem_statement = prompt_string + " " + input_text

    # Make the request to the OpenAI API using the openai.ChatCompletion.create() method
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": problem_statement},
        ],
        max_tokens=200
    )

    # Check for errors in the response
    if 'choices' in response and len(response['choices']) > 0:
        # Extract the predicted text from the API response
        predicted_text = response['choices'][0]['message']['content'].strip()

        # Split the predicted text into individual solutions if it's a list
        predicted_solutions = predicted_text.split('\n')

        # Return the predicted solutions as JSON response
        return jsonify({'predicted_items': predicted_solutions})
    else:
        # Handle the situation where 'choices' is not in the response
        print("No 'choices' in API response")
        print(response)
        return jsonify({'error': "No 'choices' in API response"})

# @app.route('/technical-requirements', methods=['POST'])
# def technical_requirements():
#     input_text = request.json['inputText']

#     # Preprocess input_text to split it into individual sentences
#     input_text_list = input_text.split(', ')
#     input_text = ' '.join(input_text_list)

#     prompt_string = "Given the following user story and acceptance criteria, generate a list of 10 high-quality technical requirements as suggestions for developing this solution. Each requirement should be no more than 100 characters long, in a list format. Only include the list in your response, no other text. The user stories and acceptance criteria are:"
#     problem_statement = prompt_string + " " + input_text
#     response = openai.Completion.create(
#         model="text-davinci-002",
#         prompt=problem_statement,
#         max_tokens=200
#     )
#     if 'choices' in response and len(response['choices']) > 0:
#         predicted_text = response['choices'][0]['text'].strip()
#         predicted_requirements = predicted_text.split('\n')
#         return jsonify({'predicted_items': predicted_requirements})
#     else:
#         print("No 'choices' in API response")
#         print(response)
#         return jsonify({'error': "No 'choices' in API response"})
    
@app.route('/tasks', methods=['POST'])
def tasks():
    input_text = request.json['inputText']

    # Preprocess input_text to split it into individual sentences
    input_text_list = input_text.split(', ')
    input_text = ' '.join(input_text_list)

    prompt_string = "Given the following acceptance criteria and technical requirements, provide a list of 10 detailed programming tasks that would be needed to build the digital solution. Each item should be no more than 100 characters long, in a list format. Only include the list in your response, no other text: "
    problem_statement = prompt_string + " " + input_text

    # Make the request to the OpenAI API using the openai.ChatCompletion.create() method
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": problem_statement},
        ],
        max_tokens=200
    )

    # Check for errors in the response
    if 'choices' in response and len(response['choices']) > 0:
        # Extract the predicted text from the API response
        predicted_text = response['choices'][0]['message']['content'].strip()

        # Split the predicted text into individual tasks if it's a list
        predicted_tasks = predicted_text.split('\n')

        # Return the predicted tasks as JSON response
        return jsonify({'predicted_items': predicted_tasks})
    else:
        # Handle the situation where 'choices' is not in the response
        print("No 'choices' in API response")
        print(response)
        return jsonify({'error': "No 'choices' in API response"})
    
@app.route('/targetCustomer', methods=['POST'])
def targetCustomer():
    input_text = request.json['inputText']

    # Preprocess input_text to split it into individual sentences
    input_text_list = input_text.split(', ')
    input_text = ' '.join(input_text_list)

    prompt_string = "Based on the following User Story, Acceptance Criteria, Technical Requirement, and Tasks, provide me with the most likely options of my who my target customer is. Each item should be no more than 100 characters long, in a list format. Only include the list in your response, no other text: "
    problem_statement = prompt_string + " " + input_text
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Assuming the model's name
        messages=[
            {"role": "user", "content": problem_statement},
        ],
        max_tokens=200
    )
    if 'choices' in response and len(response['choices']) > 0:
        predicted_text = response['choices'][0]['message']['content'].strip()
        predicted_requirements = predicted_text.split('\n')
        return jsonify({'predicted_items': predicted_requirements})
    else:
        print("No 'choices' in API response")
        print(response)
        return jsonify({'error': "No 'choices' in API response"})

    
# @app.route('/marketSize', methods=['POST'])
# def marketSize():
#     input_text = request.json['inputText']

#     # Preprocess input_text to split it into individual sentences
#     input_text_list = input_text.split(', ')
#     input_text = ' '.join(input_text_list)

#     prompt_string = "Given the following Target Customer, give me the market size, most likely being at the top. It should be in the format of 'x people across x countries' Keep them within 100 characters. No line breaks: "
#     problem_statement = prompt_string + " " + input_text
#     response = openai.Completion.create(
#         model="gpt-4",  # Assuming the model's name
#         prompt=problem_statement,
#         max_tokens=200
#     )
#     if 'choices' in response and len(response['choices']) > 0:
#         predicted_text = response['choices'][0]['text'].strip()
#         predicted_requirements = predicted_text.split('\n')
#         return jsonify({'predicted_items': predicted_requirements})
#     else:
#         print("No 'choices' in API response")
#         print(response)
#         return jsonify({'error': "No 'choices' in API response"})
    
@app.route('/dataElements', methods=['POST'])
def dataElements():
    input_text = request.json['inputText']

    # Preprocess input_text to split it into individual sentences
    input_text_list = input_text.split(', ')
    input_text = ' '.join(input_text_list)

    prompt_string = "Given the following final problem statement, acceptance criteria, and target market, list for me the 5 important data metrics to consider for my feature. Each item should be no more than 100 characters long, in a list format. Only include the list in your response, no other text: "
    problem_statement = prompt_string + " " + input_text
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Assuming the model's name
        messages=[
            {"role": "user", "content": problem_statement},
        ],
        max_tokens=200
    )
    if 'choices' in response and len(response['choices']) > 0:
        predicted_text = response['choices'][0]['message']['content'].strip()
        predicted_requirements = predicted_text.split('\n')
        return jsonify({'predicted_items': predicted_requirements})
    else:
        print("No 'choices' in API response")
        print(response)
        return jsonify({'error': "No 'choices' in API response"})

@app.route('/hypothesis', methods=['POST'])
def hypothesis():
    input_text = request.json['inputText']

    # Preprocess input_text to split it into individual sentences
    input_text_list = input_text.split(', ')
    input_text = ' '.join(input_text_list)

    prompt_string = "Based on the finalProblemStatement, the acceptanceCriteria and the targetCustomer, give me a solution hypothesis for this feature, in the format: 'X amount / percent of Target market / persona can do something / specific metric of the solution. No line breaks): "
    problem_statement = prompt_string + " " + input_text
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Assuming the model's name
        messages=[
            {"role": "user", "content": problem_statement},
        ],
        max_tokens=200
    )
    if 'choices' in response and len(response['choices']) > 0:
        predicted_text = response['choices'][0]['message']['content'].strip()
        predicted_requirements = predicted_text.split('\n')
        return jsonify({'predicted_items': predicted_requirements})
    else:
        print("No 'choices' in API response")
        print(response)
        return jsonify({'error': "No 'choices' in API response"})
    
@app.route('/marketing-material', methods=['POST'])
def marketingMaterial():
    input_text = request.json['inputText']

    # Preprocess input_text to split it into individual sentences
    input_text_list = input_text.split(', ')
    input_text = ' '.join(input_text_list)

    prompt_string = "Based on the target customer, market size, and solution hypotheses, provide me a list of potential marketing materials for this feature. Example 1: 'Blog: <title>', Example 2: 'Email: <Subject Line>, Example 3: 'Social Post: <Summary>. Each item should be no more than 200 characters long, in a list format. Only include the list in your response, no other text: "
    problem_statement = prompt_string + " " + input_text
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Assuming the model's name
        messages=[
            {"role": "user", "content": problem_statement},
        ],
        max_tokens=200
    )
    if 'choices' in response and len(response['choices']) > 0:
        predicted_text = response['choices'][0]['message']['content'].strip()
        predicted_requirements = predicted_text.split('\n')
        return jsonify({'predicted_items': predicted_requirements})
    else:
        print("No 'choices' in API response")
        print(response)
        return jsonify({'error': "No 'choices' in API response"})
    
@app.route('/feature-name', methods=['POST'])
def featureName():
    input_text = request.json['inputText']

    # Preprocess input_text to split it into individual sentences
    input_text_list = input_text.split(', ')
    input_text = ' '.join(input_text_list)

    prompt_string = "Based on the user story, target customer, and solution hypothesis, provide me a list of potential Feature Names for this feature. Each item should be no more than 4 words long, capitalised, in a list format. Only include the list in your response, no other text: "
    problem_statement = prompt_string + " " + input_text
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Assuming the model's name
        messages=[
            {"role": "user", "content": problem_statement},
        ],
        max_tokens=200
    )
    if 'choices' in response and len(response['choices']) > 0:
        predicted_text = response['choices'][0]['message']['content'].strip()
        predicted_requirements = predicted_text.split('\n')
        return jsonify({'predicted_items': predicted_requirements})
    else:
        print("No 'choices' in API response")
        print(response)
        return jsonify({'error': "No 'choices' in API response"})

# CREATE FEATURE

# Problem Statement
@app.route('/user-story', methods=['POST'])
def userStory():
    # Retrieve the input data from the request
    input_data = request.json['inputData']
    
    # Assign it to input_text
    input_text = input_data

    # Prepend the desired string to the input text
    prompt_string = "Based on the following inputs, generate a list of 5 options for a potentially suitable User Story:"
    input_text = prompt_string + " " + input_text

    # Make the request to the OpenAI API using the openai.Completion.create() method
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=input_text,
        max_tokens=200
    )

    # Extract the generated text from the API response
    generated_text = response.choices[0].text.strip()

    # Return the generated text in a JSON response
    return jsonify({'generated_text': generated_text}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run()