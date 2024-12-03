
import streamlit as st
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import random

# Load the intents data from a JSON file
with open(r'C:\Users\AcerAspireE15\Downloads\intents.json', 'r') as f:
    data = json.load(f)

# Read the dataset into a pandas DataFrame
df = pd.read_json(r'C:\Users\AcerAspireE15\Downloads\intents.json')

# New intents data for adding to the existing dataset
new_intents_data = {
    "tag": ["symptom"] * 4,  # Define the tag for these new entries
    "patterns": [
        "I have a headache",
        "I feel dizzy",
        "I'm not feeling well",
        "I have a cough"
    ],
    "responses": [
        "I'm sorry to hear that you're not feeling well. It's best to consult with a healthcare professional.",
        "Headaches can be caused by various factors. Make sure to stay hydrated and consider resting.",
        "Dizziness can occur due to various reasons; please consult a doctor if it persists.",
        "Coughing can be a sign of various conditions. If it continues, please seek medical advice."
    ]
}

# Convert the new intents data into a DataFrame and append it to the original dataset
new_intents_df = pd.DataFrame(new_intents_data)
df = pd.concat([df, new_intents_df], ignore_index=True)

# Create a dictionary to hold all the intent data in a more structured format
dic = {"tag": [], "patterns": [], "responses": []}

# Loop through the original dataset and extract the tag, patterns, and responses
for intent in data['intents']:
    tag = intent['tag']
    patterns = intent['patterns']
    responses = intent['responses']
    for pattern in patterns:
        dic['tag'].append(tag)
        dic['patterns'].append(pattern)
        dic['responses'].append(responses)  

# Convert the dictionary to a DataFrame for easier manipulation
df = pd.DataFrame.from_dict(dic)

# Define features (X) and target variable (y)
X = df['patterns']
y = df['tag']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model using a Support Vector Machine (SVM)
model = SVC()
model.fit(X_train_vec, y_train)

# Function to predict the intent of user input
def predict_intent(user_input):
    user_input_vec = vectorizer.transform([user_input])  # Vectorize user input
    intent = model.predict(user_input_vec)[0]  # Predict intent
    return intent

# Function to generate a response based on the predicted intent
def generate_response(intent):
    possible_responses = df[df['tag'] == intent]['responses'].values[0]  # Get possible responses for the intent
    response = random.choice(possible_responses)  # Randomly select one response
    return response

# Streamlit user interface
st.title("Chatbot")
st.write("Chatbot for mental health conversation.")

# Get user input
user_input = st.text_input("You:", "")

# If user input is provided, predict intent and generate response
if user_input:
    intent = predict_intent(user_input)
    response = generate_response(intent)
    st.write("Chatbot:", response)

