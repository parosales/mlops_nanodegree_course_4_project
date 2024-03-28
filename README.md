# 1. Overview

## 1.0. General Information about the project

- Author:  Pablo Rosales
- Date:  March 2024
- Course:  Udacity MLOps Nanodegree, Course 4 Final Project

## 1.1. Goal

To predict if the yearly income is above or below US$ 50,000 per year given demographic data.  

## 1.2. Setup

### 1.2.1. Source Data

- The source data is from https://archive.ics.uci.edu/dataset/20/census+income
- The data was cleansed by removing heading and trailing spaces, as well as removing rows with empty fields.

### 1.2.2. Machine Learning Model

- Type of Model:  Logistic Regression

### 1.2.3. Overall performance metrics

Overall Performance Metrics:

- Precision:  0.6468812877263581
- Recall:  0.31535066208925944
- F1:  0.42400263765248924

# 2. How to use it

## 2.1. Access

- Public API URL:  https://mlops-nanodegree-course-4-project.onrender.com
- Local API URL:  http://127.0.0.1:8000/docs

## 2.3. Inputs

- For the POST method used to make predictions (/predictions), a sample demographic data is:
{
        "age": 48,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 191277,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 1902,
        "hours_per_week": 60,
        "native_country": "United-States"
}

## 2.3. Outputs

- In the POST method for predictions:
  - 0 means <= US$ 50k;
  - 1 means > US$ 50k
