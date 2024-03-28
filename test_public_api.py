import requests

live_api_url = 'https://mlops-nanodegree-course-4-project.onrender.com/predictions'

test_data = {
    "age": 43,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 292175,
    "education": "Masters",
    "education_num": 14,
    "marital_status": "Divorced",
    "occupation": "Exec-managerial",
    "relationship": "Unmarried",
    "race": "White",
    "sex": "Female",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 45,
    "native_country": "United-States"
}

response = requests.post(live_api_url, data=test_data)

if response.status_code == 200:
    print("POST request successful")
else:
    print("POST request failed")