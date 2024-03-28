import requests
import json

live_post_api_url = 'https://mlops-nanodegree-course-4-project.onrender.com/predictions'
live_get_api_url = 'https://mlops-nanodegree-course-4-project.onrender.com/'



test_data = json.dumps({
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
})

#response =requests.get(live_get_api_url).content
#print (response)

response = requests.post(live_post_api_url, data=test_data)
print (response)

if response.status_code == 200:
    print("POST request successful")
    print (response.text)
else:
    print("POST request failed")