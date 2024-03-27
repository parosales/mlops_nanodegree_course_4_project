import json
from fastapi.testclient import TestClient
from api_main import app

client = TestClient(app)

def test_post():
    my_data = json.dumps ({
        "age": 1,
        "workclass": "test_workclass",
        "fnlgt": 0,
        "education": "test_education",
        "education_num": 0,
        "marital_status": "test_marital_status",
        "occupation": "test_occupation",
        "relationship": "test_relationship",
        "race": "test_race",
        "sex": "test_sex",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 0,
        "native_country": 0
    } )

    api_response = client.post("/predictions", data = my_data)
    #print( api_response.json()["body"]["age"] )
    print(api_response)

if __name__ == "__main__":
    test_post()

