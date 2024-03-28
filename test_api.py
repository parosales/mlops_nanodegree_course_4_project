import json
from fastapi.testclient import TestClient
from api_main import app

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    print("response.text")
    print(r.text)

    assert r.status_code == 200
    assert r.text == '"This is the API to the Prediction Model, welcome!"'


def test_post_over_50k():
    my_test_subject_3 = json.dumps({
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
    })

    api_response = client.post("/predictions", data=my_test_subject_3)
    # print(api_response.status_code)
    # print(api_response.text)

    assert api_response.status_code == 200
    assert int(api_response.text) == 1


def test_post_below_50k():
    my_test_subject_1 = json.dumps({
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

    api_response = client.post("/predictions", data=my_test_subject_1)

    # print(api_response.status_code)
    # print(api_response.text)

    assert api_response.status_code == 200
    assert int(api_response.text) == 0


if __name__ == "__main__":
    test_get_path()
    test_post_over_50k()
    test_post_below_50k()
