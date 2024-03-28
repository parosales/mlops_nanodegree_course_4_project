from fastapi import FastAPI
from pydantic import BaseModel
import model.model
from joblib import load
import pandas as pd
import json
import model.data
import pickle

model_path = './model/'
model_file = 'model.joblib'
one_hot_encoder_file = 'one_hot_encoder_file.pickle'


class SubjectAttributes(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: int
    native_country: str


# model and encoder saved in the training phase
model_obj = load(model_path + model_file)
with open(model_path + one_hot_encoder_file, 'rb') as f:
    oh_encoder = pickle.load(f)

# Instantiate the app.
app = FastAPI()


@app.get("/")
async def welcome_to_api():
    return {"This is the API to the Prediction Model, welcome!"}


@app.post("/predictions")
async def predict_salary_over50(test_subject: SubjectAttributes):
    cat_features = ['workclass', 'education', 'marital_status',
                    'occupation', 'relationship', 'race',
                    'native_country', 'sex']

    test_subject_as_json = json.dumps(test_subject.__dict__)
    data_dict = json.loads(test_subject_as_json)
    subject_df = pd.DataFrame.from_records([data_dict])
    # print (subject_df)

    test_subject, y, encoder, lb = model.data.process_data(
        X=subject_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=oh_encoder,
        lb=None
    )

    salary_over50 = model.model.inference(model_obj, test_subject)[0]
    # print( salary_over50 )
    return int(salary_over50)
