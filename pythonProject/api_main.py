from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel
import model.model
from joblib import load
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import json
import model.data


model_path = './model/'
model_file = 'model.joblib'


#age,workclass,fnlgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,salary
#39,State-gov,77516,Bachelors,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States,<=50K
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
    native_country: int

# model saved in the training phase
model_obj = load(model_path + model_file)

# Instantiate the app.
app = FastAPI()

@app.get("/")
async def welcome_to_api():
    return {"This is the API to the Prediction Model, welcome!"}

#@app.post("/predictions")
#async def exercise_function(the_body: SubjectAttributes):
#    return {"body": the_body}

@app.post("/predictions")
async def predict_salary_over50(test_subject: SubjectAttributes):
    cat_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race',
                     'native_country', 'sex']
    #label = 'salary'
    #lb = LabelBinarizer()

    test_subject_as_json = json.dumps(test_subject.__dict__)
    data_dict = json.loads(test_subject_as_json)
    subject_df = pd.DataFrame.from_records([data_dict])
    print (subject_df)

    my_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    #X_categorical = subject_df[cat_features].values
    #print ("BEFORE encoding", X_categorical)
    #X_categorical = my_encoder.fit_transform(X_categorical)
    #X_categorical = my_encoder.transform(X_categorical)
    #print("AFTER encoding", X_categorical)


    #print("X_continuous: ", X_continuous)

    test_subject, y, encoder, lb = model.data.process_data (
        X = subject_df,
        categorical_features = cat_features,
        label = None,
        training = False,
        encoder = my_encoder,
        lb = None
    )

    print("INFERENCE: ")
    print ( model.model.inference(model_obj, test_subject ) )

    return 1