from fastapi import FastAPI
from pydantic import BaseModel, Field
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
    age: int = Field(example=[48])
    workclass: str = Field(example=["Self-emp-not-inc"])
    fnlgt: int = Field(example=[191277])
    education: str = Field(example=["Doctorate"])
    education_num: int = Field(example=[16])
    marital_status: str = Field(example=["Married-civ-spouse"])
    occupation: str = Field(example=["Prof-specialty"])
    relationship: str = Field(example=["Husband"])
    race: str = Field(example=["White"])
    sex: str = Field(example=["Male"])
    capital_gain: float = Field(example=[0])
    capital_loss: float = Field(example=[1902])
    hours_per_week: int = Field(example=[60])
    native_country: str = Field(example=["United-States"])


# model and encoder saved in the training phase
model_obj = load(model_path + model_file)
with open(model_path + one_hot_encoder_file, 'rb') as f:
    oh_encoder = pickle.load(f)

# Instantiate the app.
app = FastAPI()


@app.get("/")
async def welcome_to_api():
    return "This is the API to the Prediction Model, welcome!"


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
