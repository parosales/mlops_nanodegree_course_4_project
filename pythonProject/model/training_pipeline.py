# ========================================= Setup ========================================= #

# ----------------------------------- Import Libraries ----------------------------------- #
import data, model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, label_binarize, LabelBinarizer

# ------------------------------- Define vars and constants ------------------------------- #
categorical_features = [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country', 'sex' ]
label = 'salary'
training = True
lb = LabelBinarizer()

# ============================== Ingestion and pre-processing ============================== #
clean_data_df = pd.read_csv( './data/clean_census.csv' )

# -------------------------------------- Pre-processing ------------------------------------- #
X, y, encoder, lb = data.process_data(
    clean_data_df ,
    categorical_features,
    label,
    training
)

# ====================================== Model Training ====================================== #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
model_obj = model.train_model( X_train, y_train )
print ( type(model_obj) )

# ====================================== Predict ====================================== #
#preds = model.inference(model_obj, X_test)
#print(preds)

# ====================================== Test slices ====================================== #
#for col in categorical_features:
print (" ************************ testing slices ************************ ")
for col in ['marital-status']:
    for tested_value in clean_data_df[col].unique():
        print (col, tested_value)

        X, y, encoder, lb = data.process_data(
            clean_data_df [ clean_data_df[col] == tested_value ],
            categorical_features,
            label,
            False,
            encoder,
            lb
        )

        preds = model.inference (model_obj, X)
        precision, recall, fbeta = model.compute_model_metrics(y, preds)
        print ( f"precision, recall, fbeta: {precision}, {recall}, {fbeta}" )
