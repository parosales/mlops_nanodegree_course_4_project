# =================== Setup ================ #

# --------------- Import Libraries ----------------- #
import data
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from joblib import dump
import pickle

# ---------------- Define vars and constants -------------- #
categorical_features = ['workclass', 'education', 'marital-status',
                        'occupation', 'relationship', 'race',
                        'native-country', 'sex']
label = 'salary'
lb = LabelBinarizer()

# paths and files
model_path = './model/'
model_file = 'model.joblib'
test_set_path = './test/'
test_file = 'tests.py'

# one-hot encoder for trained src_data
one_hot_encoder_file = 'one_hot_encoder_file.pickle'


# ---------------Functions --------------- #
def evaluate_on_categorical_slices(clean_data_df, model_obj, encoder, lb):
    # def evaluate_on_categorical_slices(clean_data_df, model_obj, lb):
    for col in categorical_features:
        for tested_value in clean_data_df[col].unique():
            print(col, tested_value)

            X, y, encoder, lb = data.process_data(
                clean_data_df[clean_data_df[col] == tested_value],
                categorical_features,
                label,
                False,
                encoder,
                lb
            )

            preds = model.inference(model_obj, X)
            precision, recall, fbeta = model.compute_model_metrics(y, preds)
            print(f"precision, recall, fbeta: {precision}, {recall}, {fbeta}")


# ============= Ingestion and pre-processing =============== #
clean_data_df = pd.read_csv('./src_data/clean_census.csv')

# ------------------- Pre-processing ----------------- #
X, y, encoder_for_trained_data, lb = data.process_data(
    clean_data_df,
    categorical_features,
    label,
    True
)

# save the OHE for other uses: e.g. prediction/inference
with open(model_path + one_hot_encoder_file, 'wb') as f:
    pickle.dump(encoder_for_trained_data, f)

# ==================== Model Training ================= #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=23
)
model_obj = model.train_model(X_train, y_train)
print(type(model_obj))

dump(model_obj, model_path + model_file)

# save for further tests
# np.save(test_set_path + test_file, X_test)

# ================= Evaluate Overall performance ================= #
preds = model.inference(model_obj, X_test)
print(type(preds))
precision, recall, fbeta = model.compute_model_metrics(y_test, preds)
print(f"precision, recall, fbeta: {precision}, {recall}, {fbeta}")

# ==================== Evaluate on Slices of src_data ================== #
# evaluate_on_categorical_slices(
#   clean_data_df, model_obj, encoder_for_trained_data, lb
#   )

# =================== Predict ==================== #
preds = model.inference(model_obj, X_test)
print(preds)
