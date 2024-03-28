import model.model as model
import model.data as data
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression

# ============== Ingestion and pre-processing =================== #
categorical_features = ['workclass', 'education', 'marital-status',
                        'occupation', 'relationship', 'race',
                        'native-country', 'sex']
label = 'salary'
lb = LabelBinarizer()
clean_data_df = pd.read_csv('./src_data/clean_census.csv')

# ---------------- Pre-processing --------------- #
X, y, encoder_for_trained_data, lb = data.process_data(
    clean_data_df,
    categorical_features,
    label,
    True
)

# =============== Model Training ===================== #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=23
)
model_obj = model.train_model(X_train, y_train)

preds = model.inference(model_obj, X_test)

precision, recall, fbeta = model.compute_model_metrics(y_test, preds)


def test_train_model():
    # assert type(model_obj) == LogisticRegression
    assert isinstance(model_obj, LogisticRegression)


def test_inference():
    # assert type(preds) == numpy.ndarray
    assert isinstance(preds, numpy.ndarray)


def test_compute_model_metrics():
    assert ((precision >= 0 and precision <= 1) and
            (recall >= 0 and recall <= 1) and
            (fbeta >= 0 and fbeta <= 1))
