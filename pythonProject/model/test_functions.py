import pandas as pd
import pytest

def test_model_metrics( precision, recall, fbeta ):
    assert (precision >= 0 and precision <= 1) and (recall >= 0 and recall <= 1) and (fbeta >= 0 and fbeta <= 1)

def test_precision_valid_domain(feature, value, precision):
    assert (precision >= 0 and precision <= 1), f"For {feature} with value {value}, precision of {value} not between 0.0 and 1.0."
def test_recall_valid_domain (feature, value, recall):
    assert  (recall >= 0 and recall <= 1), f"For {feature} with value {value}, recall of {value} not between 0.0 and 1.0."
def test_fbeta_valid_domain(feature, value, fbeta):
    assert (fbeta >= 0 and fbeta <= 1), f"For {feature} with value {value}, fbeta of {value} not between 0.0 and 1.0."
