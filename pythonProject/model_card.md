# Model Card

## Model Details

Author: Pablo Rosales

Type of model:  Predictive using LogisticRegression

Training/hyperparameter details:

    - Number of iterations:  1000
    - Using One Hot Encoder to encode categorical values.


## Intended Use

Predict the income range of a person (above or below 50K US$) based on demographic traits.

## Data

- The source data is from https://archive.ics.uci.edu/dataset/20/census+income
- The data was cleansed by removing heading and trailing spaces, as well as removing rows with empty fields.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

Overall Performance Metrics:

- Precision:  0.6468812877263581
- Recall:  0.31535066208925944
- F1:  0.42400263765248924

## Ethical Considerations

To protect people's identity, the records have been anonymized.

## Caveats and Recommendations
