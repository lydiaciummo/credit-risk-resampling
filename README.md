# Credit Risk Classification and Resampling

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. This project uses various techniques to train and evaluate models with imbalanced classes.  The main goal of the challenge is to utilize a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

---

## Analysis Overview

### Purpose

The purpose of this analysis is to determine if the Logistic Regression machine learning model was able to more accurately predict the credit risk of loans when trained on a dataset with imbalanced classes versus a dataset with artificially balanced classes.

### Nature of the Data and Predictions 

Our training dataset consisted of historical lending activity from a peer-to-peer lending services company. Each loan in the dataset had 7 feature: loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt. Each loan was classified as either "healthy" or "high-risk" with `0` representing a healthy loan and `1` representing a high-risk loan. Our model needed to predict whether the loan should be classified as healthy or high-risk.

Our original dataset consisted of 75,036 "healthy" loans and 2,500 "high-risk" loans, making the classes imbalanced in favor of the healthy loans.

### Process

After reading in the data from a CSV file, I divided the dataset into features and labels. I assigned the DataFrame containing the features the variable `X` and the series containing the labels the variable `y`. 

```
# Separate the y variable, the labels
y = lending_data['loan_status']

# Separate the X variable, the features
X = lending_data.drop(columns='loan_status')
```

I then determined the number of data points in each class using the `value_counts` function.

```
y.value_counts()

0    75036
1     2500
Name: loan_status, dtype: int64
```

Next, I split the data into training and testing datasets using the `train_test_split` function. The training datasets were used to train the model, and the testing datasets were used to determine the model's accuracy.

```
# Import the train_test_learn module
from sklearn.model_selection import train_test_split

# Split the data using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

#### Running the Imbalanced Data Through the Model

Now that I had the training datasets, I could then use them to fit, or train, the Logistic Regression model. Logistic Regression is a machine learning model that, on a basic level, calculates the probability of a certain outcome and converts it to a binary decision, `0` or `1`.

To use the model, we first must create an instance of the model and fit it to the training data.

```
# Import the LogisticRegression module from SKLearn
from sklearn.linear_model import LogisticRegression

# Instantiate the Logistic Regression model
model = LogisticRegression(random_state=1)

# Fit the model using training data
model.fit(X_train, y_train)
```

To test the accuracy of the model, I used `model.predict` to generate label predictions for the testing dataset and saved the predictions to the variable `testing_predictions`.

```
# Make a prediction using the testing data
testing_predictions = model.predict(X_test)
```

I then used the testing predictions to generate an accuracy score, confusion matrix, and classification report.

```
# Print the balanced_accuracy score of the model
balanced_accuracy_score(y_test, testing_predictions)

0.9520479254722232

# Generate a confusion matrix for the model
print(confusion_matrix(y_test, testing_predictions))

[[18663   102]
 [   56   563]]
 
# Print the classification report for the model
print(classification_report_imbalanced(y_test, testing_predictions))

                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.91      1.00      0.95      0.91     18765
          1       0.85      0.91      0.99      0.88      0.95      0.90       619

avg / total       0.99      0.99      0.91      0.99      0.95      0.91     19384

```

Evaluating these metrics, we can determine that the logistic regression model was very good at predicting the "healthy loan" label, with perfect precision and F1 score, and near perfect recall. However, it did not do as well at predicting the "high-risk loan" label, which makes sense given it was the minority class.

#### Running the Resampled Data Through the Model

Now that we have our accuracy metrics for the imbalanced model, it's time to test the resampled data. For this project, I used the random oversampling technique, which balances out the classes by creating more instances of the minority class data.

To do this, I created an instance of the `RandomOverSampler` model and used it to create new training datasets.

```
# Import the RandomOverSampler module form imbalanced-learn
from imblearn.over_sampling import RandomOverSampler

# Instantiate the random oversampler model
random_oversampler = RandomOverSampler(random_state=1)

# Fit the original training data to the random_oversampler model
X_resampled, y_resampled = random_oversampler.fit_resample(X_train, y_train)
```

We can see by running `value_counts` that the classes are now the same:

```
# Count the distinct values of the resampled labels data
y_resampled.value_counts()

0    56271
1    56271
Name: loan_status, dtype: int64
```

To generate the testing predictions for the resampled data, I repeated the process of running the Logistic Regression model using the new training datasets. I then generated the accuracy score, confusion matrix, and classification report.

```
# Print the balanced_accuracy score of the model 
print(balanced_accuracy_score(y_test, resampled_predictions))

0.9936781215845847

# Generate a confusion matrix for the model
print(confusion_matrix(y_test, resampled_predictions))

[[18649   116]
 [    4   615]]
 
# Print the classification report for the model
print(classification_report_imbalanced(y_test, resampled_predictions))

                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.99      1.00      0.99      0.99     18765
          1       0.84      0.99      0.99      0.91      0.99      0.99       619

avg / total       0.99      0.99      0.99      0.99      0.99      0.99     19384
```

## Results

* Machine Learning Model 1:
  * Accuracy: 95.20%
  * Precision: 100% for the class `0` and 85% for class `1`. 99% average precision
  * Recall: 99% for class `0` and 91% for class `1`. 99% average recall

* Machine Learning Model 2:
  * Accuracy: 99.36%
  * Precision: 100% for class `0` and 84% for class `1`. 99% average precision
  * Recall: 99%  for class `0` and 99% for class `1`. 99% average recall

## Summary

The logistic regression model fit with the oversampled data performed the same when predicting the "healthy loan" label, with perfect precision and F1 score, and nearly perfect recall. For the "high-risk loan" data, the precision was 1% lower for the resampled data, but the recall was 8% higher, and the F1 score was 3% higher. The accuracy score for the oversampled data was 4.16% higher than the accuracy score for the imbalanced data. Overall, based on these metrics, the model performed better with the resampled data than with the imbalanced data.

I think that how we evaluate the performance of the model does depend on the problem we are trying to solve. For example, if we are only trying to identify the "healthy" loans, the imbalanced data works just fine. However, if we are trying to identify the "high-risk" loans, I think it is better to use the resampled data.


