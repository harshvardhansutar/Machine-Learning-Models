import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import LabelEncoder  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# Step 1: Load Data
data_test = pd.read_csv("P:\\Machine_Learning_Models\\Titanic Dataset EDA\\test.csv")
data_train = pd.read_csv("P:\\Machine_Learning_Models\\Titanic Dataset EDA\\train.csv")

# Store PassengerId before dropping
test_passenger_ids = data_test['PassengerId']

# Step 2: Check Missing Values
print("Missing Values in test data:\n", data_test.isnull().sum())
print("\nMissing Values in train data:\n", data_train.isnull().sum())

# Step 3: Combine Datasets for Preprocessing
combine = [data_train, data_test]

# Step 4: Fill Missing Values
for dataset in combine:
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    dataset.drop('Cabin', axis=1, inplace=True)  # Drop 'Cabin' due to high missingness

# Re-check after filling
print("\n\nAfter Filling Missing Values : \nTrain Data:\n", data_train.isnull().sum())
print("\nTest Data:\n", data_test.isnull().sum())

# Step 5: Encode Categorical Variables
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1}).astype(int)

for i in range(len(combine)):
    combine[i] = pd.get_dummies(combine[i], columns=['Embarked'], drop_first=True)

# Step 6: Feature Engineering
# Extract titles from names
for dataset in combine:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Group rare titles
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
         'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Map titles to numbers
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping).fillna(0).astype(int)

# Create FamilySize
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create IsAlone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Create Age*Class interaction feature
for dataset in combine:
    dataset['Age*Class'] = dataset['Age'] * dataset['Pclass']

# ✅ Drop unneeded columns (fix applied here)
drop_cols = ['Name', 'Ticket', 'SibSp', 'Parch']
for dataset in combine:
    dataset.drop(columns=drop_cols, inplace=True)

# Step 7: Split data_train into X and y
X = data_train.drop(['Survived', 'PassengerId'], axis=1)
y = data_train['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train ML Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Step 9: Predict on Validation Set
y_pred = model.predict(x_test)

# Step 10: Evaluation
print("\n\nValidation Accuracy : ", accuracy_score(y_test, y_pred))
print("\n\nClassification Report :\n", classification_report(y_test, y_pred))

# Step 10: Prepare test_df and make predictions
X_test = data_test.drop(['PassengerId'], axis=1)

# ⚠️ Ensure test data has the same features as training data
missing_cols = set(X.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0  # add missing columns with default 0

X_test = X_test[X.columns]  # re-order columns to match

test_predictions = model.predict(X_test)

# Step 11: Create submission file
submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': test_predictions.astype(int)
})

submission.to_csv("submission.csv", index=False)
print("\n✅ submission.csv generated successfully!")
