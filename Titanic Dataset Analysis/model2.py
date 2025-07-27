import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Step 1: Load data
train_df = pd.read_csv("P:/Machine_Learning_Models/Titanic Dataset EDA/train.csv")
test_df = pd.read_csv("P:/Machine_Learning_Models/Titanic Dataset EDA/test.csv")
print("Data loaded successfully.\n")

# Save PassengerId for submission
test_passenger_ids = test_df['PassengerId']

# Step 2: Basic info
print("Train Info:")
print(train_df.info())
print("\nMissing Values in Train:\n", train_df.isnull().sum())
print("\nMissing Values in Test:\n", test_df.isnull().sum())

# Step 3: Fill missing values
for df in [train_df, test_df]:
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop(columns=['Cabin'], inplace=True)

    # Fare may be missing in test set
    if 'Fare' in df.columns:
        df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Step 4: Drop unneeded columns
drop_cols = ['PassengerId', 'Name', 'Ticket']
train_df.drop(columns=drop_cols, inplace=True)
test_df.drop(columns=drop_cols, inplace=True)

# Step 5: Separate features and labels
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# Step 6: Train-test split
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Encode categorical features using LabelEncoder
combined = pd.concat([x_train, x_val, test_df], axis=0)
categorical_cols = combined.select_dtypes(include='object').columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))
    label_encoders[col] = le

# Split back
x_train = combined.iloc[:len(x_train), :]
x_val = combined.iloc[len(x_train):len(x_train)+len(x_val), :]
test_final = combined.iloc[len(x_train)+len(x_val):, :]

# Step 8: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Step 9: Predictions & Evaluation on validation set
y_pred = model.predict(x_val)
print("\nValidation Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))

# Step 10: Final prediction on test.csv
test_predictions = model.predict(test_final)

# Step 11: Prepare submission file (optional)
submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': test_predictions
})
submission.to_csv("titanic_submission.csv", index=False)
print("\nSubmission file 'titanic_submission.csv' saved.")
