# Titanic - Machine Learning from Disaster 🚢

This project uses the Titanic dataset to build a machine learning model that predicts whether a passenger survived or not, based on features like age, gender, class, and fare.

## 📂 Dataset

The dataset is available on [Kaggle](https://www.kaggle.com/competitions/titanic). It contains information about the passengers on the Titanic.

- **train.csv** — training set (with survival labels)
- **test.csv** — test set (without survival labels)
- **gender_submission.csv** — example of a submission file

## 📊 Problem Statement

> Predict survival on the Titanic and perform binary classification (Survived: 1 or 0).

## 📌 Features Used

- PassengerId
- Pclass (Ticket class)
- Name
- Sex
- Age
- SibSp (Siblings/Spouses aboard)
- Parch (Parents/Children aboard)
- Ticket
- Fare
- Cabin
- Embarked (Port of Embarkation)

## 🛠️ Technologies & Libraries

- Python 🐍
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn

## 📈 Approach

1. **Data Cleaning**
   - Handling missing values (e.g., Age, Embarked)
   - Dropping irrelevant columns (e.g., Cabin, Ticket)

2. **Feature Engineering**
   - Converting categorical data (e.g., Sex, Embarked) to numeric
   - Creating new features (e.g., FamilySize, IsAlone)

3. **Modeling**
   - Logistic Regression
   - Decision Trees
   - Random Forest
   - K-Nearest Neighbors (KNN)

4. **Evaluation**
   - Accuracy Score
   - Confusion Matrix
   - Cross-validation

## 📦 Output

A `.csv` file that contains `PassengerId` and `Survived` predictions for the test dataset.

## 📁 File Structure

 
