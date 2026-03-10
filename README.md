# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import the required packages and print the present data
Print the placement data and salary data.
Find the null and duplicate values.
Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: D R Nethranjan Chowdary
RegisterNumber:  212225100031
*/
```
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics

# Load dataset
df = pd.read_csv('/content/Placement_Data.csv')
df1 = df.copy()

# Drop unnecessary columns
df1 = df1.drop(['sl_no', 'salary'], axis=1)

# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 
                    'degree_t', 'workex', 'specialisation', 'status']

for col in categorical_cols:
    df1[col] = le.fit_transform(df1[col])

# Features and target
X = df1.iloc[:, :-1]
y = df1['status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Logistic Regression model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("Confusion Matrix:\n", confusion)
print("\nClassification Report:\n", cr)

# Confusion matrix visualization
disp = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion, display_labels=['Placed', 'Not Placed']
)
disp.plot()
```

## Output:
<img width="663" height="676" alt="image" src="https://github.com/user-attachments/assets/6b4ebe34-be8d-49c0-aef1-a6a3be28c88b" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
