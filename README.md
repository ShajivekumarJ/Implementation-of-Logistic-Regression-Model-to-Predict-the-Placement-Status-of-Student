# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries.
2. Load the dataset containing student details.
3. Separate independent variables (features) and dependent variable (placement status).
4. Split the dataset into training and testing data.
5. Create a Logistic Regression model.
6. Train the model using training data.
7. Predict placement status using test data.
8. Evaluate the model using accuracy score

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SHAJIVE KUMAR J
RegisterNumber:  212225230258
```
~~~
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\acer\Downloads\Placement_Data.csv")

# Drop salary column (contains NaN values)
df.drop('salary', axis=1, inplace=True)

# Display dataset as table
print("Dataset (First 10 Records):")
print(df.head(10))

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Separate features and target
X = df.drop('status', axis=1)
y = df['status']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)   # Sensitivity

print("\nAccuracy      :", accuracy)
print("Precision     :", precision)
print("Sensitivity   :", recall)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Placement Prediction")
plt.show()

~~~
## Output:

![alt text](<Screenshot 2026-02-04 092104.png>)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
