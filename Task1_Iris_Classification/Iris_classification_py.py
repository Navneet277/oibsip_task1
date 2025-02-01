"""
Original file is located at
    https://colab.research.google.com/github/d2l-ai/d2l-en-colab/blob/master/chapter_appendix-tools-for-deep-learning/jupyter.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Function to take user input
def get_user_input():
    try:
        print("Enter the values for Sepal Length, Sepal Width, Petal Length, Petal Width (in cm):")
        sepal_length = float(input("Sepal Length: "))
        sepal_width = float(input("Sepal Width: "))
        petal_length = float(input("Petal Length: "))
        petal_width = float(input("Petal Width: "))

        # Create a list with user inputs
        return [[sepal_length, sepal_width, petal_length, petal_width]]
    except ValueError:
        print("Invalid input! Please enter numeric values.")
        return None

# Step 1: Load the Dataset
# Update the file path if necessary
data = pd.read_csv( "data/Iris.csv")

# Step 2: Data Exploration
print("First 5 rows of the dataset:")
print(data.head())
print("\nSummary statistics:")
print(data.describe())
print("\nColumns in the dataset:")
print(data.columns)

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Step 3: Data Cleaning
# Drop unnecessary columns if any (e.g., 'Id')
if 'Id' in data.columns:
    data.drop('Id', axis=1, inplace=True)

# Step 4: Exploratory Data Analysis (EDA)
# Visualize class distribution
sns.countplot(data['Species'])
plt.title("Class Distribution")
plt.show()

# Pairplot to visualize relationships
sns.pairplot(data, hue='Species', diag_kind='kde')
plt.show()

# Step 5: Data Preprocessing
# Separate features (X) and target (y)
X = data.drop('Species', axis=1)
y = data['Species']

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Build the Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Model Evaluation
# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Step 8: Visualize Results
# Plot the confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

try:
    # Take user input
    user_input = get_user_input()
    if user_input is not None:
        # Scale the user input
        user_input = pd.DataFrame(user_input, columns=X.columns)
        custom_data_scaled = scaler.transform(user_input)

        # Predict
        prediction = model.predict(custom_data_scaled)

        # Get the predicted class index instead of the label
        predicted_class_index = model.classes_.tolist().index(prediction[0])
        # Map prediction to species
        species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
        predicted_species = species_mapping[ predicted_class_index]

        print(f"The predicted species for the given input is: {predicted_species}")
    else:
        print("No prediction made due to invalid input.")
except Exception as e:
    print(f"An error occurred: {e}")
