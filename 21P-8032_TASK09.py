# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Titanic datasetN
titanic_df = pd.read_csv(r'C:\Users\saada\OneDrive\Desktop\AI\titanic.csv')

# Preprocess the data
titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df['Fare'].fillna(titanic_df['Fare'].median(), inplace=True)
titanic_df = pd.get_dummies(titanic_df, columns=['Sex', 'Embarked'])

# Split the dataset into training and test sets
X = titanic_df.drop('Survived', axis=1)
y = titanic_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the initial MLP classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp_classifier.fit(X_train, y_train)

# Evaluate the performance of the initial MLP classifier
y_pred = mlp_classifier.predict(X_test)
accuracy_initial = accuracy_score(y_test, y_pred)
precision_initial = precision_score(y_test, y_pred)
recall_initial = recall_score(y_test, y_pred)
f1_initial = f1_score(y_test, y_pred)

print("Initial MLP Classifier Performance:")
print("Accuracy:", accuracy_initial)
print("Precision:", precision_initial)
print("Recall:", recall_initial)
print("F1 Score:", f1_initial)

# Fine-tune the MLP classifier
mlp_classifier_tuned = MLPClassifier(hidden_layer_sizes=(150, 75), max_iter=800, random_state=42)
mlp_classifier_tuned.fit(X_train, y_train)

# Evaluate the performance of the fine-tuned MLP classifier
y_pred_tuned = mlp_classifier_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)
recall_tuned = recall_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)

print("\nFine-Tuned MLP Classifier Performance:")
print("Accuracy:", accuracy_tuned)
print("Precision:", precision_tuned)
print("Recall:", recall_tuned)
print("F1 Score:", f1_tuned)
