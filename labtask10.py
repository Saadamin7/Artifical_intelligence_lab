import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


# Step 1: Data Preprocessing
# Load the Titanic dataset
titanic_df = pd.read_csv(r'C:\Users\saada\OneDrive\Desktop\AI\titanic.csv')

# Preprocess the dataset
titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
titanic_df.fillna({'Age': titanic_df['Age'].median(), 'Fare': titanic_df['Fare'].median()}, inplace=True)

# Convert categorical data into numeric format using one-hot encoding
categorical_cols = ['Sex', 'Embarked']
titanic_df = pd.get_dummies(titanic_df, columns=categorical_cols, drop_first=True)

# Step 2: Normalization
scaler = StandardScaler()
numeric_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
titanic_df[numeric_cols] = scaler.fit_transform(titanic_df[numeric_cols])

# Step 3: Model Training
# Split the dataset into training and testing sets
X = titanic_df.drop('Survived', axis=1)
y = titanic_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a predictive model using MLPClassifier from scikit-learn
mlp_clf = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)
mlp_clf.fit(X_train, y_train)

# Train deep neural networks with different architectures using Keras
def create_model(hidden_layers, units):
    model = Sequential()
    model.add(Dense(units, input_dim=X_train.shape[1], activation='relu'))
    for _ in range(hidden_layers - 1):
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# List of different neural network architectures
architectures = [(1, 32), (2, 64), (3, 128)]

accuracy_scores = []

for hidden_layers, units in architectures:
    # Create and train the model
    model = create_model(hidden_layers, units)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    accuracy_scores.append(accuracy)

# Step 4: Accuracy Calculation
# Calculate the accuracy of MLPClassifier
mlp_accuracy = mlp_clf.score(X_test, y_test)
accuracy_scores.append(mlp_accuracy)

# Plotting the bar graph to compare accuracies
labels = ['NN (1 layer, 32 units)', 'NN (2 layers, 64 units)', 'NN (3 layers, 128 units)', 'MLP Classifier']
plt.bar(labels, accuracy_scores)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.show()