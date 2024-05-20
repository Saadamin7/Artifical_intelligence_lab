import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load your custom dataset
data_path = r'C:\Users\saada\OneDrive\Desktop\AI\HousingData.csv'  # Update with your dataset path
boston = pd.read_csv(data_path)  # Assuming your dataset is in CSV format

# Handle missing values (replace NaNs with the mean of each column)
imputer = SimpleImputer(strategy='mean')
boston_imputed = pd.DataFrame(imputer.fit_transform(boston), columns=boston.columns)

# Split the data into features (X) and target variable (y)
X = boston_imputed.drop('MEDV', axis=1)
y = boston_imputed['MEDV']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate mean squared error (MSE) on the test set
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Visualize the relationship between a specific feature (e.g., 'RM') and house prices (MEDV)
feature_of_interest = 'RM'  # Update with the feature you want to visualize
plt.scatter(X_test[feature_of_interest], y_test, color='blue', label='Actual')
plt.plot(X_test[feature_of_interest], y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel(feature_of_interest)  # Update with the feature name
plt.ylabel('Median Value of Homes (MEDV)')
plt.title(f'Linear Regression: Predicted vs Actual ({feature_of_interest})')
plt.legend()
plt.show()
