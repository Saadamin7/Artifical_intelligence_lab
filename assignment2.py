import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data_path = 'C:/Users/saada/OneDrive/Desktop/AI/listing_data_publish.csv'
df = pd.read_csv(data_path)

# Display basic information about the dataset
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Select relevant columns for analysis
relevant_cols = ['building_age', 'total_floor_count', 'floor_no', 'room_count', 'size', 'furnished', 'heating_type', 'price']
df = df[relevant_cols]

# Check if there are any rows remaining after data cleaning
if df.empty:
    print("No data remaining after dropping missing values. Please check your dataset.")
else:
    # Convert categorical variables ('furnished', 'heating_type') to numerical using one-hot encoding
    df = pd.get_dummies(df, columns=['furnished', 'heating_type'])

    # Normalize numerical features using Min-Max scaling if there are rows remaining
    scaler = MinMaxScaler()
    df[['building_age', 'total_floor_count', 'floor_no', 'room_count', 'size']] = scaler.fit_transform(df[['building_age', 'total_floor_count', 'floor_no', 'room_count', 'size']])

    # Split data into features (X) and target (y)
    X = df.drop('price', axis=1)
    y = df['price']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build a deep learning model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer for regression task

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.2)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")

    # Save the trained model (optional)
    model.save('real_estate_valuation_model.h5')
