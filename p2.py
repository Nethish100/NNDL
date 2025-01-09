import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a neural network with 3 hidden layers using Input layer
model = Sequential([
    Input(shape=(X.shape[1],)),                          # Input layer
    Dense(64, activation="relu"),                         # Hidden layer 1
    Dense(32, activation="relu"),                         # Hidden layer 2
    Dense(16, activation="relu"),                         # Hidden layer 3
    Dense(1, activation="sigmoid")                        # Output layer
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_classes)

# Display results
print(f"Test Accuracy: {accuracy:.4f}")
