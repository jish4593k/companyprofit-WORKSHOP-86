import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Load the dataset
dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python/50_Startups.csv')

# Extract features and target variable
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# One-Hot Encoding for categorical variable
X = np.concatenate([X, pd.get_dummies(X[:, 3], prefix='State', drop_first=True)], axis=1)
X = np.delete(X, 3, axis=1)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Define a multiple linear regression model using PyTorch
class MultipleLinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(MultipleLinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model, loss function, and optimizer
model = MultipleLinearRegressionModel(input_size=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    Y_pred = model(X_train_tensor)

    # Compute the loss
    loss = criterion(Y_pred, Y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Convert the predictions back to numpy arrays
Y_test_pred = model(X_test_tensor).detach().numpy()

# Concatenate predictions and true labels
results = np.concatenate((Y_test_pred, Y_test.reshape(len(Y_test), 1)), axis=1)
print("Predictions vs. True labels:")
print(results)
