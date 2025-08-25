import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pickle

# 1. Data preprocessing
with open("../data/input_modify_f_ECs.pkl", "rb") as f:
    df = pickle.load(f)
# df = df.sample(n=10000, random_state=42)  
# 2. Define features and targets
X_name = [
    "Al",
    "C",
    "N",
    "O",
    "H",
    "Cl",
    "Hf",
    "S",
    "F",
    "Br",
    "I",
    "OB%_CO2",
    'OB%_CO',
    "log_mass(C/H)",
    "log_mass(C/O)",
    "log_mass(C/N)",
    "log_mass(H/C)",
    "log_mass(H/O)",
    "log_mass(H/N)"
    # 'EMs_hf',
    # 'ECs_OB%_CO2',
    # 'ECs_OB%_CO',
    # 'ECs_C%',
    # 'ECs_H%',
    # 'ECs_O%',
    # 'ECs_N%'
]
X = df[X_name]
y = df[["Isp", "T_c", "Cstar"]]
# Define X (input features) and y (target outputs)
feature_names = X.columns.tolist()
print("\nFeature Names:")
for feature in feature_names:
    print(feature)
# Standardize input features X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open("isp_X.pkl", "wb") as ex:
    pickle.dump(scaler, ex)
# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

# Convert to Tensor format
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create data loaders for batch loading
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
batch_size = 256  # Suitable batch size
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=24
)  # Use 24 workers
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=24)


# 2. Build neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)  # Output 3 values (Isp, T_c, Cstar)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Select device (prefer GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Select optimizer and loss function
learning_rate = 0.001  # Set a fixed learning rate
hidden_dim = 256  # Set the number of neurons in the hidden layer
model = NeuralNet(input_dim=X_train.shape[1], hidden_dim=hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()


# 4. Training function
def train_model(model, train_loader, optimizer, loss_fn, num_epochs=10):
    model.train()  # Set to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Move data to GPU
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # Zero gradients
            outputs = model(inputs)  # Forward pass
            loss = loss_fn(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


def evaluate_model(model, train_loader, test_loader):
    model.eval()  # Set to evaluation mode
    predictions_train, targets_train = [], []
    predictions_test, targets_test = [], []

    with torch.no_grad():
        # Evaluate on the training set
        for inputs, targets_batch in train_loader:
            inputs = inputs.to(device)
            targets_batch = targets_batch.to(device)
            outputs = model(inputs)
            predictions_train.append(outputs.cpu().numpy())
            targets_train.append(targets_batch.cpu().numpy())

        # Evaluate on the test set
        for inputs, targets_batch in test_loader:
            inputs = inputs.to(device)
            targets_batch = targets_batch.to(device)
            outputs = model(inputs)
            predictions_test.append(outputs.cpu().numpy())
            targets_test.append(targets_batch.cpu().numpy())

    # Concatenate predictions and targets for both training and testing
    predictions_train = np.concatenate(predictions_train, axis=0)
    targets_train = np.concatenate(targets_train, axis=0)
    predictions_test = np.concatenate(predictions_test, axis=0)
    targets_test = np.concatenate(targets_test, axis=0)

    # Calculate MSE, R², and MAE for each target
    mse_values_train = {}
    r2_values_train = {}
    mae_values_train = {}
    mse_values_test = {}
    r2_values_test = {}
    mae_values_test = {}

    for i, label in enumerate(["Isp", "T_c", "Cstar"]):
        # For training data
        mse_values_train[label] = mean_squared_error(
            targets_train[:, i], predictions_train[:, i]
        )
        r2_values_train[label] = r2_score(targets_train[:, i], predictions_train[:, i])
        mae_values_train[label] = mean_absolute_error(
            targets_train[:, i], predictions_train[:, i]
        )
        # plot(targets_train[:, i],predictions_train[:, i],mae_values_train[label] ,mse_values_train[label],r2_values_train[label],label,train)
        # For test data
        mse_values_test[label] = mean_squared_error(
            targets_test[:, i], predictions_test[:, i]
        )
        r2_values_test[label] = r2_score(targets_test[:, i], predictions_test[:, i])
        mae_values_test[label] = mean_absolute_error(
            targets_test[:, i], predictions_test[:, i]
        )
        # plot(targets_train,predictions_train)
        plot_fit_results(
            label,
            targets_train[:, i],
            predictions_train[:, i],
            targets_test[:, i],
            predictions_test[:, i],
            sample_size_train=10000,
            sample_size_test=3000,
        )
    # Print evaluation results for training and test sets
    print(f"Training Set Evaluation Results:")
    for label in ["Isp", "T_c", "Cstar"]:
        print(
            f"{label} - Training MSE: {mse_values_train[label]:.4f}, R²: {r2_values_train[label]:.4f}, MAE: {mae_values_train[label]:.4f}"
        )

    print(f"Test Set Evaluation Results:")
    for label in ["Isp", "T_c", "Cstar"]:
        print(
            f"{label} - Test MSE: {mse_values_test[label]:.4f}, R²: {r2_values_test[label]:.4f}, MAE: {mae_values_test[label]:.4f}"
        )

    # Return all the evaluation metrics for both training and test sets

    return (
        mse_values_train,
        r2_values_train,
        mae_values_train,
        mse_values_test,
        r2_values_test,
        mae_values_test,
    )


# 6. Train the final model
train_model(model, train_loader, optimizer, loss_fn, num_epochs=20)

# Evaluate the model
(
    mse_values_train,
    r2_values_train,
    mae_values_train,
    mse_values_test,
    r2_values_test,
    mae_values_test,
) = evaluate_model(model, train_loader, test_loader)

# 7. Save model and evaluation results
torch.save(model.state_dict(), "trained_model.pth")  # Save the trained model

# Save training and test set evaluation results to a text file
with open("evaluation_HTPB.txt", "w") as f:
    f.write("Training Set Evaluation Results:\n")
    for label in ["Isp", "T_c", "Cstar"]:
        f.write(
            f"{label} - Training MSE: {mse_values_train[label]:.4f}, R²: {r2_values_train[label]:.4f}, MAE: {mae_values_train[label]:.4f}\n"
        )

    f.write("\nTest Set Evaluation Results:\n")
    for label in ["Isp", "T_c", "Cstar"]:
        f.write(
            f"{label} - Test MSE: {mse_values_test[label]:.4f}, R²: {r2_values_test[label]:.4f}, MAE: {mae_values_test[label]:.4f}\n"
        )
