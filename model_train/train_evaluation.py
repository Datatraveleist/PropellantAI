import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pickle
from Fitting_plot import plot_fit_results_test

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

data_names = ['High_throughput_NEPE_f_ECs','High_throughput_GAP_f_ECs']
# 1. Data preprocessing
for data_name in data_names:
    df = pd.read_csv(f"../data/{data_name}.csv")
    df_sampled = df.sample(n=100000, random_state=42)  
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
        "log_mass(H/N)",
        # 'EMs_hf',
        # 'ECs_OB%_CO2',
        # 'ECs_OB%_CO',
        # 'ECs_C%',
        # 'ECs_H%',
        # 'ECs_O%',
        # 'ECs_N%'
    ]
    X = df[X_name]
    if data_name == 'High_throughput_NEPE_f_ECs':
        y = df[["c_Isp", "c_T_c", "c_Cstar"]]
    else:
        y = df[["Isp", "T_c", "Cstar"]]

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # Load the saved Scaler (according to your actual situation)
    with open("isp_X.pkl", "rb") as ex:
        scaler = pickle.load(ex)

    # Use the loaded scaler to transform the test set
    X_tensor_scaled = scaler.transform(X_tensor.numpy())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model
    learning_rate = 0.001
    hidden_dim = 256
    model = NeuralNet(input_dim=X_tensor.shape[1], hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load("trained_model.pth"))
    model.eval()

    # Convert the normalized input to Tensor and move to device
    X_tensor_scaled = torch.tensor(X_tensor_scaled, dtype=torch.float32).to(device)

    # Make predictions
    predictions = model(X_tensor_scaled)

    # Calculate MSE and R² for each target
    mse_values = {}
    r2_values = {}
    mae_values = {}
    y_tensor = y_tensor.cpu().numpy()
    # predictions = predictions.cpu().numpy()
    predictions = predictions.detach().cpu().numpy()

    for i, label in enumerate(["Isp", "T_c", "Cstar"]):
        mse_values[label] = mean_squared_error(y_tensor[:, i], predictions[:, i])
        r2_values[label] = r2_score(y_tensor[:, i], predictions[:, i])
        mae_values[label] = mean_absolute_error(y_tensor[:, i], predictions[:, i])
        plot_fit_results_test(data_name,
            label,
            y_tensor[:, i],
            predictions[:, i],
            sample_size_test=10000,
         )
    # Save results to txt file
    with open(f"evaluation_result/{data_name}.txt", "w") as f:
        f.write(f"Test Evaluation Results_{data_name}:\n")
        for label in ["Isp", "T_c", "Cstar"]:
            f.write(
                f"{label} - MSE: {mse_values[label]:.4f}, R²: {r2_values[label]:.4f}, MAE: {mae_values[label]:.4f}\n"
            )

    print(f"Test Evaluation Results_{data_name}:\n")
