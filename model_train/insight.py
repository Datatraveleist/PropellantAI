import os
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import shap
import pandas as pd
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
# from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import seaborn as sns
sns.set_theme(context='paper', style='whitegrid', palette='deep', 
              font='Arial', font_scale=1.8, color_codes=True, 
              rc={'lines.linewidth': 2, 'axes.grid': True,
                  'ytick.left': True, 'xtick.bottom': True, 
                  'font.weight': 'bold', 'axes.labelweight': 'bold'})
    # train_sample_size = 3000000
    # test_sample_size = 30000
def shap_explain(model_one, name, X_train_s_t, X_test_s_t, feature_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train_s_t = torch.Tensor(X_train_s_t).to(device)
    X_test_s_t = torch.Tensor(X_test_s_t).to(device)

    # Prevent the sample size from exceeding the data volume
    train_sample_size,test_sample_size = 3000000, 30000
    train_indices = np.random.choice(len(X_train_s_t), train_sample_size, replace=False)
    test_indices = np.random.choice(len(X_test_s_t), test_sample_size, replace=False)

    X_train_random = X_train_s_t[train_indices]
    X_test_random = X_test_s_t[test_indices]

    # Build SHAP explainer
    explainer = shap.GradientExplainer(model_one, X_train_random)
    shap_values = explainer.shap_values(X_test_random)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    # Custom colors
    colors = ['#3167b0', '#c00000']
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Create canvas
    fig, ax = plt.subplots(figsize=(12, 8))

    # First draw summary plot
    shap.summary_plot(shap_values, X_test_random.cpu().numpy(), feature_names=feature_names, 
                      cmap=custom_cmap, show=False)

    ax = plt.gca()
    ax.collections[0].set_alpha(0.2)  # Scatter point transparency
    x_min, x_max = ax.get_xlim()
    x_range = max(abs(x_min), abs(x_max))
    ax.set_xlim([-x_range*1.2, x_range])
    
    # Calculate Top 20 important features
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_shap_values)[::-1]
    top_n = 20
    top_feature_names = [feature_names[i] for i in sorted_idx[:top_n]]
    top_shap_values = mean_shap_values[sorted_idx[:top_n]]

    # Get the current y-axis order
    yticklabels = [tick.get_text() for tick in ax.get_yticklabels()]
    y_pos_map = {name: idx for idx, name in enumerate(yticklabels)}
    y_mapped_positions = [y_pos_map[name] for name in top_feature_names if name in y_pos_map]

    # Draw bar chart
    bar_colors = custom_cmap(np.linspace(1, 0, len(top_feature_names)))
    for bar_value, y_pos, bar_color in zip(top_shap_values, y_mapped_positions, bar_colors):
        if name == 'T_c':
            radio = 4
        elif name == 'Isp':
            radio = 3
        else:
            radio = 4
        ax.barh(y=y_pos, width=bar_value*radio*3, 
                left=ax.get_xlim()[0],  # Start from the leftmost side
                color=bar_color, alpha=0.2, edgecolor='black', height=0.6)

    # Adjust the outer frame
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color("k")
        ax.spines[spine].set_linewidth(2)

    # ax.invert_yaxis()  # Important features on top
    save_dir = 'insight_result'
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"shap_combined_plot_{name}.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    print(f"Overlay SHAP plot saved to {save_path}")

    plt.close(fig)

def load_data():
    file_path = "../data/input_modify_f_ECs.pkl"
    # print(file_path)
    with open(file_path, "rb") as f:
        df = pickle.load(f)
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

    # Read the saved Scaler (according to your actual situation)
    with open("isp_X.pkl", "rb") as ex:
        scaler = pickle.load(ex)
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    # feature_names = X_train.columns
    # Convert to tensor
    X_train_s_t = torch.Tensor(X_train.astype(np.float32))
    X_test_s_t = torch.Tensor(X_test.astype(np.float32))
    Y_train_t = torch.Tensor(y_train.to_numpy(dtype=np.float32))
    Y_test_t = torch.Tensor(y_test.to_numpy(dtype=np.float32))
    return X_train_s_t, X_test_s_t, Y_train_t, Y_test_t,X_name


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


class SingleOutputModel(nn.Module):
    def __init__(self, base_model, output_idx=0):
        super(SingleOutputModel, self).__init__()
        self.base_model = base_model
        self.output_idx = output_idx

    def forward(self, x):
        # The original output shape of base_model: (batch_size, 3)
        # We only take the output at output_idx (e.g., 0 => Isp), and keep the shape as (batch_size, 1)
        out_all = self.base_model(x)
        out_single = out_all[:, self.output_idx].unsqueeze(
            -1
        )  # => shape (batch_size, 1)
        return out_single


# Define model
X_train_s_t, X_test_s_t, Y_c_t_train_t, Y_c_t_test_t, feature_names = load_data()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001
hidden_dim = 256
model = NeuralNet(input_dim=X_train_s_t.shape[1], hidden_dim=hidden_dim).to(device)
model.load_state_dict(torch.load("trained_model.pth"))
# model_c_t = SingleOutputModel(model, output_idx=0).eval().to(device)
# model_isp = SingleOutputModel(model, output_idx=1).eval().to(device)
# model_cstar = SingleOutputModel(model, output_idx=2).eval().to(device)
print('Start')
names = ['Isp','T_c','Cstar']
for index, name in enumerate(names):
    # if index==1:
    #     break
    print(index)
    model_one = SingleOutputModel(model, output_idx=index).eval().to(device)
    shap_explain(
        model_one.to(device), name, X_train_s_t.to(device), X_test_s_t.to(device), feature_names
    )
