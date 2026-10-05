import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==============================================================================
# 0. Device Configuration (GPU Setup)
# ==============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple Silicon GPU acceleration (MPS)")
else:
    device = torch.device("cpu")
    print("⚠️ GPU backend not found, falling back to CPU")

# ==============================================================================
# 1. Load Original Data (Low-Resolution Dataset)
# ==============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'Lorentz_attractor_dropped_2pct.csv') 

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find the base dataset at: {csv_path}")

df = pd.read_csv(csv_path)
data = df[['X', 'Y', 'Z']].values 
N = len(data)
print(f"Loaded dataset with {N} nodes at 0.01s sampling rate.")

# ==============================================================================
# 2. Define Graph Convolutional Network Architecture
# ==============================================================================
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        support = self.linear(x)
        return torch.matmul(adj, support)

class TimeSeriesGCN(nn.Module):
    def __init__(self):
        super(TimeSeriesGCN, self).__init__()
        self.gcn1 = GCNLayer(3, 128)
        self.relu = nn.ReLU()
        self.gcn2 = GCNLayer(128, 3)

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.relu(x)
        x = self.gcn2(x, adj)
        return x

def create_line_graph_adjacency(num_nodes):
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        if i > 0: adj[i, i-1] = 1
        if i < num_nodes-1: adj[i, i+1] = 1
    
    adj += np.eye(num_nodes)
    deg = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

# ==============================================================================
# 3. Train the GCN Modality
# ==============================================================================
adj_train_norm = create_line_graph_adjacency(N)
x_train_tensor = torch.FloatTensor(data).to(device)
adj_train_tensor = torch.FloatTensor(adj_train_norm).to(device)

model = TimeSeriesGCN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002) 
criterion = nn.MSELoss()

print("\n--- Phase 1: Training GCN Neural Processor ---")
for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    
    out = model(x_train_tensor, adj_train_tensor)
    mse_loss = criterion(out, x_train_tensor)
    smoothness_loss = torch.mean(torch.diff(out, dim=0) ** 2)
    total_loss = mse_loss + 0.5 * smoothness_loss
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 400 == 0:
        print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.6f} (MSE: {mse_loss.item():.6f})")

# ==============================================================================
# 4. Inference & High-Resolution Upscaling (10x Interpolation)
# ==============================================================================
print("\n--- Phase 2: Running Trajectory Upscaling Inference (10x) ---")
K = 10  
N_new = (N - 1) * K + 1 

adj_test_norm = create_line_graph_adjacency(N_new)
adj_test_tensor = torch.FloatTensor(adj_test_norm).to(device)

x_test = np.zeros((N_new, 3))
for i in range(N - 1):
    start_val = data[i]
    end_val = data[i + 1]
    for j in range(K):
        alpha = j / K
        x_test[i * K + j] = (1 - alpha) * start_val + alpha * end_val
x_test[-1] = data[-1]

x_test_tensor = torch.FloatTensor(x_test).to(device)

model.eval()
with torch.no_grad():
    out_new = model(x_test_tensor, adj_test_tensor)

out_new_np = out_new.cpu().numpy()

# Force hard baseline constraints on anchor coordinates
for i in range(N):
    out_new_np[i * K] = data[i]

# Compute exact physical timeline array variables
time_s_new = np.array([i * 0.001 for i in range(N_new)])
dt = 0.001  # Structured sampling resolution passes directly to SINDy numerical stencils

# Optional: Save back explicitly if external logging records require it
df_new = pd.DataFrame({
    'Time_s': time_s_new, 'X': out_new_np[:, 0], 'Y': out_new_np[:, 1], 'Z': out_new_np[:, 2]
})
output_file = os.path.join(base_dir, 'Lorentz_attractor50_0_001.csv')
df_new.to_csv(output_file, index=False)
print(f"Upscaled states successfully preserved to disk at: {output_file}")

# ==============================================================================
# 5. SINDy System: Direct Gradient Numerical Stencils
# ==============================================================================
print("\n--- Phase 3: Executing SINDy Sparse Identification ---")
X_sindy = out_new_np
dX = np.zeros((N_new, 3))

# High-order quintic central difference stencils
for i in range(2, N_new - 2):
    dX[i, :] = (-X_sindy[i+2, :] + 8*X_sindy[i+1, :] - 8*X_sindy[i-1, :] + X_sindy[i-2, :]) / (12 * dt)

# High-order edge boundary corrections
dX[0, :] = (-3*X_sindy[0, :] + 4*X_sindy[1, :] - X_sindy[2, :]) / (2 * dt)
dX[1, :] = (-3*X_sindy[1, :] + 4*X_sindy[2, :] - X_sindy[3, :]) / (2 * dt)
dX[-2, :] = (3*X_sindy[-2, :] - 4*X_sindy[-3, :] + X_sindy[-4, :]) / (2 * dt)
dX[-1, :] = (3*X_sindy[-1, :] - 4*X_sindy[-2, :] + X_sindy[-3, :]) / (2 * dt)

# Constructing Mathematical Polynomial Feature Space Space
x_col = X_sindy[:, 0]
y_col = X_sindy[:, 1]
z_col = X_sindy[:, 2]

Theta = np.column_stack([
    np.ones(N_new),
    x_col, y_col, z_col,
    x_col**2, x_col*y_col,
    x_col*z_col, y_col**2,
    y_col*z_col, z_col**2
])
library_labels = ['1', 'x', 'y', 'z', 'x^2', 'xy', 'xz', 'y^2', 'yz', 'z^2']

# ==============================================================================
# 6. Sequential Thresholded Least Squares (STLS) Core Loop
# ==============================================================================
lambda_thresh = 0.88  
max_iterations = 15

Xi = np.linalg.lstsq(Theta, dX, rcond=None)[0]

for iteration in range(max_iterations):
    small_indices = np.abs(Xi) < lambda_thresh
    Xi[small_indices] = 0
    for var in range(3):
        big_indices = ~small_indices[:, var]
        Xi[big_indices, var] = np.linalg.lstsq(Theta[:, big_indices], dX[:, var], rcond=None)[0]

print("=========================================")
print("  SINDy Identified Equations (Combined)")
print("=========================================")
state_names = ['dX/dt', 'dY/dt', 'dZ/dt']
for var in range(3):
    eq_str = f"{state_names[var]} = "
    terms = []
    for term_idx, label in enumerate(library_labels):
        coef = Xi[term_idx, var]
        if np.abs(coef) > 1e-4:
            terms.append(f"{coef:.4f}*{label}" if label != '1' else f"{coef:.4f}")
    eq_str += " + ".join(terms).replace("+ -", "- ") if terms else "0"
    print(eq_str)
print("=========================================\n")

# ==============================================================================
# 7. Comparative Forecasting Out To 50 Seconds
# ==============================================================================
print("Running comparative 50-second ODE simulations...")
t_eval = np.arange(0.0, 50.0, 0.01)
initial_state = X_sindy[0, :]

def true_lorenz(t, state):
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    return [
        sigma * (state[1] - state[0]),
        state[0] * (rho - state[2]) - state[1],
        state[0] * state[1] - beta * state[2]
    ]

def discovered_sindy_ode(t, state):
    x, y, z = state
    theta_row = np.array([1, x, y, z, x**2, x*y, x*z, y**2, y*z, z**2])
    return theta_row @ Xi

sol_true = solve_ivp(true_lorenz, (0, 50), initial_state, t_eval=t_eval, method='RK45')
sol_sindy = solve_ivp(discovered_sindy_ode, (0, 50), initial_state, t_eval=t_eval, method='RK45')

# ==============================================================================
# 8. Complete Multi-Panel Execution Plots
# ==============================================================================
fig = plt.figure(figsize=(16, 7))

# Subplot A: Phase Space Reconstruction
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(sol_true.y[0], sol_true.y[1], sol_true.y[2], color='#0055ff', linewidth=1.5, label='True Lorenz Model')
ax1.plot(sol_sindy.y[0], sol_sindy.y[1], sol_sindy.y[2], color='#ff0e0e', linewidth=1.2, label='SINDy via GCN-Cleaned Data')
ax1.set_xlabel('X Axis')
ax1.set_ylabel('Y Axis')
ax1.set_zlabel('Z Axis')
ax1.set_title('3D Phase Space Trajectory Tracking')
ax1.legend()

# Subplot B: Temporal Trajectory Discrepancies
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(sol_true.t, sol_true.y[0], color='#0055ff', linewidth=1.5, label='True Model')
ax2.plot(sol_sindy.t, sol_sindy.y[0], color='#ff0e0e', linewidth=1.2, label='SINDy Engine')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('X Coordinate')
ax2.set_title('Temporal Dynamics Comparison (X Axis Tracking)')
ax2.set_xlim([0, 50])
ax2.legend()

plt.tight_layout()
plt.show()