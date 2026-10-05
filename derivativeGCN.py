import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 0. Device Configuration (Mac GPU Setup)
# ==========================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple Silicon GPU acceleration (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Using Nvidia GPU acceleration (CUDA)")
else:
    device = torch.device("cpu")
    print("⚠️ GPU backend not found, falling back to CPU")

# ==========================================
# 1. Load Data (0.01s High-Resolution Dataset)
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'Lorentz_attractor_5s_0.010_rate.csv')

df = pd.read_csv(csv_path)
data = df[['X', 'Y', 'Z']].values
N = len(data)
dt_coarse = 0.01
print(f"Loaded dataset with {N} nodes at {dt_coarse}s sampling rate.")

# ==========================================
# 2. Finite-difference pseudo-labels for derivatives
#    (this is the ONLY source of derivative information we have,
#     since true dX/dt, dY/dt, dZ/dt are unknown)
# ==========================================
def finite_difference_derivative(y, dt):
    """
    4th-order accurate central difference in the interior,
    2nd-order one-sided stencils at the two boundary points on each side.
    y: (N, C) array, uniformly spaced by dt.
    Returns dy/dt with the same shape.
    """
    n = y.shape[0]
    dydt = np.zeros_like(y, dtype=np.float64)

    if n >= 5:
        dydt[2:n-2] = (-y[4:n] + 8*y[3:n-1] - 8*y[1:n-3] + y[0:n-4]) / (12.0 * dt)

    if n >= 3:
        dydt[1] = (y[2] - y[0]) / (2.0 * dt)
        dydt[n-2] = (y[n-1] - y[n-3]) / (2.0 * dt)

    if n >= 3:
        dydt[0] = (-3*y[0] + 4*y[1] - y[2]) / (2.0 * dt)
        dydt[n-1] = (3*y[n-1] - 4*y[n-2] + y[n-3]) / (2.0 * dt)
    elif n == 2:
        dydt[0] = dydt[1] = (y[1] - y[0]) / dt

    return dydt

deriv_labels_coarse = finite_difference_derivative(data, dt_coarse)  # (N, 3), pseudo ground truth

# ==========================================
# 3. Graph adjacency (line-graph construction)
# ==========================================
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

adj_train_norm = create_line_graph_adjacency(N)

# ==========================================
# 4. Networks
# ==========================================
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        support = self.linear(x)
        return torch.matmul(adj, support)

class TimeSeriesGCN(nn.Module):
    """Coordinate upscaler."""
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

class DerivativeGCN(nn.Module):
    """Learns coordinates -> (dX/dt, dY/dt, dZ/dt), denoising the raw FD estimate."""
    def __init__(self):
        super(DerivativeGCN, self).__init__()
        self.gcn1 = GCNLayer(3, 128)
        self.gcn2 = GCNLayer(128, 128)
        self.relu = nn.ReLU()
        self.gcn3 = GCNLayer(128, 3)

    def forward(self, x, adj):
        h = self.relu(self.gcn1(x, adj))
        h = self.relu(self.gcn2(h, adj))
        return self.gcn3(h, adj)

# ==========================================
# 5. Train the coordinate upscaler
# ==========================================
x_train_tensor = torch.FloatTensor(data).to(device)
adj_train_tensor = torch.FloatTensor(adj_train_norm).to(device)

coord_model = TimeSeriesGCN().to(device)
coord_optimizer = optim.Adam(coord_model.parameters(), lr=0.002)
mse = nn.MSELoss()

print("\nTraining coordinate upscaling GCN...")
for epoch in range(2000):
    coord_model.train()
    coord_optimizer.zero_grad()
    out = coord_model(x_train_tensor, adj_train_tensor)
    mse_loss = mse(out, x_train_tensor)
    smooth_loss = torch.mean(torch.diff(out, dim=0) ** 2)
    loss = mse_loss + 0.5 * smooth_loss
    loss.backward()
    coord_optimizer.step()
    if epoch % 400 == 0:
        print(f"  [coord] epoch {epoch} | loss {loss.item():.6f}")

# ==========================================
# 6. Train the derivative GCN, supervised by FD pseudo-labels
# ==========================================
deriv_labels_tensor = torch.FloatTensor(deriv_labels_coarse).to(device)

deriv_model = DerivativeGCN().to(device)
deriv_optimizer = optim.Adam(deriv_model.parameters(), lr=0.002)

deriv_mean = deriv_labels_tensor.mean(dim=0, keepdim=True)
deriv_std = deriv_labels_tensor.std(dim=0, keepdim=True) + 1e-8
deriv_labels_norm = (deriv_labels_tensor - deriv_mean) / deriv_std

print("\nTraining derivative GCN (supervised by finite-difference pseudo-labels)...")
for epoch in range(3000):
    deriv_model.train()
    deriv_optimizer.zero_grad()

    pred_norm = deriv_model(x_train_tensor, adj_train_tensor)

    fd_loss = mse(pred_norm, deriv_labels_norm)
    smooth_loss = torch.mean(torch.diff(pred_norm, dim=0) ** 2)

    loss = fd_loss + 0.05 * smooth_loss
    loss.backward()
    deriv_optimizer.step()

    if epoch % 500 == 0:
        print(f"  [deriv] epoch {epoch} | loss {loss.item():.6f} (fd_mse: {fd_loss.item():.6f})")

# ==========================================
# 7. Inference: upscale coordinates to 0.001s, then predict derivatives there
# ==========================================
K = 10
N_new = (N - 1) * K + 1

adj_test_norm = create_line_graph_adjacency(N_new)
adj_test_tensor = torch.FloatTensor(adj_test_norm).to(device)

x_test = np.zeros((N_new, 3))
for i in range(N - 1):
    start_val, end_val = data[i], data[i + 1]
    for j in range(K):
        alpha = j / K
        x_test[i * K + j] = (1 - alpha) * start_val + alpha * end_val
x_test[-1] = data[-1]
x_test_tensor = torch.FloatTensor(x_test).to(device)

coord_model.eval()
with torch.no_grad():
    coords_fine = coord_model(x_test_tensor, adj_test_tensor)
    coords_fine_np = coords_fine.cpu().numpy()
    for i in range(N):
        coords_fine_np[i * K] = data[i]
    coords_fine_tensor = torch.FloatTensor(coords_fine_np).to(device)

deriv_model.eval()
with torch.no_grad():
    deriv_fine_norm = deriv_model(coords_fine_tensor, adj_test_tensor)
    deriv_fine = deriv_fine_norm * deriv_std + deriv_mean  # de-normalize
    deriv_fine_np = deriv_fine.cpu().numpy()

# ==========================================
# 8. Save results — just the X, Y, Z derivatives
# ==========================================
dt_fine = 0.001
time_s_new = np.array([i * dt_fine for i in range(N_new)])

df_out = pd.DataFrame({
    'Time_s': time_s_new,
    'dX_dt': deriv_fine_np[:, 0],
    'dY_dt': deriv_fine_np[:, 1],
    'dZ_dt': deriv_fine_np[:, 2],
})
output_file = os.path.join(base_dir, 'Lorentz_derivatives_0_001.csv')
df_out.to_csv(output_file, index=False)
print(f"\nSuccess! Derivative dataset saved as ({N_new} rows): {output_file}")