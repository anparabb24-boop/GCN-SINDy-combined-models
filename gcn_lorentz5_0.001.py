import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
print(f"Loaded dataset with {N} nodes at 0.01s sampling rate.")

# ==========================================
# 2. Define the Graph Convolutional Network
# ==========================================
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

# ==========================================
# 3. Create Graph Adjacency Matrix (Training)
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

x_train_tensor = torch.FloatTensor(data).to(device)
adj_train_tensor = torch.FloatTensor(adj_train_norm).to(device)

# ==========================================
# 4. Train the Model
# ==========================================
model = TimeSeriesGCN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002) 
criterion = nn.MSELoss()

print("Training GCN on GPU...")
for epoch in range(2000): # Bumped total epochs to 2000 for convergence
    model.train()
    optimizer.zero_grad()
    
    out = model(x_train_tensor, adj_train_tensor)
    
    # 1. Base structural reconstruction loss
    mse_loss = criterion(out, x_train_tensor)
    
    # 2. Smoothness penalty (penalizes sharp Euclidean jumps between adjacent sequence elements)
    smoothness_loss = torch.mean(torch.diff(out, dim=0) ** 2)
    
    # Combine losses (alpha scaling factor of 0.5 works great for high resolution trajectories)
    total_loss = mse_loss + 0.5 * smoothness_loss
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 400 == 0:
        print(f"Epoch {epoch} | Total Loss: {total_loss.item():.6f} (MSE: {mse_loss.item():.6f})")

# ==========================================
# 5. Inference: Upscale to 0.001s Dataset (10x Upscaling)
# ==========================================
K = 10  # Scaling factor (0.01s / 0.001s)

# Total new points calculation
N_new = (N - 1) * K + 1 

# Generate the massive 10x upscaled structural adjacency matrix
adj_test_norm = create_line_graph_adjacency(N_new)
adj_test_tensor = torch.FloatTensor(adj_test_norm).to(device)

# Initialize the expanded feature tensor array
x_test = np.zeros((N_new, 3))

# Linearly interpolate the initial positions for all 9 midpoints between anchor blocks
for i in range(N - 1):
    start_val = data[i]
    end_val = data[i + 1]
    
    # Fill the block of 10 elements with a smooth linear transition blend
    for j in range(K):
        alpha = j / K
        x_test[i * K + j] = (1 - alpha) * start_val + alpha * end_val

# Assign the very last absolute anchor point
x_test[-1] = data[-1]

# Convert and push your initialization tensor matrix to your Mac GPU
x_test_tensor = torch.FloatTensor(x_test).to(device)

model.eval()
with torch.no_grad():
    out_new = model(x_test_tensor, adj_test_tensor)

out_new_np = out_new.cpu().numpy()

# Enforce original 0.01s ground truth anchor nodes stay perfectly locked
for i in range(N):
    out_new_np[i * K] = data[i]

# --- BULLETPROOF LENGTH ALIGNMENT ---
final_row_count = out_new_np.shape[0]
max_time = df['Time_s'].max() if 'Time_s' in df.columns else 5.0

# --- CHANGE THIS IN YOUR GCN SCRIPT (SECTION 5) ---
final_row_count = out_new_np.shape[0]

# INSTEAD OF: time_s_new = np.linspace(0.0, max_time, num=final_row_count)
# USE THIS: Generate exact 0.001s steps so the intervals match the physics
time_s_new = np.array([i * 0.001 for i in range(final_row_count)])

df_new = pd.DataFrame({
    'Time_s': time_s_new,
    'X': out_new_np[:, 0],
    'Y': out_new_np[:, 1],
    'Z': out_new_np[:, 2]
})

output_file = os.path.join(base_dir, 'Lorentz_attractor50_0_001.csv')
df_new.to_csv(output_file, index=False)
print(f"\nSuccess! 10x Upscaled dataset saved as ({final_row_count} rows): {output_file}")

# ==========================================
# 6. Unified 3D Matplotlib Visualization
# ==========================================
print("Generating unified 3D trajectory plot...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 1. Plot the Original Dataset as a thick blue line
ax.plot(data[:, 0], data[:, 1], data[:, 2], 
        color="#0055ff", linestyle='-', linewidth=2.5, label='Original Path (0.01s)')

# 2. Plot the Full Upscaled GCN Dataset as a thinner red line right on top
# This connects all 1,001 points (anchors + upscaled steps) continuously
ax.plot(out_new_np[:, 0], out_new_np[:, 1], out_new_np[:, 2], 
        color="#ff0e0e", linestyle='-', linewidth=1.2, alpha=0.9, label='GCN Upscaled Path (0.001s)')

# Customizing titles, labels, and view angles
ax.set_title("Superposition of Original & 10x Upscaled Continuous Trajectories", fontsize=14, pad=15)
ax.set_xlabel("X Space", fontsize=10)
ax.set_ylabel("Y Space", fontsize=10)
ax.set_zlabel("Z Space", fontsize=10)

# Legend configuration
ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
plt.show()