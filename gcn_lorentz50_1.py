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
data = df[['x', 'y', 'z']].values 
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
for epoch in range(1500):
    model.train()
    optimizer.zero_grad()
    
    out = model(x_train_tensor, adj_train_tensor)
    loss = criterion(out, x_train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 300 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

# ==========================================
# 5. Inference: Upscale to 0.005s Dataset
# ==========================================
N_new = 2 * N - 1 
adj_test_norm = create_line_graph_adjacency(N_new)
adj_test_tensor = torch.FloatTensor(adj_test_norm).to(device)

x_test = np.zeros((N_new, 3))
for i in range(N):
    x_test[i*2] = data[i] 
for i in range(N-1):
    x_test[i*2 + 1] = (data[i] + data[i+1]) / 2.0 

x_test_tensor = torch.FloatTensor(x_test).to(device)

model.eval()
with torch.no_grad():
    out_new = model(x_test_tensor, adj_test_tensor)

out_new_np = out_new.cpu().numpy()

# Enforce that the original points remain exactly the same as the ground truth
for i in range(N):
    out_new_np[i*2] = data[i]

# --- BULLETPROOF LENGTH ALIGNMENT ---
final_row_count = out_new_np.shape[0]
max_time = df['Time_s'].max() if 'Time_s' in df.columns else 5.0

time_s_new = np.linspace(0.0, max_time, num=final_row_count)

df_new = pd.DataFrame({
    'Time_s': time_s_new,
    'X': out_new_np[:, 0],
    'Y': out_new_np[:, 1],
    'Z': out_new_np[:, 2]
})

output_file = os.path.join(base_dir, 'Lorentz_attractor50_0_005.csv')
df_new.to_csv(output_file, index=False)
print(f"\nSuccess! Upscaled dataset saved as ({final_row_count} rows): {output_file}")

# ==========================================
# 6. Unified 3D Matplotlib Visualization
# ==========================================
print("Generating unified 3D trajectory plot...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 1. Plot the Original Dataset as a clean, continuous blue line
ax.plot(data[:, 0], data[:, 1], data[:, 2], 
         color="#0055ff", linestyle='-', linewidth=2.0, label='Original Path (0.01s continuous)')

# 2. Extract only the upscaled indices (odd rows) to isolate the generated midpoints
interpolated_indices = np.arange(1, N_new, 2)

# 3. Overlay the GCN generated points as discrete red dots
ax.scatter(out_new_np[interpolated_indices, 0], 
            out_new_np[interpolated_indices, 1], 
            out_new_np[interpolated_indices, 2], 
            color="#ff0e0e", s=10, marker='o', alpha=0.8, depthshade=False, label='GCN Generated Points (0.005s)')

# Customizing titles, labels, and view angles
ax.set_title("Superposition of Original Trajectory & GCN Upscaled Points", fontsize=14, pad=15)
ax.set_xlabel("X Space", fontsize=10)
ax.set_ylabel("Y Space", fontsize=10)
ax.set_zlabel("Z Space", fontsize=10)

# Legend configuration
ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
plt.show()