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
# 1. Load Data & Compute Ground Truth Gradients
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'Lorentz_attractor_5s_0.010_rate.csv') 

df = pd.read_csv(csv_path)
data = df[['X', 'Y', 'Z']].values 
N = len(data)
print(f"Loaded dataset with {N} nodes at 0.01s sampling rate.")

# Compute continuous ground truth derivatives for training GCN 1
# Dividing by 0.01 accounts for the time step Δt = 0.01s
derivatives_gt = np.gradient(data, 0.01, axis=0)

# Move both tracking datasets to the chosen device
x_train_tensor = torch.FloatTensor(data).to(device)
derivatives_train_tensor = torch.FloatTensor(derivatives_gt).to(device)

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

# Instantiate GCN 1 (Physics Judge) and GCN 2 (Upscaler)
gcn_derivative = TimeSeriesGCN().to(device)
gcn_upscaler = TimeSeriesGCN().to(device)

opt_derivative = optim.Adam(gcn_derivative.parameters(), lr=0.002)
opt_upscaler = optim.Adam(gcn_upscaler.parameters(), lr=0.002)
criterion = nn.MSELoss()

# ==========================================
# 3. Create Graph Adjacency Matrix Helper
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

# Create normalized adjacency matrix for the training nodes (0.01s coarse step)
adj_train_norm = create_line_graph_adjacency(N)
adj_train_tensor = torch.FloatTensor(adj_train_norm).to(device)

# ==========================================
# 4. Phase 1: Independent Network Pre-training
# ==========================================
print("\n🏋️ Pre-training GCN 1 (Derivative Operator) to map Positions -> Velocities...")
for epoch in range(1500):
    gcn_derivative.train()
    opt_derivative.zero_grad()
    
    out_deriv = gcn_derivative(x_train_tensor, adj_train_tensor)
    loss_deriv = criterion(out_deriv, derivatives_train_tensor)
    
    loss_deriv.backward()
    opt_derivative.step()
    if epoch % 500 == 0:
        print(f"  GCN 1 Epoch {epoch:04d} | Derivative Reconstruction MSE: {loss_deriv.item():.6f}")

print("\n🏋️ Pre-training GCN 2 (Upscaler) to reconstruct spatial paths...")
for epoch in range(1500):
    gcn_upscaler.train()
    opt_upscaler.zero_grad()
    
    out_upscale = gcn_upscaler(x_train_tensor, adj_train_tensor)
    loss_upscale = criterion(out_upscale, x_train_tensor)
    
    loss_upscale.backward()
    opt_upscaler.step()
    if epoch % 500 == 0:
        print(f"  GCN 2 Epoch {epoch:04d} | Spatial Identity MSE: {loss_upscale.item():.6f}")

# ==========================================
# 5. Phase 2: Pure-Neural Physics Feedback Loop
# ==========================================
K = 10  # Scaling factor (0.01s / 0.001s)
N_new = (N - 1) * K + 1 

# Generate the high-res 10x upscaled structural graph adjacency matrix for GCN 2
adj_test_norm = create_line_graph_adjacency(N_new)
adj_test_tensor = torch.FloatTensor(adj_test_norm).to(device)

# Initialize the expanded high-res feature array using linear transitions
x_test = np.zeros((N_new, 3))
for i in range(N - 1):
    start_val = data[i]
    end_val = data[i + 1]
    for j in range(K):
        alpha = j / K
        x_test[i * K + j] = (1 - alpha) * start_val + alpha * end_val
x_test[-1] = data[-1]
x_test_tensor = torch.FloatTensor(x_test).to(device)

print("\n🔄 Initializing Pure-Neural Physics Feedback Loop...")
print("GCN 1 will evaluate downsampled high-res anchor footprints to penalize GCN 2 errors.\n")

# Freeze GCN 1 completely so it acts as an immutable physical judge
gcn_derivative.eval() 

# Pre-calculate the "Gold Standard" derivatives on raw coarse data using GCN 1
with torch.no_grad():
    gold_standard_derivatives = gcn_derivative(x_train_tensor, adj_train_tensor)

# Run the feedback fine-tuning loop to optimize GCN 2
for loop_epoch in range(1200):
    gcn_upscaler.train()
    opt_upscaler.zero_grad()
    
    # Step A: GCN 2 generates the high-res 0.001s candidate coordinates
    upscaled_positions = gcn_upscaler(x_test_tensor, adj_test_tensor)
    
    # Step B: Slice out only the anchor locations (every 10th row) from GCN 2's output
    upscaled_anchors = upscaled_positions[::K]
    
    # Step C: Ask GCN 1 to calculate derivatives at these specific candidate positions
    predicted_derivatives = gcn_derivative(upscaled_anchors, adj_train_tensor)
    
    # Step D: Pure-Neural Derivative Error (Mismatches between physics profiles)
    derivative_error_loss = criterion(predicted_derivatives, gold_standard_derivatives)
    
    # Step E: Anchor Position Retention Error (Keeps GCN 2 glued to real anchor metrics)
    position_anchor_loss = criterion(upscaled_anchors, x_train_tensor)
    
    # Combined Loss function driving structural velocity alignment
    total_upscaler_loss = position_anchor_loss + 0.2 * derivative_error_loss
    
    total_upscaler_loss.backward()
    opt_upscaler.step()
    
    if loop_epoch % 200 == 0:
        print(f"  Loop Epoch {loop_epoch:04d} | Total Loss: {total_upscaler_loss.item():.6f} | Physics Deviation: {derivative_error_loss.item():.6f}")

# ==========================================
# 6. Final Inference and 6-Column Export
# ==========================================
gcn_upscaler.eval()
gcn_derivative.eval()

with torch.no_grad():
    # Generate final optimized fine-grained positions
    final_positions = gcn_upscaler(x_test_tensor, adj_test_tensor).cpu().numpy()
    
    # Force strict original anchor values one last time for spatial boundary security
    for i in range(N):
        final_positions[i * K] = data[i]
        
    # Map the COMPLETE high-res positions into GCN 1 to derive continuous 0.001s velocity data
    final_pos_tensor = torch.FloatTensor(final_positions).to(device)
    final_derivatives = gcn_derivative(final_pos_tensor, adj_test_tensor).cpu().numpy()

# Establish the precise chronological 0.001s intervals
time_s_new = np.array([i * 0.001 for i in range(final_positions.shape[0])])

# Compile the final 6-channel output DataFrame
df_final = pd.DataFrame({
    'Time_s': time_s_new,
    'X': final_positions[:, 0],
    'Y': final_positions[:, 1],
    'Z': final_positions[:, 2],
    'dX_dt': final_derivatives[:, 0],
    'dY_dt': final_derivatives[:, 1],
    'dZ_dt': final_derivatives[:, 2]
})

output_file = os.path.join(base_dir, 'Lorentz_attractor_6_channel_0_001.csv')
df_final.to_csv(output_file, index=False)
print(f"\n🚀 Success! 6-column physics-consistent dataset saved to: {output_file}")

# ==========================================
# 7. Verification Plots
# ==========================================
print("Generating tracking profiles...")

# --- Plot 1: Timeline Velocity & Tracking Traces (2D Subplots) ---
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

# Plot Spatial Trajectory Profile (X dimension example)
ax1.plot(df['Time_s'], data[:, 0], 'bo', label='Original Anchors (0.01s)', markersize=4)
ax1.plot(df_final['Time_s'], final_positions[:, 0], 'r-', alpha=0.8, label='Physics Upscaled Path (0.001s)')
ax1.set_ylabel('X Position', fontsize=11)
ax1.legend(loc='upper right')
ax1.set_title('Neural Feedback Optimization: Position & Derivative Verification')

# Plot Derivative Velocity Profile (dX_dt dimension example)
ax2.plot(df['Time_s'], derivatives_gt[:, 0], 'b.', alpha=0.4, label='Raw Ground Truth Δ Derivatives')
ax2.plot(df_final['Time_s'], final_derivatives[:, 0], 'g-', alpha=0.8, label='GCN 1 Continuous Velocity ($\dot{X}$)')
ax2.set_xlabel('Time (seconds)', fontsize=11)
ax2.set_ylabel('dX/dt (Velocity)', fontsize=11)
ax2.legend(loc='upper right')
plt.tight_layout()

# --- Plot 2: 3D Lorenz Attractor Trajectory Geometric Phase Space ---
fig2 = plt.figure(figsize=(10, 8))
ax3 = fig2.add_subplot(111, projection='3d')

# Plot the upscaled continuous curve generated by the dual GCNs
ax3.plot(final_positions[:, 0], final_positions[:, 1], final_positions[:, 2], color='#ff0e0e', linestyle='-', linewidth=1.2, alpha=0.85, label='GCN Upscaled Attractor (0.001s)')

# Scatter the raw, sparse coarse anchor positions to show how the path pins to them
ax3.plot(data[:, 0], data[:, 1], data[:, 2], color='#0055ff', linestyle='-', linewidth=1.2, alpha=0.85, label='Original Coarse Node Points (0.01s)')

ax3.set_title('GCN Neural Feedback Inferred Lorenz Attractor Manifold', fontsize=13, pad=15)
ax3.set_xlabel('X Dimension Space')
ax3.set_ylabel('Y Dimension Space')
ax3.set_zlabel('Z Dimension Space')
ax3.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

# Optimize initial camera positioning to view the manifold clearly
ax3.view_init(elev=20, azim=45)
plt.tight_layout()

# Render both windows sequentially
plt.show()