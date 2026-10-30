import numpy as np
import pandas as pd

# ==========================================
# 1. LOAD DATASETS
# ==========================================
excel_path = "/Users/Arya/Desktop/gcn_env/sindy_in_PSA/IEEE9BusSystem_loadflow_results.xlsx"
xls = pd.ExcelFile(excel_path)

nodes_df = pd.read_excel(excel_path, sheet_name="Nodes")
busbars_df = pd.read_excel(excel_path, sheet_name="Busbars")
connections_df = pd.read_excel(excel_path, sheet_name="Connections")

# Load generator time-series baselines
g1_df = pd.read_csv("/Users/Arya/Desktop/gcn_env/sindy_in_PSA/healthy datasets/G1_H.csv")
g2_df = pd.read_csv("/Users/Arya/Desktop/gcn_env/sindy_in_PSA/healthy datasets/G2_H.csv")
g3_df = pd.read_csv("/Users/Arya/Desktop/gcn_env/sindy_in_PSA/healthy datasets/G3_H.csv")

print("--- Data Successfully Loaded ---")

# ==========================================
# 2. EXTRACT STEADY-STATE INITIALIZATION
# ==========================================
# Extract generator initial conditions from G1, G2, G3 CSVs at t = 0
init_states = {}
for name, df in zip(["Gen1", "Gen2", "Gen3"], [g1_df, g2_df, g3_df]):
  init_states[name] = {
      "t0": df["t"].iloc[0],
      "fd_p": df["fd_p"].iloc[0],  # Field excitation voltage
      "Pm_pu": df["Pm_pu"].iloc[0],  # Mechanical power input (pu)
      "theta": df["theta"].iloc[0],  # Rotor angle (deg)
      "omega": df["omega"].iloc[0],  # Rotor speed (rad/s)
  }

print("\n[Generator 1 Initial Conditions from G1_H.csv]:")
for k, v in init_states["Gen1"].items():
  print(f"  {k}: {v}")

# Extract Bus Voltage Profiles from Busbars sheet
bus_profiles = busbars_df[
    ["Row", "Voltage Magnitude, pu", "Voltage Angle, deg"]
].copy()
print("\n[Busbar Voltage Profiles (First 3 Buses)]:")
print(bus_profiles.head(3))


# ==========================================
# 3. SETUP NETWORK TOPOLOGY & DAE TEMPLATE
# ==========================================
class IEEE9BusDAESystem:

  def __init__(self, busbars_df, connections_df, generator_inits):
    self.busbars = busbars_df
    self.connections = connections_df
    self.generator_inits = generator_inits
    self.system_frequency = 60.0  # Hz
    self.omega_s = 2 * np.pi * self.system_frequency  # Synchronous speed (377 rad/s)

  def initialize_state_vector(self):
    """Constructs the initial DAE state vector x0 combining generators and network nodes."""
    # Example state mapping per generator: [delta_i, omega_i, E_q_i', E_d_i']
    x0 = []
    for gen_name in ["Gen1", "Gen2", "Gen3"]:
      # Convert initial angle to radians
      delta_0 = np.radians(self.generator_inits[gen_name]["theta"])
      omega_0 = self.generator_inits[gen_name]["omega"]
      fd_0 = self.generator_inits[gen_name]["fd_p"]

      # Append generator states
      x0.extend([delta_0, omega_0, fd_0])
    return np.array(x0)

  def compute_algebraic_residuals(self, t, x, V_bus, theta_bus):
    """Placeholder for network algebraic power balance equations (Y_bus coupling)."""
    # Enforces P_gi - P_ld,i = sum(|V_i||V_j||Y_ij| cos(theta_ij - delta_i + delta_j))
    residuals = []
    # Add algebraic constraint calculations here during integration steps
    return residuals


# Instantiate and prepare system
dae_system = IEEE9BusDAESystem(busbars_df, connections_df, init_states)
x0 = dae_system.initialize_state_vector()

print(f"\nInitial State Vector x0 shape: {x0.shape}")
print("Initialized DAE Base Successfully.")