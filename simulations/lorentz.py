import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # (Optional, or just omit it entirely)

def lorenz_simulation():
    # Define the standard Lorenz parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    
    # 1. Set sampling rate and time span (0 to 5 seconds)
    sampling_interval = 0.01
    end_time = 20.0
    t_eval = np.arange(0, end_time + sampling_interval / 2.0, sampling_interval)
    tspan = (0, end_time)
    
    # Initial conditions [x, y, z]
    initial_state = [1.0, 1.0, 1.0]
    
    # 2. Solve the system using SciPy's solve_ivp (equivalent to MATLAB's ode45)
    sol = solve_ivp(
        fun=lambda t, s: lorenz_system(t, s, sigma, rho, beta),
        t_span=tspan,
        y0=initial_state,
        t_eval=t_eval,
        method='RK45'
    )
    
    t = sol.t
    x = sol.y[0]
    y = sol.y[1]
    z = sol.y[2]
    
    # --- Calculate Time Derivatives ---
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    
    # 3. Create a DataFrame and export to CSV
    filename = f"Lorentz_attractor_{int(end_time)}s_{sampling_interval:.3f}_rate.csv"
    lorenz_dataset = pd.DataFrame({
        'Time_s': t,
        'X': x,
        'Y': y,
        'Z': z,
        'dX_dt': dx_dt,
        'dY_dt': dy_dt,
        'dZ_dt': dz_dt
    })
    lorenz_dataset.to_csv(filename, index=False)
    print(f"Dataset successfully saved to current folder as: {filename}")
    
    # --- Plotting the Time Derivatives ---
    plt.figure(figsize=(10, 4), facecolor='w')
    plt.plot(t, dx_dt, linewidth=1.5, color='#D95319', label='dX/dt (x_dot)')
    plt.plot(t, dy_dt, linewidth=1.5, color='#77AC30', label='dY/dt (y_dot)')
    plt.plot(t, dz_dt, linewidth=1.5, color='#7E2F8E', label='dZ_dt (z_dot)')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Derivatives')
    plt.title('Time Derivatives (dx/dt, dy/dt, dz/dt)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    # --- Optional: Create the Animated Plot ---
    animate_attractor(x, y, z, sampling_interval)

# Local function defining the Lorenz system differential equations
def lorenz_system(t, s, sigma, rho, beta):
    x, y, z = s
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Local function to animate the trajectory over time
def animate_attractor(x, y, z, sampling_interval):
    plt.ion() # Turn on interactive mode for animation
    fig = plt.figure(figsize=(8, 6), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    
    # Style the background like MATLAB's dark theme
    ax.set_facecolor('black')
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Lorenz Attractor Evolution ({sampling_interval:.3f}s steps)', color='white')
    
    # Set axis limits so the frame stays steady
    ax.set_xlim([min(x), max(x)])
    ax.set_ylim([min(y), max(y)])
    ax.set_zlim([min(z), max(z)])
    
    total_points = len(x)
    skip_step = max(1, round(total_points / 200))
    
    # Initialize empty line object
    line, = ax.plot([], [], [], color='cyan', linewidth=1.2)
    
    # Animation loop
    for k in range(0, total_points, skip_step):
        line.set_data(x[:k], y[:k])
        line.set_3d_properties(z[:k])
        plt.draw()
        plt.pause(0.01)
        
    # Draw final frame completely
    line.set_data(x, y)
    line.set_3d_properties(z)
    plt.draw()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    lorenz_simulation()