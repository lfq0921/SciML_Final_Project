import numpy as np
from scipy.integrate import solve_ivp

def get_pendulum_data(t_span=(0, 50), num_points=1000):
    """Generate ground truth data using RK45 solver."""
    g_L, b, F, omega = 1.0, 0.5, 1.2, 2/3
    
    def dynamics(t, y):
        theta, omega_theta = y
        return [omega_theta, F * np.cos(omega * t) - b * omega_theta - g_L * np.sin(theta)]
        
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    sol = solve_ivp(dynamics, t_span, [0.2, 0.0], t_eval=t_eval, method='RK45')
    
    return sol.t, sol.y[0], sol.y[1]

def get_euler_data(t_data):
    """Generate data using Forward Euler for comparison."""
    dt = t_data[1] - t_data[0]
    N = len(t_data)
    theta, omega_v = np.zeros(N), np.zeros(N)
    
    theta[0], omega_v[0] = 0.2, 0.0
    g_L, b, F, omega = 1.0, 0.5, 1.2, 2/3
    
    for i in range(N - 1):
        theta[i+1] = theta[i] + omega_v[i] * dt
        accel = F * np.cos(omega * t_data[i]) - b * omega_v[i] - g_L * np.sin(theta[i])
        omega_v[i+1] = omega_v[i] + accel * dt
        
    return theta