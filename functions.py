import numpy as np
from scipy.integrate import solve_ivp
# from numpy import trapz

# Define the ODE system for pedestrian movement
def pedestrian_ode(t, l, c, s):
    # Default circumference
    circumference = 2*np.pi*3 + 4 * 2

    n = len(l)  # Number of pedestrians
    dl_dt = np.zeros(n)
    
    for i in range(n): 
        if i == n - 1:
            gap = l[0] - (l[i] - circumference)
        else:
            gap = l[i + 1] - l[i]
        if gap > s: 
            dl_dt[i] = c[i] * (1 - s / gap)
        else:
            dl_dt[i] = 0  # If too close, pedestrian stops
    
    return dl_dt

# def second_order_pedestrian_ode(t, y, c, s, tau):
#     """
#     Second-order pedestrian model:
#     y = [x_0, ..., x_n, u_0, ..., u_n]  (positions + velocities)
#     """
#     # Default circumference
#     circumference = 2*np.pi*3 + 4 * 2

#     n = len(y) // 2  # Number of pedestrians
#     x = y[:n]  # Positions
#     u = y[n:]  # Velocities

#     dxdt = u  # dx/dt = velocity
#     dudt = np.zeros(n)  # Initialize acceleration
    
#     for i in range(n): 
#         next_index = (i + 1) % n  # Wrap around for circular track
#         gap = (x[next_index] - x[i]) % circumference  # Circular gap
#         if gap > s:
#             desired_velocity = c * (1 - s / gap)  # Desired velocity based on gap
#             dudt[i] = (desired_velocity - u[i]) / tau  # Acceleration towards desired velocity
#         else:
#             dudt[i] = -u[i] / tau  # Deceleration if too close
        
#     return np.concatenate([dxdt, dudt])  # Return both position and velocity derivatives

def loss_function(real_positions, predicted_positions, t_eval):
    """
    j(t) = \frac{1}{N} \sum_{i=1}^N \frac{1}{2} (\hat{x}_i (t) - x_i (t))^2
    J = \int_0^T e^{-t/T} j(t) dt

    Compute the loss function J for pedestrian trajectory prediction.
    
    Parameters:
        params: Model parameters (first N values are c_i, last one is s)
        real_positions: Actual pedestrian positions (shape: [num_pedestrians, num_frames])
        t_eval: Time evaluation points (shape: [num_frames])
        initial_positions: Initial pedestrian positions (shape: [num_pedestrians])
    
    Returns:
        Loss value J
    """
    # Default circumference
    circumference = 2*np.pi*3 + 4 * 2

    num_pedestrians, num_frames = real_positions.shape
    T = t_eval[-1]  # Total simulation time

    # Compute squared error
    delta = np.abs(predicted_positions - real_positions) % circumference
    error = np.minimum(delta, circumference - delta)
    squared_errors = 0.5 * error ** 2  # Shape: [num_pedestrians, num_frames]

    # Apply e^(-t/T)
    time_weights = np.exp(-t_eval / T)  # Shape: [num_frames]
    weighted_errors = squared_errors * time_weights 

    # Integrate over time (using summation)
    dt = t_eval[1] - t_eval[0]
    loss_per_pedestrian = np.sum(weighted_errors, axis=1) * dt
    J = np.mean(loss_per_pedestrian)  # Average over all pedestrians

    return J

# Reset position to 0 at the starting point (3, 0)
def reset_laps(person_position):
    lap_positions = []
    for index in range(len(person_position)):
        lap_positions.append(person_position[index])
        if index < len(person_position) -1 and (person_position[index] > person_position[index + 1] + 20 or person_position[index] < person_position[index + 1] - 20):
            lap_positions.append(np.nan)
    return lap_positions