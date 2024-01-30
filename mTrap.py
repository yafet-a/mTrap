''' 
MTrap project
Import python code for multiprocessing
from multiprocessing import Pool

'''
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
from scipy.integrate import solve_ivp

# Constants
mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability
I = 500  # Current through the coil in Ampere
R = 0.025  # Coil radius in meter (2.5 cm)

# Function to calculate magnetic field components
def magnetic_field(r, z, Z0):
    """
    Calculate the magnetic field components B_r and B_z for a single coil.
    :param r: radial distance from the z-axis (m)
    :param z: z-coordinate (m)
    :param Z0: z-coordinate of the coil center (m)
    :return: B_r, B_z - radial and axial components of the magnetic field (T)
    """
    z_rel = z - Z0
    if r == 0:
        # Special case for the z-axis
        B_r = 0
        B_z = mu_0 * I * R**2 / (2 * ((R**2) + (z_rel**2))**(3/2))
    else:
        m = 4 * R * r / ((R + r)**2 + z_rel**2)
        E = ellipe(m)
        K = ellipk(m)
        common_factor = mu_0 * I / (2 * np.pi)

        B_r = common_factor * z_rel / (r * np.sqrt((R + r)**2 + z_rel**2)) * ((R**2 + r**2 + z_rel**2) / ((R - r)**2 + z_rel**2) * E - K)
        B_z = common_factor / np.sqrt((R + r)**2 + z_rel**2) * ((R**2 - r**2 - z_rel**2) / ((R - r)**2 + z_rel**2) * E + K)
    
    return B_r, B_z

# Test the magnetic field calculation at a specific point
test_r = 0.01  # 1 cm from z-axis
test_z = 0.05  # 5 cm along the z-axis
test_Z0 = -0.1  # Coil center at z=-10 cm

B_r_test, B_z_test = magnetic_field(test_r, test_z, test_Z0)
B_r_test, B_z_test  # Return the magnetic field components for testing

# Function to calculate total magnetic field magnitude
def total_magnetic_field_magnitude(r, z):
    """
    Calculate the total magnetic field magnitude for a given r and z.
    :param r: radial distance from the z-axis (m)
    :param z: z-coordinate (m)
    :return: Total magnetic field magnitude (T)
    """
    # Coil centers
    Z0_1 = -0.1  # -10 cm
    Z0_2 = 0.1   # 10 cm
    constant_B_field = 1  # Constant magnetic field of 1 Tesla in the +z-direction

    # Calculate magnetic field components for each coil
    B_r1, B_z1 = magnetic_field(r, z, Z0_1)
    B_r2, B_z2 = magnetic_field(r, z, Z0_2)

    # Total magnetic field components
    B_r_total = B_r1 + B_r2
    B_z_total = B_z1 + B_z2 + constant_B_field

    # Total magnetic field magnitude
    B_total_magnitude = np.sqrt(B_r_total**2 + B_z_total**2)
    return B_total_magnitude
    

# Prepare data for plotting
z_values = np.linspace(-0.15, 0.15, 500)  # z-axis interval between -15 cm and 15 cm
radii = np.linspace(0, 0.024, 5)  # 5 radii from 0 cm to 2.4 cm

# # Plotting
# plt.figure(figsize=(10, 6))
# for r in radii:
#     B_magnitudes = [total_magnetic_field_magnitude(r, z) for z in z_values]
#     plt.plot(z_values, B_magnitudes, label=f'r = {r:.3f} m')

# plt.xlabel('z-coordinate (m)')
# plt.ylabel('Magnetic Field Magnitude (T)')
# plt.title('Magnetic Field Magnitude Along the z-axis for Different Radii')
# plt.legend()
# plt.grid(True)
# plt.show()

"""Part 2: Begin"""
from scipy.constants import e as e_charge, c, m_e, eV
from scipy.constants import epsilon_0  # Vacuum permittivity

# Define tau for the radiation reaction force factor
tau = (e_charge**2) / (6 * np.pi * epsilon_0 * m_e * (c**3))
# print(f'tau: {tau}')
#Parameter Scanning setup. Setting up the grid for paramter scanning by defining the ranges for the emission angles and radial distances
# Range of emission angles (in degrees) and radial distances (in meters)
emission_angles = np.linspace(76, 89, num=10)  # Adjust the num parameter as needed
radial_distances = np.linspace(0, 0.024, num=7)  # Adjust the num parameter as needed

# Grid for parameter scanning
parameters = [(r, theta) for r in radial_distances for theta in emission_angles]

def lorentz_dirac(t, y):
    x, y, z, vx, vy, vz = y
    r = np.sqrt(x**2 + y**2)
    v = np.array([vx, vy, vz])
    v_squared = np.dot(v, v)
    
    # Prevent superluminal speeds
    if v_squared >= c**2:
        return np.zeros_like(y)
    
    gamma = 1 / np.sqrt(1 - v_squared / c**2)
    # print(f'gamma: {gamma}')
    
    # Magnetic field calculation
    B_r1, B_z1 = magnetic_field(r, z, -0.1)  # Coil 1 at Z0 = -0.1 m
    B_r2, B_z2 = magnetic_field(r, z, 0.1)   # Coil 2 at Z0 = 0.1 m
    B = np.array([B_r1 + B_r2, 0, B_z1 + B_z2 + 1])  # Total magnetic field
    
    # Cyclotron frequency omega
    omega = (e_charge / m_e) * B
    
    # mu vector
    mu = omega - (np.dot(v, omega) / c**2) * v
    
    # M magnitude
    M_magnitude = 1 + gamma**4 * tau**2 * np.dot(mu, mu)
    
    # Lorentz-Dirac acceleration components
    ax = ((1 + mu[0]**2 * gamma**4 * tau**2) * (omega[2]*vy - omega[1]*vz) +
         (mu[0]*mu[2]*gamma**4*tau**2 - mu[1]*gamma**2*tau) * (omega[1]*vx - omega[0]*vy) +
         (mu[0]*mu[1]*gamma**4*tau**2 + mu[2]*gamma**2*tau) * (omega[0]*vz - omega[2]*vx)) / (gamma * M_magnitude)
    
    ay = ((1 + mu[1]**2 * gamma**4 * tau**2) * (omega[0]*vz - omega[2]*vx) +
         (mu[1]*mu[2]*gamma**4*tau**2 + mu[0]*gamma**2*tau) * (omega[1]*vx - omega[0]*vy) +
         (mu[0]*mu[1]*gamma**4*tau**2 - mu[2]*gamma**2*tau) * (omega[2]*vy - omega[1]*vz)) / (gamma * M_magnitude)
    
    az = ((1 + mu[2]**2 * gamma**4 * tau**2) * (omega[1]*vx - omega[0]*vy) +
         (mu[1]*mu[2]*gamma**4*tau**2 - mu[0]*gamma**2*tau) * (omega[0]*vz - omega[2]*vx) +
         (mu[0]*mu[2]*gamma**4*tau**2 + mu[1]*gamma**2*tau) * (omega[2]*vy - omega[1]*vx)) / (gamma * M_magnitude)
    
    return [vx, vy, vz, ax, ay, az]


# Compute constants outside of the function
kinetic_energy_eV = 18.6e3
lorentz_factor = (kinetic_energy_eV * eV) / (m_e * c**2) + 1
speed = c * np.sqrt(1 - 1 / (lorentz_factor**2))

def simulate_electron_motion(r0, emission_angle):
    # Use pre-computed constants
    global kinetic_energy_eV, lorentz_factor, speed
    
    # print(f'speed: {speed}')
    
    # Convert angle to radians and calculate initial velocity components
    angle_rad = np.radians(emission_angle)
    # Assuming the electron is emitted in the xy-plane, we can set vx and vy based on the emission angle
    vz = speed * np.cos(angle_rad)
    vy = speed * np.sin(angle_rad)
    vx = 0  # Initial velocity in the z-direction is set to 0

    # Initial conditions
    initial_conditions = [r0, 0, 0, vx, vy, vz]

    # Time span for the simulation (in seconds)
    t0, tf = 0, 1e-6 # Adjust this time span based on your requirements
    max_steps = 1e-11
    
    # Define the event function for bounce detection
    def z_crossing(t, y):
        return y[2]  # y[2] is the z-coordinate

    # The event triggers when the z-coordinate changes sign
    z_crossing.terminal = True
    z_crossing.direction = -1  # Will trigger on any crossing of z=0

    # Solve the Lorentz-Dirac equations with event handling for bounce
    solution = solve_ivp(lorentz_dirac, [t0, tf], initial_conditions, method='RK45', 
                         max_step=max_steps, events=z_crossing)
    
    # Check if a bounce occurred (i.e., if there is a z-crossing event)
    if solution.t_events[0].size > 0:
        # Time of the first z=0 crossing, assuming the electron starts at z>0
        time_to_z0 = solution.t_events[0][0]
        # Time of return to z=0 is double the time_to_z0
        return_time = 2 * time_to_z0
        bounce_frequency = (1 / return_time)/ 1e6  # Frequency calculation in MHz
    else:
        bounce_frequency = 0  # No bounce detected

    return bounce_frequency


from multiprocessing import Pool, Manager

def worker_function(args):
    r, angle = args
    frequency = simulate_electron_motion(r, angle)
    return r, angle, frequency

def collect_result(result):
    global results
    global progress_counter
    results.append(result)
    progress_counter.value += 1
    print(f"{progress_counter.value} / {total_tasks}")

if __name__ == "__main__":

    parameters = [(r, theta) for r in radial_distances for theta in emission_angles]
    total_tasks = len(parameters)

    num_processes = 10  # Change this to the number of processes you want to use

    # Use Manager to create a shared counter for progress tracking
    with Manager() as manager:
        results = manager.list()  # List to store results
        progress_counter = manager.Value('i', 0)  # Counter initialized to 0

        with Pool(num_processes) as pool:
            # Using apply_async instead of map
            for param in parameters:
                pool.apply_async(worker_function, args=(param,), callback=collect_result)

            pool.close()
            pool.join()

        # Convert the manager list back to a regular list
        results = list(results)
        
    # Filter results to find the frequency at 76 degrees and 0.024 meters radial distance
    desired_angle = 76
    desired_radial_distance = 0.024

    for r, angle, frequency in results:
        if angle == desired_angle and r == desired_radial_distance:
            print(f"Frequency at {desired_angle} degrees and {desired_radial_distance} meters: {frequency} MHz")


    # Convert results to arrays for plotting
    r_values, angle_values, frequencies = zip(*results)

    # Plotting 3D Plot
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(r_values, angle_values, frequencies, c=frequencies)
    ax.set_xlabel('Radial Distance (m)')
    ax.set_ylabel('Emission Angle (degrees)')
    ax.set_zlabel('Bounce Frequency (MHz)')
    plt.show()

# # Running the simulation with the test parameters
# r0 = 0
# # Kinetic energy in eV (18.6 keV converted to eV)
# kinetic_energy_eV = 18.6e3

# # Calculate Lorentz factor
# lorentz_factor = (kinetic_energy_eV * eV) / (m_e * c**2) + 1

# # Calculate velocity using the Lorentz factor
# speed = c * np.sqrt(1 - 1 / (lorentz_factor**2))
# # Convert angle to radians and calculate initial velocity components
# angle_rad = np.radians(76)
# # Assuming the electron is emitted in the xy-plane, we can set vx and vy based on the emission angle
# vz = speed * np.cos(angle_rad)
# vy = speed * np.sin(angle_rad)
# max_steps = 1e-12
# vx = 0  # Initial velocity in the x-direction is set to 0
# initial_conditions = [r0, 0, 0, vx, vy, vz]
# solution_test = solve_ivp(lorentz_dirac, [0, 1e-7], initial_conditions, method='RK45', max_step=max_steps)
# # # Plotting the trajectory
# # fig = plt.figure()
# # ax = fig.add_subplot(projection="3d")
# # ax.plot(solution_test.y[0], solution_test.y[1], solution_test.y[2])

# # ax.set_xlabel('X Coordinate')
# # ax.set_ylabel('Y Coordinate')
# # ax.set_zlabel('Z Coordinate')
# # ax.set_title('Electron Trajectory in the Magnetic Field')

# # plt.show()

# points_to_plot = 200  # Define how many points we want to examine

# # Extract the coordinates of the first few points
# x_coords = solution_test.y[0][:points_to_plot]
# y_coords = solution_test.y[1][:points_to_plot]
# z_coords = solution_test.y[2][:points_to_plot]

# # Now let's plot these points to visualize the trajectory segment
# plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot(x_coords, y_coords, z_coords, marker='')

# ax.set_xlabel('X Coordinate')
# ax.set_ylabel('Y Coordinate')
# ax.set_zlabel('Z Coordinate')
# ax.set_title('Segment of the Electron Trajectory in the Magnetic Field')

# plt.show()
