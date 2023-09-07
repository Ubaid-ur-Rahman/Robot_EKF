# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Define constants
room_width = 10 # meters
room_height = 8 # meters
robot_width = 0.3 # meters
robot_length = 0.5 # meters
encoder_noise = 0.01 # meters per second
lidar_noise = 0.02 # meters
lidar_frequency = 1 # Hz
encoder_frequency = 100 # Hz
robot_slide_noise = 0.05 # meters per second^2
floor_unevenness_noise = 0.02 # meters

# Define initial variables
robot_pose = np.array([1, 1, 0]) # [x, y, theta]
landmark_positions = np.array([[2, 4], [5, 6], [8, 3]]) # [x, y] for each landmark
simulation_duration = 10 # seconds
time_steps = int(simulation_duration * encoder_frequency)

# Create plot
fig, ax = plt.subplots()

# Draw room layout
ax.plot([0, room_width, room_width, 0, 0], [0, 0, room_height, room_height, 0], 'k-', linewidth=2)

# Plot landmark positions
ax.plot(landmark_positions[:, 0], landmark_positions[:, 1], 'bs', markersize=10)

# Initialize arrays to store simulation data
robot_trajectory = np.zeros((time_steps, 2))
lidar_measurements = np.zeros((time_steps, 3, 2)) # 3 landmarks with (r, alpha) each
encoder_measurements = np.zeros((time_steps, 2))

# Loop for simulation duration
for t in range(time_steps):
    
    # Simulate encoder measurements
    v_l = 1 + np.random.normal(0, encoder_noise)
    v_r = 1 + np.random.normal(0, encoder_noise)
    encoder_measurements[t] = np.array([v_l, v_r])
    
    # Simulate robot motion
    v = (v_l + v_r) / 2
    omega = (v_r - v_l) / robot_width
    robot_pose[0] += v * np.cos(robot_pose[2]) + np.random.normal(0, robot_slide_noise)
    robot_pose[1] += v * np.sin(robot_pose[2]) + np.random.normal(0, robot_slide_noise)
    robot_pose[2] += omega / encoder_frequency
    robot_trajectory[t] = robot_pose[:2]
    
    # Simulate lidar measurements
    if t % (int(encoder_frequency / lidar_frequency)) == 0:
        for i, landmark_position in enumerate(landmark_positions):
            r = np.linalg.norm(robot_pose[:2] - landmark_position) + np.random.normal(0, lidar_noise)
            alpha = np.arctan2(landmark_position[1] - robot_pose[1], landmark_position[0] - robot_pose[0]) - robot_pose[2] + np.random.normal(0, lidar_noise)
            lidar_measurements[t, i] = np.array([r, alpha])
    
    # Update plot
    ax.clear()
    ax.plot([0, room_width, room_width, 0, 0], [0, 0, room_height, room_height, 0], 'k-', linewidth=2)
    ax.plot(landmark_positions[:, 0], landmark_positions[:, 1], 'bs', markersize=10)
    ax.plot(robot_trajectory[:t+1, 0], robot_trajectory[:t+1, 1], 'r-', linewidth=2)
    for i in range(3):
        if lidar_measurements[t, i, 0] != 0:
            x = robot_pose[0] + lidar_measurements[t, i, 0] * np.cos(lidar_measurements[t, i, 1] + robot_pose[2])
            y = robot_pose[1] + lidar_measurements[t, i, 0] * np.sin(lidar_measurements[t, i, 1] + robot_pose[2])
            ax.plot(x, y, 'ro')
    plt.xlim([0, room_width])
    plt.ylim([0, room_height])
    plt.title('Simulation time: {} s'.format(t / encoder_frequency))
    plt.draw()
    plt.pause(0.1)

