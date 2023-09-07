import numpy as np
import matplotlib.pyplot as plt
import time

# Define room layout and landmark positions
room_length = 10.0
room_width = 8.0
landmark1_pos = np.array([0.1, room_width/2])
landmark2_pos = np.array([room_length/2, room_width-0.1])
landmark3_pos = np.array([room_length-0.1, room_width/2])

# Define robot dimensions
robot_length = 0.3
robot_width = 0.2

# Define initial robot pose and covariance matrix
robot_pose = np.array([1.0, 1.0, 0.0])
robot_cov = np.eye(3) * 0.01

# Define sensor parameters
lidar_frequency = 1.0 # Hz
encoder_frequency = 100.0 # Hz
lidar_stddev_range = 0.01
lidar_stddev_bearing = 0.01
encoder_stddev_left = 0.001
encoder_stddev_right = 0.001

# Define motion command
v_l = 0.1 # left wheel velocity
v_r = 0.1 # right wheel velocity
L = 0.15 # distance between wheels

# Define motion model noise
motion_stddev_linear = 0.01
motion_stddev_angular = 0.01

# Define EKF SLAM algorithm parameters
Q = np.eye(2) * 0.01 # measurement noise covariance matrix
R_motion = np.eye(3) * 0.01 # motion model noise covariance matrix
R_measurement = np.eye(2) * 0.01 # measurement model noise covariance matrix

# Initialize landmark positions and covariance matrix
landmark1_est = np.array([1.0, 2.0])
landmark2_est = np.array([2.0, 7.0])
landmark3_est = np.array([9.0, 5.0])
landmark_cov = np.eye(2) * 0.01

# Create the plot and display room layout and landmarks
fig, ax = plt.subplots()
ax.set_xlim(0, room_length)
ax.set_ylim(0, room_width)
ax.set_aspect('equal')
ax.plot(landmark1_pos[0], landmark1_pos[1], 'ro')
ax.plot(landmark2_pos[0], landmark2_pos[1], 'ro')
ax.plot(landmark3_pos[0], landmark3_pos[1], 'ro')
ax.plot([0, 0, room_length, room_length, 0], [0, room_width, room_width, 0, 0], 'k')

# Run the simulation loop
while True:
    # Simulate motion and generate noisy encoder measurements
    v_l_noisy = np.random.normal(v_l, encoder_stddev_left)
    v_r_noisy = np.random.normal(v_r, encoder_stddev_right)
    u = np.array([0.5, 0.1, 0.01]) # [forward velocity, angular velocity, time interval]
    omega_noisy = (v_r_noisy - v_l_noisy) / L
    v_noisy = (v_r_noisy + v_l_noisy) / 2.0
    
    #Calculate motion model Jacobian matrix
    J_motion = np.array([[1.0, 0.0, -v_noisy*np.sin(robot_pose[2])],
    [0.0, 1.0, v_noisy*np.cos(robot_pose[2])],
    [0.0, 0.0, 1.0]])

    #Generate noisy motion measurements
    motion_noise = np.array([np.random.normal(0, motion_stddev_linear), np.random.normal(0, motion_stddev_angular), 0])
    u_noisy = u + motion_noise

    #Calculate predicted robot pose using motion model
    delta_t = u_noisy[2]
    if omega_noisy != 0:
        R = v_noisy / omega_noisy
        robot_pose = robot_pose + np.array([-R*np.sin(robot_pose[2]) + R*np.sin(robot_pose[2]+omega_noisy*delta_t),
        R*np.cos(robot_pose[2]) - R*np.cos(robot_pose[2]+omega_noisy*delta_t),
        omega_noisy*delta_t])
    else:
        robot_pose = robot_pose + np.array([v_noisy*np.cos(robot_pose[2])*delta_t, v_noisy*np.sin(robot_pose[2])*delta_t, omega_noisy*delta_t])

    #Update robot covariance matrix using motion model
    G = np.eye(3) + J_motion.T @ R_motion @ J_motion
    robot_cov = G @ robot_cov @ G.T + J_motion.T @ R_motion @ J_motion

    z_expected = np.zeros((3, 2))
    delta1 = np.array([0, 0])
    delta2 = np.array([0, 0])
    delta3 = np.array([0, 0])
    r1 = np.array([0, 0])
    r2 = np.array([0, 0])
    r3 = np.array([0, 0])
    b1 = np.array([0, 0])
    b2 = np.array([0, 0])
    b3 = np.array([0, 0])

    #Simulate lidar measurements and update landmark positions and covariance matrix
    if time.time() % (1/lidar_frequency) < 0.1:
    # Calculate expected landmark measurements
        delta1 = landmark1_est - robot_pose[:2]
        delta2 = landmark2_est - robot_pose[:2]
        delta3 = landmark3_est - robot_pose[:2]
        r1 = np.sqrt(delta1[0]**2 + delta1[1]**2)
        r2 = np.sqrt(delta2[0]**2 + delta2[1]**2)
        r3 = np.sqrt(delta3[0]**2 + delta3[1]**2)
        b1 = np.arctan2(delta1[1], delta1[0]) - robot_pose[2]
        b2 = np.arctan2(delta2[1], delta2[0]) - robot_pose[2]
        b3 = np.arctan2(delta3[1], delta3[0]) - robot_pose[2]
        z_expected = np.array([[r1, b1],[r2, b2],[r3, b3]])

    # Add noise to expected measurements
    z_expected[:, 0] += np.random.normal(0, lidar_stddev_range, size=3)
    z_expected[:, 1] += np.random.normal(0, lidar_stddev_bearing, size=3)

    # Calculate measurement Jacobian matrix
    H = np.zeros((6, 5))
    H[:3, :3] = np.eye(3)
    H[3:5, 3:5] = np.eye(2)
    eps = 1e-6
    if landmark1_est is not None:
        H[0, 3:5] = temp = np.array([-delta1[0]/(r1+eps)])
    if landmark2_est is not None:
        H[1, 3:5] = np.array([-delta2[0]/(r2+eps)])
    if landmark3_est is not None:
        H[2, 3:5] = np.array([-delta3[0]/(r3+eps)])

    # Calculate expected landmark bearings
    phi1 = np.arctan2(delta1[1], delta1[0]) - robot_pose[2]
    phi2 = np.arctan2(delta2[1], delta2[0]) - robot_pose[2]
    phi3 = np.arctan2(delta3[1], delta3[0]) - robot_pose[2]
    expected_measurements = np.array([[r1, phi1], [r2, phi2], [r3, phi3]])

    # Add noise to the expected measurements
    measurement_noise = np.random.normal(0, lidar_stddev_range, size=(3, 1))
    measurement_noise = np.tile(measurement_noise, (1, 2)) # Repeat along the second axis
    measurements = expected_measurements + measurement_noise

    landmark_est = np.array([landmark1_pos, landmark2_pos, landmark3_pos])
    landmark_cov = np.eye(2) * 1e-6

    # Update the landmark estimates and their covariance matrix using EKF SLAM
    for i, landmark in enumerate([landmark1_pos, landmark2_pos, landmark3_pos]):
        # Check if the landmark is in the field of view
        delta = landmark - robot_pose[:2]
        angle = robot_pose[2]
        range_est = np.linalg.norm(delta)
        bearing_est = np.arctan2(delta[1], delta[0]) - angle
        if abs(bearing_est) > np.pi / 2:
            continue # landmark is behind the robot
        # Predict measurement and measurement Jacobian
        z_expected[i,:] = np.array([range_est, bearing_est])
        H = np.array([[-range_est/delta[0], -range_est/delta[1]],
              [delta[1]/range_est**2, -delta[0]/range_est**2]])
        # Compute Kalman gain
        S = H @ landmark_cov @ H.T + Q
        K = landmark_cov @ H.T @ np.linalg.inv(S)
        # Update landmark estimate and covariance matrix
        delta_z = np.array([np.random.normal(0, lidar_stddev_range), np.random.normal(0, lidar_stddev_bearing)])
        z = z_expected[i,:] + delta_z
        delta = K @ (z - z_expected[i,:])
        landmark_est[i,:] = landmark_est[i,:] + delta[:2]
        landmark_cov = (np.eye(2) - K @ H) @ landmark_cov
    
    # Define measurement noise covariance matrix
    Q = np.eye(2) * 0.01 

    # Update robot pose and covariance matrix for each landmark
    for i in range(2):
        # Predict measurement
        delta = np.array([landmark_est[i][0] - robot_pose[0], landmark_est[i][1] - robot_pose[1]])
        q = delta.T @ delta
        z_expected[i] = np.array([np.sqrt(q), np.arctan2(delta[1], delta[0]) - robot_pose[2]])

        # Calculate measurement model Jacobian matrix
        H = np.array([[-np.sqrt(q)*delta[0], -np.sqrt(q)*delta[1], 0],
                    [delta[1], -delta[0], -q]])

        # Update robot pose and covariance matrix using EKF SLAM
        K = robot_cov @ H.T @ np.linalg.inv(H @ robot_cov @ H.T + Q)
        robot_pose += K @ (z[i] - z_expected[i])
        robot_cov = (np.eye(3) - K @ H) @ robot_cov

    # Plot the robot and landmark estimates
    ax.plot(robot_pose[0], robot_pose[1], 'bo')
    ax.plot(landmark1_est[0], landmark1_est[1], 'go')
    ax.plot(landmark2_est[0], landmark2_est[1], 'go')
    ax.plot(landmark3_est[0], landmark3_est[1], 'go')
    plt.draw()
    plt.pause(0.001)
