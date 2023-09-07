import numpy as np
import matplotlib.pyplot as plt

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
R_motion = np.eye(2) * 0.01 # motion model noise covariance matrix
R_measurement = np.eye(2) * 0.01 # measurement model noise covariance matrix

# Initialize landmark positions and covariance matrix
landmark1_est = np.array([1.0, 2.0])
landmark2_est = np.array([2.0, 7.0])
landmark3_est = np.array([9.0, 5.0])
landmark_cov = np.eye(2) * 0.01

for i in range(100):
    # Simulate motion and generate noisy encoder measurements
    v_l_noisy = np.random.normal(v_l, encoder_stddev_left)
    v_r_noisy = np.random.normal(v_r, encoder_stddev_right)
    u = np.array([0.5, 0.1, 0.01]) # [forward velocity, angular velocity, time interval]
    omega_noisy = (v_r_noisy - v_l_noisy) / L
    v_noisy = (v_r_noisy + v_l_noisy) / 2
    motion_noise_linear = np.random.normal(0, motion_stddev_linear, size=2)
    motion_noise_angular = np.random.normal(0, motion_stddev_angular)
    robot_pose[0] += v_noisy * np.cos(robot_pose[2]) * 1/lidar_frequency + motion_noise_linear[0]
    robot_pose[1] += v_noisy * np.sin(robot_pose[2]) * 1/lidar_frequency + motion_noise_linear[1]
    robot_pose[2] += omega_noisy * 1/lidar_frequency + motion_noise_angular
    F_motion = np.array([[1, 0, -v_noisy*np.sin(robot_pose[2])/lidar_frequency],
    [0, 1, v_noisy*np.cos(robot_pose[2])/lidar_frequency],
    [0, 0, 1]])
    G_motion = np.array([[np.cos(robot_pose[2])/lidar_frequency, -v_noisy*np.sin(robot_pose[2])/(lidar_frequency**2)],
    [np.sin(robot_pose[2])/lidar_frequency, v_noisy*np.cos(robot_pose[2])/(lidar_frequency**2)],
    [0, 1/lidar_frequency]])
    robot_cov = np.dot(F_motion, np.dot(robot_cov, F_motion.T)) 
    temp = np.dot(G_motion, R_motion)
    robot_cov += np.dot(G_motion, temp.T)


# Generate LiDAR measurements
measurement_noise = np.random.normal(0, lidar_stddev_range, size=3) # assuming 3 landmarks
measurement1 = np.sqrt((landmark1_est[0]-robot_pose[0])**2 + (landmark1_est[1]-robot_pose[1])**2)
measurement2 = np.sqrt((landmark2_est[0]-robot_pose[0])**2 + (landmark2_est[1]-robot_pose[1])**2)
measurement3 = np.sqrt((landmark3_est[0]-robot_pose[0])**2 + (landmark3_est[1]-robot_pose[1])**2)
measurement1_bearing = np.arctan2(landmark1_est[1]-robot_pose[1], landmark1_est[0]-robot_pose[0]) - robot_pose[2]
measurement2_bearing = np.arctan2(landmark2_est[1]-robot_pose[1], landmark2_est[0]-robot_pose[0]) - robot_pose[2]
measurement3_bearing = np.arctan2(landmark3_est[1]-robot_pose[1], landmark3_est[0]-robot_pose[0]) - robot_pose[2]
measurement1_bearing = (measurement1_bearing + np.pi) % (2*np.pi) - np.pi # wrap to [-pi, pi]
measurement2_bearing = (measurement2_bearing + np.pi) % (2*np.pi) - np.pi # wrap to [-pi, pi]
measurement3_bearing = (measurement3_bearing + np.pi) % (2*np.pi) - np.pi # wrap to [-pi, pi]
measurement1 += measurement_noise[0] * lidar_stddev_range
measurement2 += measurement_noise[1] * lidar_stddev_range
measurement3 += measurement_noise[2] * lidar_stddev_range
measurement1_bearing += measurement_noise[0] * lidar_stddev_bearing
measurement2_bearing += measurement_noise[1] * lidar_stddev_bearing
measurement3_bearing += measurement_noise[2] * lidar_stddev_bearing
measurements = [(1, measurement1), (2, measurement2), (3, measurement3)]
measurement_bearings = [measurement1_bearing, measurement2_bearing, measurement3_bearing]


# Associate measurements with landmarks based on spatial distance criteria
for i, measurement in enumerate(measurements):
    min_distance = np.inf
    associated_landmark = None
    for landmark in [landmark1_est, landmark2_est, landmark3_est]:
        distance = np.sqrt((landmark[0]-robot_pose[0])**2 + (landmark[1]-robot_pose[1])**2)
        if distance < min_distance:
            min_distance = distance
            associated_landmark = landmark
    # Update landmark estimate using EKF
    H = np.array([[-(associated_landmark[0]-robot_pose[0])/min_distance, -(associated_landmark[1]-robot_pose[1])/min_distance],
              [(associated_landmark[1]-robot_pose[1])/min_distance**2, -(associated_landmark[0]-robot_pose[0])/min_distance**2]])
    S = np.dot(H, np.dot(landmark_cov, H.T)) + Q
    K = np.dot(np.dot(landmark_cov, H), np.linalg.inv(S))
    innovation = np.array([measurement-min_distance, measurement_bearings[i]])
    landmark_est = associated_landmark + np.dot(K, innovation)
    landmark_cov = np.dot((np.eye(2)-np.dot(K, H)), landmark_cov)


# Update robot pose estimate using EKF
F_motion = np.array([[1, 0, -v_noisy*np.sin(robot_pose[2])/lidar_frequency],
                     [0, 1, v_noisy*np.cos(robot_pose[2])/lidar_frequency],
                     [0, 0, 1]])
G_motion = np.array([[(np.cos(robot_pose[2]) + np.cos(robot_pose[2]+omega_noisy/lidar_frequency))/2, -(np.sin(robot_pose[2]) + np.sin(robot_pose[2]+omega_noisy/lidar_frequency))/2],
                     [(np.sin(robot_pose[2]) + np.sin(robot_pose[2]+omega_noisy/lidar_frequency))/2, (np.cos(robot_pose[2]) + np.cos(robot_pose[2]+omega_noisy/lidar_frequency))/2],
                     [1/L, -1/L]])
R_motion_noise = np.array([[motion_stddev_linear**2, 0, 0],
                           [0, motion_stddev_linear**2, 0],
                           [0, 0, motion_stddev_angular**2]])
robot_pose_est = np.dot(F_motion, robot_pose) + np.dot(G_motion, np.array([v_noisy, omega_noisy])) + np.random.multivariate_normal([0,0,0], R_motion_noise)
robot_cov = np.dot(np.dot(F_motion, robot_cov), F_motion.T) + np.dot(np.dot(G_motion, R_motion), G_motion.T)

# Display results
plt.clf()
plt.xlim((0, room_length))
plt.ylim((0, room_width))
plt.gca().set_aspect('equal', adjustable='box')
plt.plot([0, room_length, room_length, 0, 0], [0, 0, room_width, room_width, 0], 'k-', linewidth=2)
plt.plot(landmark1_pos[0], landmark1_pos[1], 'rs', markersize=10)
plt.plot(landmark2_pos[0], landmark2_pos[1], 'bs', markersize=10)
plt.plot(landmark3_pos[0], landmark3_pos[1], 'gs', markersize=10)
plt.plot(landmark1_est[0], landmark1_est[1], 'ro', markersize=5)
plt.plot(landmark2_est[0], landmark2_est[1], 'bo', markersize=5)
plt.plot(landmark3_est[0], landmark3_est[1], 'go', markersize=5)

for i in range(len(measurements)):
    landmark_id = i + 1
    landmark_pos = measurements[i]
    if landmark_id == 1:
        plt.plot([robot_pose[0], landmark_pos[0]], [robot_pose[1], landmark_pos[1]], 'r--')
    elif landmark_id == 2:
        plt.plot([robot_pose[0], landmark_pos[0]], [robot_pose[1], landmark_pos[1]], 'b--')
    elif landmark_id == 3:
        plt.plot([robot_pose[0], landmark_pos[0]], [robot_pose[1], landmark_pos[1]], 'g--')
        
plt.plot(robot_pose[0], robot_pose[1], 'bo', markersize=5)
plt.show()


