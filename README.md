# Robot Localization and Landmark Estimation
This repository contains a Python program implemented to simulate robot motion and landmark localization using an Extended Kalman Filter (EKF). Below, you'll find an overview of the project and its key features.

# Overview
1. Simulation of Landmarks and Robot Motion
In this project, I have developed a comprehensive simulation environment for robot motion within a rectangular room. The simulation includes the following features:

Graphics Interface: The program provides a graphical interface that visually represents the room layout, the real-time location of the robot, and the positions of predefined landmarks.

Sensor Noise: To mimic real-world conditions, Gaussian noise has been added to both sensor observations and inputs from wheel encoders.

Sensor Frequencies: The LiDAR sensor operates at 1Hz, while the wheel encoders provide data at 100Hz.

Robot Imperfections: To account for the challenges faced by real robots, I introduced factors such as incorrect robot dimensions, sliding motion during movement, and variations in the floor's planarity.

The selection of covariance values to represent measurement uncertainties is thoroughly justified within the code.

2. Extended Kalman Filter (EKF) Implementation
The core of this project is the implementation of an Extended Kalman Filter (EKF) for robot localization and the estimation of landmark positions over time. The notebook includes detailed explanations of the EKF algorithm and its application in this context.

The notebook also features plots illustrating the 2D positions and uncertainty (covariance) associated with the estimated robot location and landmark positions. The initialization of state and covariance values aligns with the explanations provided in the previous section.

3. Convergence and Real-Time Visualization
I have carefully selected the number of iterations (time steps) to ensure that the state estimation converges effectively. Throughout the simulation, you can observe real-time updates in the graphical interface, which displays the estimated robot location and landmark positions as the EKF refines its estimations. Ground truth values are also plotted for performance evaluation.

# Usage
To run the simulation and explore the EKF implementation, follow these steps:

Clone this repository to your local machine.
Install all the necessary libraries.
Feel free to contribute to this project by submitting pull requests or opening issues if you encounter any problems or have suggestions for improvement.

Thank you for exploring this robot localization and landmark estimation project!
