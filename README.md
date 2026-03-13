# Physics-Informed Neural Networks for Chaotic Systems
**Northwestern University - SciML Final Project**

This repository contains the code, data, and final report for solving and analyzing the driven damped nonlinear pendulum using Physics-Informed Neural Networks (PINNs). 

## Repository Structure
* `dataset.py`: Handles data generation using RK45 and Forward Euler solvers.
* `models.py`: Defines the neural network architectures and physics-informed loss functions.
* `main.py`: The main execution script that trains the models and generates comparison plots.
* `Final_Report.pdf`: The 2-page academic report detailing the mathematical formulation and discussions.
* `images/`: Directory containing all generated plots.

## Requirements & How to Run
Ensure you have the following libraries installed:
`pip install torch numpy scipy matplotlib`

To run the full suite of experiments (baseline ablation, gap extrapolation, and parameter discovery), simply execute:
`python main.py`

## Key Results
* **Data Gap Extrapolation:** The PINN successfully maintains qualitative physical dynamics inside a 20-second data blind spot, outperforming pure data-driven models.
* **Parameter Discovery:** The network inverted the physical system, discovering the hidden damping coefficient $b \approx 0.5024$ (True: 0.5) from observational data.