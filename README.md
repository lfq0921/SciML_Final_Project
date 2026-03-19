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



## Mathematical Development

### Governing Equation
The system is modeled as a **Driven Damped Nonlinear Pendulum**. The second-order ordinary differential equation (ODE) is:

$$\frac{d^2\theta}{dt^2} + b\frac{d\theta}{dt} + \frac{g}{L}\sin(\theta) = F\cos(\omega t)$$

Where:
- $\theta$: Angular displacement
- $b$: Damping coefficient ($0.5$)
- $g/L$: Normalized gravity/length ratio ($1.0$)
- $F, \omega$: Forcing amplitude ($1.2$) and frequency ($2/3$)

### PINN Formulation
The core idea is to treat the ODE as a constraint. We define the **Physics Residual** ($f$):

$$f := \frac{d^2\hat{\theta}}{dt^2} + b\frac{d\hat{\theta}}{dt} + \frac{g}{L}\sin(\hat{\theta}) - F\cos(\omega t)$$

The total loss function minimized by the neural network is:

$$\mathcal{L}_{total} = w_{data}\mathcal{L}_{data} + w_{phys}\mathcal{L}_{phys}$$

$$\mathcal{L}_{data} = \frac{1}{N}\sum |\hat{\theta}_i - \theta_i|^2, \quad \mathcal{L}_{phys} = \frac{1}{M}\sum |f_j|^2$$

---

## Implementation & Explanations

### Why PINNs?
In chaotic systems like this, standard Neural Networks (Black-box) fail to extrapolate when data is sparse or noisy. By embedding the ODE into the loss function, the model learns the **underlying physical laws**, allowing it to:
1. **Fill data gaps** where sensors might fail.
2. **Discover hidden parameters** (like the damping $b$).
3. Maintain **physical consistency** even in chaotic regimes.

### Model Architecture
- **Type:** Multi-Layer Perceptron (MLP)
- **Layers:** 3 hidden layers with 32 neurons each.
- **Activation:** `Tanh` (Crucial for computing smooth second-order derivatives).
- **Optimizer:** Adam ($\eta = 0.005$) for 5000 epochs.

