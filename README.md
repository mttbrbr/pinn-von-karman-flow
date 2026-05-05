# NeuralKarman

**Physics-Informed Neural Networks (PINN) for 2D Navier-Stokes simulation of the Von Kármán Vortex Street.**

### About
This repository implements a meshless CFD solver using **DeepXDE** and **PyTorch** (with ROCm support). It solves the incompressible Navier-Stokes equations to simulate fluid flow around a cylinder at $Re=200$.

### Core Concept
Unlike traditional CFD (e.g., OpenFOAM), this approach utilizes **PINNs** to approximate velocity ($u, v$) and pressure ($p$) fields. By embedding the governing partial differential equations (PDEs) directly into the neural network's loss function, the model learns the physics of the flow without the need for a computational mesh or complex overset connectivity.

### Key Features
*   **Meshless Framework:** No grid generation required.
*   **Physics-Driven:** Loss function based on Navier-Stokes residuals, Boundary Conditions (BCs), and Initial Conditions (ICs).
*   **Hybrid Training:** Uses Adam for global exploration followed by L-BFGS for fine-tuning.
*   **Hardware Accelerated:** Optimized for execution on AMD GPUs via ROCm.

### Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python train.py`
3. Visualize: Import `results_karman.csv` into **ParaView**.
