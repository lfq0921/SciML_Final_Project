import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from dataset import get_pendulum_data, get_euler_data
from models import PINN, InversePINN, physics_loss, inverse_physics_loss

def main():
    os.makedirs('images', exist_ok=True)
    
    # 1. Load Data
    print("Loading Ground Truth Data...")
    t_data, theta_data, _ = get_pendulum_data()
    t_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
    theta_tensor = torch.tensor(theta_data, dtype=torch.float32).view(-1, 1)

    # 2. Setup Data Gap (Sensor Failure Scenario)
    valid_idx = torch.where((t_tensor < 15) | (t_tensor > 35))[0]
    torch.manual_seed(42)
    idx_gap = torch.randperm(len(valid_idx))[:100]
    train_idx_gap = valid_idx[idx_gap]
    
    t_train_gap = t_tensor[train_idx_gap].detach()
    theta_train_gap = theta_tensor[train_idx_gap].detach()

    # 3. Train Pure NN (Ablation Baseline)
    print("Training Pure Neural Network (Data Gap)...")
    model_pure = PINN()
    optimizer_pure = optim.Adam(model_pure.parameters(), lr=0.005)
    for _ in range(5000):
        optimizer_pure.zero_grad()
        loss = torch.mean((model_pure(t_train_gap) - theta_train_gap)**2)
        loss.backward()
        optimizer_pure.step()

    # 4. Train PINN (Data Gap)
    print("Training Physics-Informed Neural Network (Data Gap)...")
    model_pinn = PINN()
    optimizer_pinn = optim.Adam(model_pinn.parameters(), lr=0.005)
    for _ in range(5000):
        optimizer_pinn.zero_grad()
        loss_data = torch.mean((model_pinn(t_train_gap) - theta_train_gap)**2)
        loss_ode = physics_loss(model_pinn, t_tensor)
        (loss_data + loss_ode).backward()
        optimizer_pinn.step()

    # Plot Extrapolation Comparison
    model_pure.eval()
    model_pinn.eval()
    with torch.no_grad():
        theta_pred_pure = model_pure(t_tensor).numpy().flatten()
        theta_pred_pinn = model_pinn(t_tensor).numpy().flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(t_data, theta_data, 'b-', label='Ground Truth', alpha=0.5)
    plt.plot(t_data, theta_pred_pure, 'orange', linestyle='--', label='Pure NN (Failed)')
    plt.plot(t_data, theta_pred_pinn, 'r-', label='PINN (Succeeded)')
    plt.scatter(t_train_gap.numpy(), theta_train_gap.numpy(), color='green', s=20, label='Training Data', zorder=5)
    plt.axvspan(15, 35, color='red', alpha=0.1, label='Sensor Failure Zone')
    plt.legend()
    plt.title('Extrapolation in Data Gap: Pure NN vs PINN')
    plt.savefig('images/gap_comparison.png')
    print("Saved gap_comparison.png")

    # 5. Inverse Problem (Parameter Discovery)
    print("Training Inverse PINN for Parameter Discovery...")
    # Use evenly distributed sparse data for inverse problem
    N_train = int(len(t_data) * 0.20)
    train_idx = torch.randperm(len(t_data))[:N_train]
    
    t_train = t_tensor[train_idx].detach()
    theta_train = theta_tensor[train_idx].detach()
    
    t_tensor_clean = t_tensor.detach()

    model_inv = InversePINN()
    optimizer_inv = optim.Adam(model_inv.parameters(), lr=0.005)
    b_history = []
    
    for _ in range(5000):
        optimizer_inv.zero_grad()
        loss_data = torch.mean((model_inv(t_train) - theta_train)**2)
        loss_ode = inverse_physics_loss(model_inv, t_tensor_clean)
        (loss_data + loss_ode).backward()
        optimizer_inv.step()
        b_history.append(model_inv.b_pred.item())

    plt.figure(figsize=(8, 5))
    plt.plot(b_history, 'g-', label='Predicted b')
    plt.axhline(y=0.5, color='b', linestyle='--', label='True b (0.5)')
    plt.legend()
    plt.title('Parameter Discovery Convergence')
    plt.savefig('images/parameter_discovery.png')
    print("Saved parameter_discovery.png")

if __name__ == "__main__":
    main()