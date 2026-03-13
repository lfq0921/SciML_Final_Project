import torch
import torch.nn as nn

class PINN(nn.Module):
    """Standard Physics-Informed Neural Network."""
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        return self.net(t)

class InversePINN(nn.Module):
    """PINN with a learnable physical parameter for inverse problems."""
    def __init__(self):
        super(InversePINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
        # Learnable damping coefficient initialized to 0.0
        self.b_pred = nn.Parameter(torch.tensor([0.0]))

    def forward(self, t):
        return self.net(t)

def physics_loss(model, t, g_L=1.0, b=0.5, F=1.2, omega=2/3):
    """Calculate the ODE residual for the standard PINN."""
    t.requires_grad = True
    theta = model(t)
    
    dtheta_dt = torch.autograd.grad(theta, t, grad_outputs=torch.ones_like(theta), create_graph=True)[0]
    d2theta_dt2 = torch.autograd.grad(dtheta_dt, t, grad_outputs=torch.ones_like(dtheta_dt), create_graph=True)[0]
    
    residual = d2theta_dt2 + b * dtheta_dt + g_L * torch.sin(theta) - F * torch.cos(omega * t)
    return torch.mean(residual**2)

def inverse_physics_loss(model, t, g_L=1.0, F=1.2, omega=2/3):
    """Calculate the ODE residual with the learnable parameter b_pred."""
    t.requires_grad = True
    theta = model(t)
    
    dtheta_dt = torch.autograd.grad(theta, t, grad_outputs=torch.ones_like(theta), create_graph=True)[0]
    d2theta_dt2 = torch.autograd.grad(dtheta_dt, t, grad_outputs=torch.ones_like(dtheta_dt), create_graph=True)[0]
    
    residual = d2theta_dt2 + model.b_pred * dtheta_dt + g_L * torch.sin(theta) - F * torch.cos(omega * t)
    return torch.mean(residual**2)