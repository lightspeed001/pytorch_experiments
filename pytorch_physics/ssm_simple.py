import torch
import torch.nn as nn

class PhysicsInspiredSSM(nn.Module):
    """SSM framed as a Hamiltonian system."""
    def __init__(self, state_dim, input_dim, output_dim):
        super().__init__()
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.randn(state_dim, state_dim))  # State transition (Hamiltonian flow)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim))  # Control input matrix
        self.C = nn.Parameter(torch.randn(output_dim, state_dim)) # Observation matrix
        self.Q = nn.Parameter(torch.eye(state_dim))  # "Mass matrix" (energy metric)
        self.R = nn.Parameter(torch.eye(output_dim)) # Observation noise covariance

    def forward(self, x, u):
        """State transition: x_t = A x_{t-1} + B u_t + w_t (w_t ~ N(0, Q))"""
        # Predict next state (Hamiltonian flow)
        x_next = self.A @ x + self.B @ u + torch.randn_like(x) @ torch.linalg.cholesky(self.Q)

        # Observation: y_t = C x_t + v_t (v_t ~ N(0, R))
        y = self.C @ x_next + torch.randn_like(x_next) @ torch.linalg.cholesky(self.R)

        return x_next, y

# Example usage
if __name__ == "__main__":
    ssm = PhysicsInspiredSSM(state_dim=3, input_dim=2, output_dim=1)
    x = torch.randn(3)  # Initial state
    u = torch.randn(2)  # Input
    x_next, y = ssm(x, u)
    print("Next state:", x_next)
    print("Observation:", y)
