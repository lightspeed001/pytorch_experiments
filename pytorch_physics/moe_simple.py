import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """Expert as a particle in parameter space."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """Expert's response to input perturbation."""
        return self.net(x)

class PhysicsInspiredMoE(nn.Module):
    """MoE model framed as a Hamiltonian system."""
    def __init__(self, num_experts, input_dim, output_dim, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)  # Hamiltonian-like gating network

    def forward(self, x):
        # Compute Hamiltonian (energy) for each expert
        hamiltonian = self.gate(x)  # Shape: (batch_size, num_experts)

        # Gating probabilities via softmax (Boltzmann distribution)
        gate_probs = F.softmax(hamiltonian, dim=-1)  # Shape: (batch_size, num_experts)

        # Select top-k experts (lowest Hamiltonian)
        top_k_probs, top_k_indices = torch.topk(gate_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Renormalize

        # Compute output as superposition of expert responses
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # Mask for expert i
            mask = (top_k_indices == i).float()
            # Weighted sum of expert's output
            output += mask.unsqueeze(-1) * top_k_probs.unsqueeze(-1) * expert(x)

        return output

# Example usage
if __name__ == "__main__":
    model = PhysicsInspiredMoE(num_experts=4, input_dim=10, output_dim=5, top_k=2)
    x = torch.randn(3, 10)  # Batch of 3 inputs, each of dimension 10
    y = model(x)
    print(y.shape)  # Should be (3, 5)
