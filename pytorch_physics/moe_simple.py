import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Expert as a particle in parameter space."""

    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(input_dim, output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x):
        """Expert's response to input perturbation."""
        return self.net(x)


class PhysicsInspiredMoE(nn.Module):
    """MoE model framed as a Hamiltonian system.

    The gate produces an "energy" score per expert. Low energy = favoured
    state, so we route each token to the top-k lowest-energy experts and
    weight their outputs by a Boltzmann-style softmax over those energies.

    Includes the two auxiliary pieces real MoEs need:
      - a load-balancing loss (prevents gate collapse onto few experts)
      - an importance/load tracker for monitoring
    """

    def __init__(self, num_experts, input_dim, output_dim, top_k=2,
                 hidden_dim=None, load_balance_weight=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_dim = output_dim
        self.load_balance_weight = load_balance_weight

        self.experts = nn.ModuleList(
            [Expert(input_dim, output_dim, hidden_dim) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(input_dim, num_experts)  # energy per expert

        # Tracks average expert usage (for monitoring, not training state)
        self.register_buffer("expert_load", torch.zeros(num_experts))
        self.register_buffer("num_tokens_processed", torch.zeros(1))

    def forward(self, x):
        """
        x: (B, input_dim) or (B, T, input_dim)
        returns:
            output: (B, output_dim) or (B, T, output_dim)
            aux_loss: scalar load-balancing loss
        """
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])            # (N, input_dim), N = B or B*T

        # --- Hamiltonian (energy) for each expert ---
        energy = self.gate(flat)                        # (N, E)

        # --- Select top-k lowest-energy experts ---
        top_k_vals, top_k_idx = torch.topk(-energy, k=self.top_k, dim=-1)  # (N, k)
        # Boltzmann weights: softmax over the selected experts only
        top_k_w = F.softmax(top_k_vals, dim=-1)         # (N, k), sums to 1 over k

        # --- Dispatch: run every expert on its assigned tokens only ---
        output = torch.zeros(
            flat.size(0), self.output_dim,
            device=x.device, dtype=x.dtype,
        )                                               # (N, D)
        for i, expert in enumerate(self.experts):
            token_idx, slot = torch.where(top_k_idx == i)   # tokens routed to expert i
            if token_idx.numel() == 0:
                continue
            w = top_k_w[token_idx, slot].unsqueeze(-1)     # (n_i, 1)
            output.index_add_(0, token_idx, w * expert(flat[token_idx]))

        # --- Load-balancing loss (Switch-Transformer style) ---
        # load: fraction of tokens routed to each expert (no grad)
        # importance: mean gate probability per expert (grad flows through the gate)
        with torch.no_grad():
            counts = torch.bincount(
                top_k_idx.reshape(-1), minlength=self.num_experts
            ).to(x.dtype)
            load = counts / counts.sum()
        importance = F.softmax(energy, dim=-1).mean(dim=0)  # (E,), keeps grad
        aux_loss = self.num_experts * torch.sum(importance * load)

        # --- Update monitoring stats ---
        with torch.no_grad():
            self.expert_load += counts
            self.num_tokens_processed += flat.size(0)

        output = output.reshape(*orig_shape[:-1], self.output_dim)
        return output, aux_loss

    @torch.no_grad()
    def load_stats(self):
        """Average routing distribution since last reset."""
        total = self.num_tokens_processed.clamp(min=1.0)
        return (self.expert_load / (total * self.top_k)).tolist()

    @torch.no_grad()
    def reset_load_stats(self):
        self.expert_load.zero_()
        self.num_tokens_processed.zero_()


if __name__ == "__main__":
    torch.manual_seed(0)
    model = PhysicsInspiredMoE(num_experts=4, input_dim=10, output_dim=5, top_k=2)

    x = torch.randn(3, 10)          # batch of 3 tokens
    y, aux = model(x)
    print("output:", y.shape)       # torch.Size([3, 5])
    print("aux_loss:", aux.item())
    print("load:", model.load_stats())

    # Works with sequence data too: (batch, time, features)
    xs = torch.randn(2, 7, 10)
    ys, _ = model(xs)
    print("seq output:", ys.shape)  # torch.Size([2, 7, 5])

    # Quick sanity: output equals the weighted sum of the selected experts
    energy = model.gate(x)
    _, idx = torch.topk(-energy, 2, dim=-1)
    manual = torch.zeros_like(y)
    for b in range(x.size(0)):
        vals, sel = torch.topk(-energy[b], 2)
        w = F.softmax(vals, dim=-1)
        for j, e in enumerate(sel):
            manual[b] += w[j] * model.experts[e](x[b:b + 1])[0]
    print("matches manual routing:", torch.allclose(y, manual, atol=1e-6))
