import torch
import triton
import triton.language as tl
from torch import autograd

@triton.jit
def pinn_heat_equation_forward(
    # Input coordinates (x, t)
    x_ptr, t_ptr,
    # Neural network weights (simplified for demo)
    weights_ptr,
    # Output predictions
    u_ptr,
    # Dimensions
    N,  # Number of points
    # Physics parameters
    alpha: tl.constexpr,  # Thermal diffusivity
    # Neural network hyperparameters
    HIDDEN_SIZE: tl.constexpr = 64,
    NUM_LAYERS: tl.constexpr = 3,
):
    """
    Physics-Informed Neural Network (PINN) for the 1D heat equation.
    Solves: du/dt = alpha * d²u/dx².

    Physics:
    - The heat equation is enforced as a loss term.
    - The neural network approximates u(x,t).
    - Derivatives are computed via automatic differentiation (AD).

    Triton:
    - Parallelizes computation across (x,t) points.
    - Uses fused multiply-add (FMA) for efficiency.
    """
    # -----------------------------------------------------------
    # Physics: Load input coordinates (x, t)
    # -----------------------------------------------------------
    pid = tl.program_id(axis=0)
    x = tl.load(x_ptr + pid)
    t = tl.load(t_ptr + pid)

    # -----------------------------------------------------------
    # Physics: Neural Network Forward Pass (Hamiltonian Flow)
    # -----------------------------------------------------------
    # Simplified MLP (for demo; in practice, use a proper MLP kernel)
    # Here, we use a single hidden layer for simplicity.
    h = tl.zeros([HIDDEN_SIZE], dtype=tl.float32)
    for i in range(HIDDEN_SIZE):
        # First layer: h = ReLU(W1 @ [x, t] + b1)
        w1 = tl.load(weights_ptr + i * 2 + 0)  # Weight for x
        w2 = tl.load(weights_ptr + i * 2 + 1)  # Weight for t
        h[i] = tl.math.relu(w1 * x + w2 * t)

    # Output layer: u = W2 @ h + b2
    u = tl.zeros([1], dtype=tl.float32)
    for i in range(HIDDEN_SIZE):
        w2 = tl.load(weights_ptr + HIDDEN_SIZE * 2 + i)
        u[0] += w2 * h[i]
    u = u + tl.load(weights_ptr + HIDDEN_SIZE * 3)  # Bias term

    # Store the prediction
    tl.store(u_ptr + pid, u)

@triton.jit
def pinn_heat_equation_backward(
    # Input coordinates (x, t)
    x_ptr, t_ptr,
    # Neural network weights
    weights_ptr,
    # Gradients w.r.t. weights
    grad_weights_ptr,
    # Physics loss gradient
    grad_physics_ptr,
    # Dimensions
    N,
    # Physics parameters
    alpha: tl.constexpr,
    # Neural network hyperparameters
    HIDDEN_SIZE: tl.constexpr = 64,
    NUM_LAYERS: tl.constexpr = 3,
):
    """
    Backward pass for the PINN heat equation solver.
    Computes gradients of the physics loss w.r.t. weights.
    """
    pid = tl.program_id(axis=0)
    x = tl.load(x_ptr + pid)
    t = tl.load(t_ptr + pid)

    # -----------------------------------------------------------
    # Physics: Compute derivatives (Automatic Differentiation)
    # -----------------------------------------------------------
    # Compute du/dt and d²u/dx² using finite differences (for demo).
    # In practice, use autograd for exact derivatives.
    h = 1e-3  # Small step for finite differences

    # du/dt ≈ (u(x, t+h) - u(x, t)) / h
    u_t = (pinn_heat_equation_forward(x, t + h, weights_ptr, None, N, alpha, HIDDEN_SIZE, NUM_LAYERS) -
           pinn_heat_equation_forward(x, t, weights_ptr, None, N, alpha, HIDDEN_SIZE, NUM_LAYERS)) / h

    # d²u/dx² ≈ (u(x+h, t) - 2u(x, t) + u(x-h, t)) / h²
    u_xx = (pinn_heat_equation_forward(x + h, t, weights_ptr, None, N, alpha, HIDDEN_SIZE, NUM_LAYERS) -
            2 * pinn_heat_equation_forward(x, t, weights_ptr, None, N, alpha, HIDDEN_SIZE, NUM_LAYERS) +
            pinn_heat_equation_forward(x - h, t, weights_ptr, None, N, alpha, HIDDEN_SIZE, NUM_LAYERS)) / (h ** 2)

    # Physics loss: (du/dt - alpha * d²u/dx²) ** 2
    physics_loss = (u_t - alpha * u_xx) ** 2

    # -----------------------------------------------------------
    # Physics: Compute gradients w.r.t. weights (Energy Minimization)
    # -----------------------------------------------------------
    # Simplified gradient computation (in practice, use autograd).
    # Here, we approximate the gradient of the physics loss w.r.t. weights.
    for i in range(HIDDEN_SIZE):
        # Gradient w.r.t. first layer weights (w1, w2)
        w1 = tl.load(weights_ptr + i * 2 + 0)
        w2 = tl.load(weights_ptr + i * 2 + 1)
        grad_w1 = 2 * physics_loss * (u_t - alpha * u_xx) * (x * tl.math.relu_derivative(w1 * x + w2 * t))
        grad_w2 = 2 * physics_loss * (u_t - alpha * u_xx) * (t * tl.math.relu_derivative(w1 * x + w2 * t))
        tl.store(grad_weights_ptr + i * 2 + 0, grad_w1)
        tl.store(grad_weights_ptr + i * 2 + 1, grad_w2)

        # Gradient w.r.t. second layer weights (w2)
        w2_out = tl.load(weights_ptr + HIDDEN_SIZE * 2 + i)
        grad_w2_out = 2 * physics_loss * (u_t - alpha * u_xx) * h[i]
        tl.store(grad_weights_ptr + HIDDEN_SIZE * 2 + i, grad_w2_out)

    # Gradient w.r.t. bias term
    grad_bias = 2 * physics_loss * (u_t - alpha * u_xx)
    tl.store(grad_weights_ptr + HIDDEN_SIZE * 3, grad_bias)

    # Store gradient of physics loss (for debugging)
    tl.store(grad_physics_ptr + pid, physics_loss)

# -----------------------------------------------------------
# Python Wrapper (for testing)
# -----------------------------------------------------------
def pinn_heat_equation_solve(
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float = 1.0,
    hidden_size: int = 64,
    num_layers: int = 3,
    lr: float = 1e-3,
    num_steps: int = 1000,
):
    """
    Solve the 1D heat equation using a Physics-Informed Neural Network (PINN).

    Args:
        x: Spatial coordinates (N,).
        t: Time coordinates (N,).
        alpha: Thermal diffusivity.
        hidden_size: Size of the hidden layer.
        num_layers: Number of layers (simplified for demo).
        lr: Learning rate.
        num_steps: Number of training steps.

    Returns:
        u: Predicted solution u(x,t).
        loss_history: History of physics loss.
    """
    # Initialize weights (simplified for demo)
    # In practice, use a proper initialization (e.g., Xavier).
    weights = torch.randn(hidden_size * 2 + hidden_size + 1, device=x.device, dtype=torch.float32, requires_grad=True)

    # Move inputs to GPU
    x = x.cuda()
    t = t.cuda()

    # Training loop
    loss_history = []
    for step in range(num_steps):
        # Forward pass
        u = torch.empty_like(x)
        grid = lambda META: (x.numel(),)
        pinn_heat_equation_forward[grid](
            x, t, weights, u, x.numel(), alpha, hidden_size, num_layers
        )

        # Compute physics loss (autograd for exact derivatives)
        u_tensor = u.clone().requires_grad_(True)
        u_t = autograd.grad(u_tensor, t, grad_outputs=torch.ones_like(u_tensor), create_graph=True)[0]
        u_xx = autograd.grad(u_t.sum(), x, create_graph=True)[0]  # d²u/dx² ≈ d/dx (du/dt)
        physics_loss = ((u_t - alpha * u_xx) ** 2).mean()

        # Backward pass
        weights.grad = torch.zeros_like(weights)
        physics_loss.backward()

        # Update weights
        with torch.no_grad():
            weights -= lr * weights.grad

        # Log loss
        loss_history.append(physics_loss.item())
        if step % 100 == 0:
            print(f"Step {step}, Physics Loss: {physics_loss.item():.4f}")

    # Final prediction
    u_final = torch.empty_like(x)
    pinn_heat_equation_forward[grid](
        x, t, weights, u_final, x.numel(), alpha, hidden_size, num_layers
    )
    return u_final, loss_history

# -----------------------------------------------------------
# Test the Kernel
# -----------------------------------------------------------
if __name__ == "__main__":
    # Define the domain (x ∈ [0, 1], t ∈ [0, 1])
    N = 100
    x = torch.linspace(0, 1, N, device="cuda")
    t = torch.linspace(0, 1, N, device="cuda")

    # Solve the heat equation
    u_pred, loss_history = pinn_heat_equation_solve(x, t, alpha=0.1, num_steps=1000)

    # Plot the results (requires matplotlib)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Physics Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("PINN Training for 1D Heat Equation")
    plt.legend()
    plt.show()

    # Plot the solution at t=0.5
    t_idx = int(0.5 * N)
    plt.figure(figsize=(10, 5))
    plt.plot(x.cpu(), u_pred.cpu(), label=f"u(x, t={t[t_idx]:.2f})")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("PINN Solution to the 1D Heat Equation")
    plt.legend()
    plt.show()
