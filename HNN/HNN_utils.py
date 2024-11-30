import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim


# Data Loss
def data_loss(model: nn.Module, data_dl: list) -> torch.Tensor:
    """Compute the mean suqared error data loss.

    Args:
        model (nn.Module): The Hamiltonian neural network.
        data_dl (list): [q_dl, p_dl, h_dl], where q_dl and p_dl are phase space coordinates,
                        and h_dl is true hamiltonian evaluated at (q_dl, p_dl).
                        Each item in the list has a dimension (batch_size, 1).  # ???????each item ?
    Returns:
        torch.Tensor: Data loss (MSE of the true and predicted hamiltonian on the data points.) shape(1, )
    """
    y_pred = model(data_dl[0], data_dl[1])
    y_true = data_dl[2]
    loss_function = nn.MSELoss()

    loss = loss_function(y_pred, y_true)
    return loss


# Define the physics-informed loss term (Impose Hamilton's equations as a soft constraint)
def hamiltonian_loss(
    model: nn.Module, q: torch.Tensor, p: torch.Tensor
) -> torch.Tensor:
    """
    Enforce a soft contraint to satisfy the Hamilton's equations at the collocation points via a loss term.
    This is typically referred to as the 'physics-informed' loss term.

    Args:
        model (nn.Module): The Hamiltonian neural network.
        q (torch.Tensor): Generalized position (batch_size, 1).
        p (torch.Tensor): Generalized momentum (batch_size, 1).

    Returns:
        torch.Tensor: Physics-informed loss.
    """

    q.requires_grad_()
    p.requires_grad_()

    h_pred = model(q, p)

    batch_size = q.shape[0]

    dh_dq = torch.zeros(batch_size, 1)
    dh_dp = torch.zeros(batch_size, 1)
    for batch in range(batch_size):  # iterate over the batches
        gradients = torch.autograd.grad(
            h_pred[batch],
            [q, p],
            grad_outputs=torch.ones_like(h_pred[batch]),
            create_graph=True,
        )
        dh_dq[batch] = gradients[0][batch]
        dh_dp[batch] = gradients[1][batch]

    dq_dt_pred = -dh_dq
    dp_dt_pred = dh_dp

    dq_dt, dp_dt = set_vector_field_pendulum(q, p)

    loss_function = nn.MSELoss()

    loss = loss_function(dq_dt, dq_dt_pred) + loss_function(dp_dt_pred, dp_dt)
    return loss


def set_vector_field_pendulum(
    q: torch.Tensor, p: torch.Tensor
) -> list[torch.Tensor, torch.Tensor]:
    """Return true vector field for the single pendulum.

    Args:
        q (torch.Tensor): Generalized position (batch_size, 1).
        p (torch.Tensor): Generalized momentum (batch_size, 1).

    Returns:
        list[torch.Tensor, torch.Tensor]: dq_dt_true, dp_dt_true
    """
    dq_dt, dp_dt = p, -torch.sin(q)
    return [dq_dt, dp_dt]


# Total Loss
def total_loss(
    model: nn.Module,
    data_physics: list,
    data_supervised: list,
    lambda_physics: float = 1.0,
    lambda_data: float = 0.1,
) -> torch.Tensor:
    """Compute the total loss by adding the physics and data loss terms.

    Args:
        model (nn.Module): The Hamiltonian neural network.
        data_physics (List): Collocation points in phase space [q, p], where q and p with shapes (batch_size, 1)
                                are points at which the Hamilton's equations are imposed as soft constraints.
        data_supervised (list): [q_dl, p_dl, h_dl], where q_dl (shape = (n_sup, 1)), p_dl (shape = (n_sup, 1))
                                are phase space coordinates, and h_dl (shape = (n_sup, 1)) is the true Hamiltonian
                                evaluated at (q_dl, p_dl) and n_sup is the number of points where true Hamiltonian is
                                known.
        lambda_physics (float, optional): Weighting parameter for the physics loss term. Defaults to 1.0.
        lambda_data (float, optional): Weighting parameter for the data loss term. Defaults to 0.8.

    Returns:
        torch.Tensor: Total loss.
    """
    return lambda_physics * hamiltonian_loss(
        model, data_physics[0], data_physics[1]
    ) + lambda_data * data_loss(model, data_supervised)


# Training function for the Hamiltonian Neural Network
def train_hnn(
    model: nn.Module,
    dataloader: DataLoader,
    true_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    num_epochs: int = 1000,
    learning_rate: float = 0.001,
) -> list:
    """Training loop for the Hamiltonian neural network.

    Args:
        model (nn.Module): Hamiltonian neural network.
        dataloader (DataLoader): Dataloader for training data.
        true_data (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Generalized positions (q), momenta (p) where true Hamiltonian is known (for supervised (data) loss term).
        num_epochs (int, optional): Number of epochs. Defaults to 1000.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.

    Returns:
        list[float]: Training loss over epochs.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for q_batch, p_batch in dataloader:
            model.train()
            optimizer.zero_grad()

            loss_batch = total_loss(model, [q_batch, p_batch], true_data)

            loss_batch.backward()
            optimizer.step()

            epoch_loss += loss_batch
        print(f"loss at epoch {epoch} = {epoch_loss}")
        losses.append(epoch_loss.item())

    return losses


# Define the true Hamiltonian function for the single pendulum
def true_hamiltonian(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Compute the true Hamiltonian of the system.

    Args:
        q (torch.Tensor): Generalized position.
        p (torch.Tensor): Generalized momentum.

    Returns:
        torch.Tensor: Ground truth H(q, p).
    """
    return 0.5 * p**2 + (1 - np.cos(q))


# Plot the loss curve
def plot_loss_curve(losses):
    """Plot a loss curve.

    Args:
        losses (list): Plot loss Vs Epoch.
    """
    print(type(plt))  # Should output: <class 'module'>
    # plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss for Hamiltonian Neural Network")
    plt.savefig("loss_curve.png")


# Hint: Can use the following function to compare the true and learned Hamiltonians!
def compare_hamiltonian(
    Q: np.ndarray, P: np.ndarray, H_true: np.ndarray, H_learned: np.ndarray
) -> None:
    """Compare true and learned Hamiltonian functions.

    Args:
        Q (np.ndarray): Test points of the generalized position (n_test, n_test).
        P (np.ndarray): Test points of the generalized momentum (n_test, n_test).
        H_true (np.ndarray):True Hamiltonian (n_test, n_test).
        H_learned (np.ndarray): Learned Hamiltonian (n_test, n_test).
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Plot the true and learned Hamiltonians
    contour1 = axes[0].contourf(Q, P, H_true, levels=200, cmap="viridis")
    fig.colorbar(contour1, ax=axes[0])
    axes[0].set_title("True Hamiltonian")
    axes[0].set_xlabel("q")
    axes[0].set_ylabel("p")

    # Plot the learned Hamiltonian
    contour2 = axes[1].contourf(Q, P, H_learned, levels=200, cmap="viridis")
    fig.colorbar(contour2, ax=axes[1])
    axes[1].set_title("Learned Hamiltonian")
    axes[1].set_xlabel("q")
    axes[1].set_ylabel("p")
    plt.suptitle("Comparison of True and Learned Hamiltonians")
    plt.tight_layout()
    plt.savefig("h_comparison.png")


# The following function is NOT tested!
def generate_data(num_samples: int = 1000, batch_size: int = 32) -> DataLoader:
    """
    Generate synthetic trajectory data for a single pendulum system and return a DataLoader.

    Args:
        num_samples (int): Number of data points to generate.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: Dataloader providing batches of (q, p).
    """
    q = torch.linspace(-1, 1, num_samples).view(-1, 1)
    p = torch.linspace(-1, 1, num_samples).view(-1, 1)
    dataset = TensorDataset(q, p)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader


# Compute vector field from the hamiltonian
def compute_vector_field_from_hamiltonian(hamiltonianNN, q, p):
    """
    Compute the time derivatives (dq/dt, dp/dt) from the Hamiltonian.

    Parameters:
    - hamiltonianNN: HamiltonianNN, the learned Hamiltonian neural network.
    - q: Tensor, position.
    - p: Tensor, momentum.

    Returns:
    - dq_dt: Tensor, time derivative of position.
    - dp_dt: Tensor, time derivative of momentum.
    """

    # Enable autograd for q and p to compute partial derivatives
    q = q.detach().clone().requires_grad_(True)
    p = p.detach().clone().requires_grad_(True)

    # Compute the Hamiltonian
    H = hamiltonianNN(q, p)

    # Calculate dH/dq and dH/dp
    grad_H = torch.autograd.grad(
        H, (q, p), grad_outputs=torch.ones_like(H), create_graph=True
    )
    dH_dq, dH_dp = grad_H[0], grad_H[1]

    # dq/dt and dp/dt according to the Hamilton's equations
    dq_dt = dH_dp
    dp_dt = -dH_dq

    return dq_dt, dp_dt


# Symplectic Euler method step
def symplectic_euler_step(hamiltonian, q, p, dt):
    """
    Perform one step of the Symplectic Euler method.

    Parameters:
    - hamiltonian: HamiltonianNN, the learned Hamiltonian neural network.
    - q: Tensor, position at the current step.
    - p: Tensor, momentum at the current step.
    - dt: float, time step size.

    Returns:
    - q_next: Tensor, updated position.
    - p_next: Tensor, updated momentum.
    """
    # Compute momentum update first (uses current q, p)
    dq_dt, dp_dt = compute_vector_field_from_hamiltonian(hamiltonian, q, p)
    p_next = p + dp_dt * dt

    # Update position using the new momentum
    dq_dt_next, _ = compute_vector_field_from_hamiltonian(hamiltonian, q, p_next)
    q_next = q + dq_dt_next * dt

    return q_next, p_next


def integrate_trajectory(hamiltonian, q0, p0, dt, n_steps):
    """
    Integrates the trajectory using the learned Hamiltonian.

    Parameters:
    - hamiltonian: HamiltonianNN, the learned Hamiltonian neural network.
    - q0: Tensor, initial position.
    - p0: Tensor, initial momentum.
    - dt: float, time step size.
    - n_steps: int, number of integration steps.

    Returns:
    - trajectory: list of tuples (q, p) representing the trajectory.
    """
    trajectory = [(q0, p0)]
    q, p = q0, p0

    for _ in range(n_steps):
        q, p = symplectic_euler_step(hamiltonian, q, p, dt)
        trajectory.append((q, p))

    return trajectory


# Plot trajectory in phase space
# Plot trajectory in phase space
def plot_trajectory(positions: np.ndarray, momenta: np.ndarray):
    """Plot the trajectory generated with the learned Hamiltonian.

    Args:
        positions (np.ndarray): Positions.
        momenta (np.ndarray): Momenta.
    """
    plt.figure(figsize=(3, 3))
    plt.plot(momenta, positions, label="Trajectory")
    plt.xlabel("p")
    plt.ylabel("q")
    plt.title("Phase Space Trajectory")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("hnn_dynamics.png")
    plt.show()


def plot_hamilton_evolution(
    t_traj: np.ndarray, h_traj: np.ndarray, filename="hamiltonian_evolution.png"
):
    """Plot evolution of the Hamiltonian in time.

    Args:
        t_traj (np.ndarray): time-steps (shape: (n_timesteps, ))
        h_traj (np.ndarray): h(t_traj) (shape, n_timesteps,)
        filename (str): Figure name to be saved.
    """
    plt.figure(figsize=(3, 3))
    plt.plot(t_traj, h_traj)
    plt.xlabel("t")
    plt.ylabel("H(p,q)")
    plt.title("Energy Vs Time")
    plt.show()
    plt.tight_layout()
    plt.savefig(filename)


# Compute Hamiltonian
def compute_hamiltonian_pendulum(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamiltonin function h evaluated p, q that is h(p,q) for a single pendulum.

    Args:
        p (np.ndarray): Phase space coordinate (generalized momenta).
        q (np.ndarray): Phase space coordinate (generalized positions).

    Returns:
        np.ndarray: h(p, q)
    """
    return 0.5 * p**2 + (1 - np.cos(q))
