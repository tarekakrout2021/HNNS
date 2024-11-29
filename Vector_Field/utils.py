import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch
import torch.utils
import torch.autograd as autograd


# Generate synthetic data
def compute_vector_field_pendulum(X: torch.tensor) -> torch.tensor:
    """Phase-space coordinates X = [P, Q] with shape (n_points, 2) with columns
    containing generalized momenta (P) and positions (Q)

    Args:
        X (torch.tensor): [P, Q]

    Returns:
        torch.tensor: [dP_dt, dQ_dt] with shape (n_points, 2)
    """

    res = torch.zeros_like(X)
    # iterate over the samples
    for i in range(X.size(0)):
        p = X[i][0]
        q = X[i][1]
        res[i][0] = -np.sin(q)
        res[i][1] = p

    return res


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


def generate_training_data(
    n_train: int = 1000, seed: int = 123
) -> tuple[torch.tensor, torch.tensor]:
    """Generate training data randomly (NOT with np.linspace(...)).

    Args:
        n_train (int, optional): number of training points in phase space. Defaults to 1000.
        seed (int, optional): Random seed for torch. Defaults to seed.

    Returns:
        tuple[torch.tensor, torch.tensor]: X_train (training points) and Y_train (Ground truth at training points).
                                            Both X_train and Y_train have shapes (n_train, 2).
    """

    X_train = torch.rand(n_train, 2) * 2 - 1  # to be in range [-1, 1]
    Y_train = compute_vector_field_pendulum(X_train)

    return [X_train, Y_train]


def train_vector_field_model(
    model,
    X_train,
    Y_train,
    num_epochs=500,
    learning_rate=0.001,
    batch_size=64,
    verbose=True,
):
    """
    Train a neural network model to learn a 2D vector field.

    Args:
        model (nn.Module): The neural network model.
        X_train (torch.Tensor): Training inputs, shape (N, 2).
        Y_train (torch.Tensor): Training outputs, shape (N, 2).
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        verbose (bool): If True, print loss every 50 epochs.

    Returns:
        List of training losses for each epoch.
    """

    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    training_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size
    )
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ret = []
    for epoch in range(num_epochs):
        total_loss = 0
        for i, data in enumerate(training_loader):
            x, y = data

            model.train()
            optimizer.zero_grad()

            y_pred = model(x)

            loss = loss_function(y_pred, y)
            loss.backward()
            total_loss += loss.item()

            optimizer.step()

        if verbose is True and (epoch + 1) % 50 == 0:
            print(f"total loss at {epoch + 1 } : {total_loss}")
        ret.append(total_loss)

    return ret


def plot_vec_field_traj(
    x,
    y,
    v_x,
    v_y,
    p_traj,
    q_traj,
    t_traj,
    initial_position,
    filename="vector_field.png",
):
    # Create a quiver plot with colors based on vector magnitude
    plt.figure(figsize=(4, 3))

    # Compute the magnitude of the vectors for coloring
    magnitude = np.sqrt(v_x**2 + v_y**2)

    plt.quiver(x, y, v_x, v_y, magnitude, cmap="viridis", scale=10)
    # plt.colorbar(label='Magnitude (vector field)', location='bottom', shrink=0.8)

    # Prepare the trajectory for gradient color plotting
    points = np.array([p_traj, q_traj]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a LineCollection with a colormap
    norm = plt.Normalize(0, t_traj[-1])  # Normalize time for color mapping
    lc = LineCollection(segments, cmap="bwr", norm=norm)
    lc.set_array(t_traj)  # Color by time
    lc.set_linewidth(2)

    # Add the LineCollection to the plot
    plt.gca().add_collection(lc)
    plt.colorbar(lc, label="time")
    plt.scatter(
        *initial_position, color="blue", s=100, label="Start", marker="x"
    )  # Mark the start point
    plt.scatter(
        p_traj[-1], q_traj[-1], color="red", s=100, label="End", marker="x"
    )  # Mark the start point
    plt.xlabel("p")
    plt.ylabel("q")
    plt.title("Vector Field")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)


# Figure: Hamiltonin Vs Time
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
