import torch
import numpy as np
from HNN_utils import generate_data, true_hamiltonian
from HNN_utils import (
    train_hnn,
    plot_loss_curve,
    compare_hamiltonian,
    compute_hamiltonian_pendulum,
)
from HNN_utils import integrate_trajectory, plot_trajectory, plot_hamilton_evolution
from HNN import HamiltonianNN

if __name__ == "__main__":
    # Instantiate and train the model
    model = HamiltonianNN(hidden_dim=200)

    # dataset, dataloader = generate_trajectory_data(num_samples=2000, batch_size=500)
    dataset, dataloader = generate_data(num_samples=4000, batch_size=500)

    # Data for data loss (values where the ground truth is known)
    q_data_loss = torch.tensor(
        [dataset[0][0], dataset[-1][0], dataset[0][0], dataset[-1][0]]
    ).view(-1, 1)
    p_data_loss = torch.tensor(
        [dataset[0][1], dataset[-1][1], dataset[-1][1], dataset[0][1]]
    ).view(-1, 1)

    # Debugging
    print(q_data_loss.shape)
    print(p_data_loss.shape)

    print(q_data_loss)

    data_dl = [q_data_loss, p_data_loss, true_hamiltonian(q_data_loss, p_data_loss)]

    # Uncomment the following for Lotka-Volterra system (Not tested for this exercise!)
    # lotka_volterra_params = [-2, -1, -1, -1]
    # data_dl = [q_data_loss, p_data_loss, true_hamiltonian(q_data_loss, p_data_loss, example='lotka-volterra', args=lotka_volterra_params)]

    # Train the Hamiltonian Neural Network and compute losses at each epoch
    print("Training the Hamiltonian Neural Network: ")
    losses = train_hnn(
        model,
        dataloader,
        true_data=data_dl,
        num_epochs=100,
        learning_rate=0.001,
    )
    print("Training finished!")

    # Plot loss curve
    plot_loss_curve(losses=losses)

    # Test data: Generate a grid of values for q and p
    n_test_per_dim = 200
    q_values = torch.linspace(-1, 1, n_test_per_dim)
    p_values = torch.linspace(-1, 1, n_test_per_dim)
    Q, P = torch.meshgrid(q_values, p_values)
    Q_flat, P_flat = Q.reshape(-1, 1), P.reshape(-1, 1)

    # Compute the true Hamiltonian on the grid
    H_true = (
        true_hamiltonian(Q_flat, P_flat)
        .reshape(n_test_per_dim, n_test_per_dim)
        .detach()
        .numpy()
    )

    # Uncomment the following for Lotka-Volterra (Not tested for this exercise!)
    # H_true = true_hamiltonian(Q_flat, P_flat, example='lotka-volterra', args=lotka_volterra_params).reshape(n_test_per_dim, n_test_per_dim).detach().numpy()

    # Compute the learned Hamiltonian on the same grid using the neural network
    H_learned = (
        model(Q_flat, P_flat).reshape(n_test_per_dim, n_test_per_dim).detach().numpy()
    )

    # Plot true and learned Hamiltonian
    compare_hamiltonian(Q.numpy(), P.numpy(), H_true, H_learned)

    # Initial conditions
    q0 = torch.tensor([[0.7]])  # Initial position
    p0 = torch.tensor([[0.7]])  # Initial momentum

    # Integration parameters
    dt = 0.001  # Time step size
    n_steps = 50000  # Number of integration steps

    # Integrate trajectory
    trajectory = integrate_trajectory(model, q0, p0, dt, n_steps)

    # Extract positions and momenta for plotting
    positions = [q.detach().numpy().flatten() for q, p in trajectory]
    momenta = [p.detach().numpy().flatten() for q, p in trajectory]

    # Plot trajectory using the learned Hamiltonian
    plot_trajectory(positions=positions, momenta=momenta)

    h_traj = compute_hamiltonian_pendulum(
        np.array(momenta), np.array(positions)
    ).reshape(
        -1,
    )
    plot_hamilton_evolution(
        t_traj=np.linspace(0, dt * n_steps, n_steps + 1), h_traj=h_traj
    )
