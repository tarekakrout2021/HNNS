import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import NeuralNetwork
from utils import generate_training_data, train_vector_field_model
from utils import compute_vector_field_pendulum, compute_hamiltonian_pendulum
from utils import plot_hamilton_evolution, plot_vec_field_traj
from scipy.integrate import solve_ivp

torch.manual_seed(123)
np.random.seed(123)


if __name__ == "__main__":
    # Instantiate the NN model to learn vector field: nn_v
    input_dim = (2,)
    output_dim = (2,)
    hidden_dim = (64,)
    n_hidden_layers = 3
    nn_v = NeuralNetwork(input_dim=2, output_dim=2, hidden_dim=64, n_hidden_layers=3)

    # Define the loss and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(nn_v.parameters(), lr=0.001)

    # Generate training data
    seed = 123

    # Generate training data
    X_train, Y_train = generate_training_data(n_train=1000, seed=123)

    # Train the neural network to learn the vector field
    losses = train_vector_field_model(
        nn_v,
        X_train,
        Y_train,
        num_epochs=50,
        learning_rate=0.001,
        batch_size=64,
        verbose=True,
    )

    # Generate test data
    n_test = 30
    p_test = torch.linspace(-1, 1, n_test)
    q_test = torch.linspace(-1, 1, n_test)
    P, Q = torch.meshgrid(p_test, q_test)
    X_test = torch.stack([P.ravel(), Q.ravel()], axis=-1)
    print(X_test.shape)
    Y_test = compute_vector_field_pendulum(X_test)
    print(Y_test.shape)

    # Evaluate the model on test data
    nn_v.eval()
    with torch.no_grad():
        predictions = nn_v(X_test)
        test_loss = loss_function(predictions, Y_test)
        print(f"Test Loss: {test_loss.item():.4f}")

    # Reshape to match the grid shape for plotting
    U = predictions[:, 0].reshape((n_test, n_test))
    V = predictions[:, 1].reshape((n_test, n_test))

    # Set initial conditions and time span for the integration
    initial_position = [0.7, 0.7]  # Starting point for the trajectory
    t_span = (0, 500)  # Time interval
    t_eval = np.linspace(*t_span, 2000)  # Points at which to evaluate the solution

    # Define the neural network-based vector field function for solve_ivp
    def nn_vector_field(t, y, model):
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(
            0
        )  # Reshape to (1, 2)
        with torch.no_grad():
            vector = (
                model(y_tensor).numpy().flatten()
            )  # Get output from the model and flatten to (2,)
        return vector

    def true_vector_field(t, y):
        return (
            compute_vector_field_pendulum(
                torch.tensor(y, dtype=torch.float32).unsqueeze(0)
            )
            .numpy()
            .flatten()
        )


    # Evaluate trajectory in the phase-space with the true vector field
    trajectory_true = solve_ivp(
        true_vector_field,
        t_span,
        initial_position,
        t_eval=t_eval,
        method="RK23",
        atol=1e-3,
        rtol=1e-3,
    )
    p_traj_true = trajectory_true.y[0]
    q_traj_true = trajectory_true.y[1]
    t_traj_true = trajectory_true.t

    # Evaluate trajectory in the phase-space with the learned vector field
    trajectory_learned = solve_ivp(
        nn_vector_field,
        t_span,
        initial_position,
        t_eval=t_eval,
        method="RK23",
        atol=1e-3,
        rtol=1e-3,
        args=(nn_v,),
    )
    p_traj_learned = trajectory_learned.y[0]
    q_traj_learned = trajectory_learned.y[1]
    t_traj_learned = trajectory_learned.t

    # Plot the true vector field
    plot_vec_field_traj(
        x=P,
        y=Q,
        v_x=U,
        v_y=V,
        p_traj=p_traj_true,
        q_traj=q_traj_true,
        t_traj=t_traj_true,
        initial_position=initial_position,
        filename="vf_true.png",
    )

    # Plot the learned vector field
    plot_vec_field_traj(
        x=P,
        y=Q,
        v_x=U,
        v_y=V,
        p_traj=p_traj_learned,
        q_traj=q_traj_learned,
        t_traj=t_traj_learned,
        initial_position=initial_position,
        filename="vf_learned.png",
    )

    # Compute Hamiltonian along the true trajectory
    h_traj_true = compute_hamiltonian_pendulum(
        trajectory_true.y[0], trajectory_true.y[1]
    )

    # Compute Hamiltonian along the learned trajectory
    h_traj_learned = compute_hamiltonian_pendulum(
        trajectory_learned.y[0], trajectory_learned.y[1]
    )

    # Plot the Hamiltonian as a function of time for a single trajectory in phase-space
    plot_hamilton_evolution(trajectory_true.t, h_traj_true, filename="h_true")

    # Plot the Hamiltonian as a function of time for a single trajectory in phase-space
    plot_hamilton_evolution(trajectory_learned.t, h_traj_learned, filename="h_learned")
