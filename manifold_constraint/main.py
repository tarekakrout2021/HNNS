import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

import matplotlib.pyplot as plt

from manifold_ode import SphereFC, compute_trajectory
from plotting import plot_trajectories

rng = np.random.default_rng(2)
torch.manual_seed(rng.random(size=1))


BATCH_SIZE = 50
EPOCHS = 500
LEARNING_RATE = 1e-3
WEIGTH_DECAY = 0

DIM_NETWORK = [
    100,
    100,
    100,
]  # Network with three layers with 10 20 15 neurons. Feel free to change.

DIM_INPUT = 3
DIM_OUTPUT = 3

TRAINING_PREPROJECTED = (
    True  # True if training with vector field not projected to tangent space.
)
# False if training with output in tangent field.

TRAINING = True  # True if want to train model
PLOTTING = True  # True if want to plot trajectories with trained model
PLOT_TRUE_TRAJ = True  # True if one want to plot the true trajectory as well.

NUM_IC_PLOT = 10  # Number of trajectories to plot from validation set.
TIMESTEP_PLOT = 1e-2  # Time step when computing trajectories.
NUMBER_OF_TIMESTEPS = 20  # Number of timesteps to take when computing trajectories.
MESH_SIZE = 100  # Size of mesh when creating sphere when plotting

if __name__ == "__main__":
    # Training set and validation set. Both the true value without projecting to tangent space and with projection.
    X_train, Y_train_preprojected, Y_train_projected = torch.load(
        "training_data.pt", weights_only=True
    )
    X_valid, Y_valid_preprojected, Y_valid_projected = torch.load(
        "validation_data.pt", weights_only=True
    )
    X_valid = X_valid.float()

    if TRAINING:
        if TRAINING_PREPROJECTED is True:
            dl_training = DataLoader(
                TensorDataset(
                    X_train.reshape(-1, 3).float(),
                    Y_train_preprojected.reshape(-1, 3).float(),
                ),
                batch_size=BATCH_SIZE,
                shuffle=True,
            )
            Y_valid = Y_valid_preprojected.float()

        else:
            dl_training = DataLoader(
                TensorDataset(
                    X_train.reshape(-1, 3).float(),
                    Y_train_projected.reshape(-1, 3).float(),
                ),
                batch_size=BATCH_SIZE,
                shuffle=True,
            )
            Y_valid = Y_valid_projected.float()

        vector_field_model = SphereFC(
            dim_network=DIM_NETWORK,
            dim_input=DIM_INPUT,
            dim_output=DIM_OUTPUT,
            preprojected=TRAINING_PREPROJECTED,
        )

        optimizer = Adam(
            vector_field_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGTH_DECAY
        )
        loss_fun = MSELoss()

        for epoch in range(EPOCHS):
            for X, Y in dl_training:
                optimizer.zero_grad()

                predictions = vector_field_model(X)
                loss = loss_fun(predictions, Y)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                predictions = vector_field_model(X_valid)
                valid_loss = loss_fun(predictions, Y_valid)

                train_loss = 0
                for X, Y in dl_training:
                    predictions = vector_field_model(X)
                    train_loss += loss_fun(predictions, Y)

                train_loss /= len(dl_training)

                print(
                    f"Epoch {epoch+1}; Training loss {train_loss:5.5f}; Validation loss {valid_loss:5.5f}"
                )

        torch.save(vector_field_model.state_dict(), "vector_field_model.pt")
        torch.save([DIM_NETWORK, DIM_INPUT, DIM_OUTPUT], "vector_field_architecture.pt")

    if PLOTTING:
        with torch.no_grad():
            true_traj_valid = None
            if PLOT_TRUE_TRAJ:
                true_traj_valid = torch.load(
                    "true_traj_validation.pt", weights_only=True
                )
                true_traj_valid = [p[:NUM_IC_PLOT, :] for p in true_traj_valid]

            dim_network, dim_input, dim_output = torch.load(
                "vector_field_architecture.pt", weights_only=True
            )
            vector_field_model = SphereFC(
                dim_network=dim_network,
                dim_input=dim_input,
                dim_output=dim_output,
                preprojected=False,
            )
            vector_field_model.load_state_dict(
                torch.load("vector_field_model.pt", weights_only=True)
            )

            fig, ax = plot_trajectories(
                X_valid[:NUM_IC_PLOT, :],
                vector_field_model,
                MESH_SIZE,
                TIMESTEP_PLOT,
                NUMBER_OF_TIMESTEPS,
                true_traj_valid,
            )

            # Comment out depending on if one want to show plot or store it
            plt.show()
            fig.savefig("trajectories_prediction.png")
