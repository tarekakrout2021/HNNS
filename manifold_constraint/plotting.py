import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch


from manifold_ode import compute_trajectory


def plot_trajectories(
    ic,
    vector_field_model,
    mesh_size=100,
    timestep=1e-2,
    number_of_timesteps=20,
    true_traj=None,
):
    """
    Plots the trajectories given a SphereFC model (vector_field_model). Starts with initial
    condition ic, and take steps number_of_timesteps for each initial condition with
    given timestep.

    mesh_size determines the size of the mesh to create the sphere.
    When true_traj is a list of tensors of trajectories starting from same ic and timeste/number_of_timesteps, it
    will plot these trajectories as well.
    """
    # Creates sphere first
    u = np.linspace(0, 2 * np.pi, mesh_size)
    v = np.linspace(0, np.pi, mesh_size)

    x_mesh = np.outer(np.cos(u), np.sin(v))
    y_mesh = np.outer(np.sin(u), np.sin(v))
    z_mesh = np.outer(np.ones(np.size(u)), np.cos(v))

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle("Trajectories computed")
    ax = fig.add_subplot(projection="3d")

    ax.plot_surface(x_mesh, y_mesh, z_mesh, rstride=4, cstride=4, color="b", alpha=0.1)

    # Computes the trajectory with the predictive model, and plots each trajectory as red dots.
    ic = ic.reshape(-1, 3)
    for x_traj in ic:
        points = compute_trajectory(
            x_traj.reshape(-1, 3), vector_field_model, timestep, number_of_timesteps
        )
        traj = np.array([i.detach().numpy() for i in points]).reshape(-1, 3)

        for i in range(0, traj.shape[0], 1):
            ax.plot(traj[i : i + 2, 0], traj[i : i + 2, 1], traj[i : i + 2, 2], "ro-")

    red_patch = mpatches.Patch(color="red", label="Prediction")

    # Plots the true trajectories as blue crosses.
    if true_traj is not None:
        assert (
            timestep == 1e-2 and number_of_timesteps == 20
        ), "Require timestep and number of timesteps to be default values"
        for i in range(true_traj[0].shape[0]):
            traj = np.array(
                [(true_traj[j])[i, :].detach().numpy() for j in range(len(true_traj))]
            ).reshape(-1, 3)
            for j in range(0, traj.shape[0], 1):
                ax.plot(
                    traj[j : j + 2, 0],
                    traj[j : j + 2, 1],
                    traj[j : j + 2, 2],
                    "bx-",
                    alpha=0.1,
                )

        blue_patch = mpatches.Patch(color="blue", label="True value")
        ax.legend(handles=[red_patch, blue_patch])

    else:
        ax.legend(handles=[red_patch])

    return fig, ax
