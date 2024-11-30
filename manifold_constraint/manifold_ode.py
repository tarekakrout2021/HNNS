import torch
import torch.nn as nn


class SphereFC(nn.Module):
    def __init__(self, dim_network, dim_input, dim_output, preprojected=True):
        """
        Initialize a network with len(dim_network) layers and tanh activation function.
        Each entry into dim_network list is the dimension of the corresponding layer. So
        dim_network = [10, 20, 10] makes a network with three hidden layers with 10, 20,
        and 10 neurons respectively.

        Parameters:
        ------------
        dim_network: list with positive integers.
            len(dim_network) is number of layers, each entry is number of neurons for given layer.
        dim_input: integer.
            dimension of input, which will be 3 in our case.
        dim_output: integer.
            dimension of vector field, which is 3 as we are working in the embedding space.
        preprojected: boolean.
            If true, the output of the network (forward method) does not project down to tangent space. Can be used to train
            with non-projected output. When computing trajectories, set to False.
        """
        super().__init__()
        layers = []

        self.dim_input = dim_input
        self.preprojected = preprojected

        prev_layer_dim = dim_input
        for dim_layer in dim_network:
            linear_layer = nn.Linear(prev_layer_dim, dim_layer)
            activation_func = nn.Tanh()  # Feel free to change activation function

            layers.append(linear_layer)
            layers.append(activation_func)

            prev_layer_dim = dim_layer

        layers.append(nn.Linear(prev_layer_dim, dim_output))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Outputs element of the vector field. Projected to tangent space if self.preprojected is False.

        Parameters:
        ------------
        x: torch.tensor float of dimension Nx3
        """
        output = self.model(x)
        if self.preprojected:
            return output

        output = self.project_to_tangent(x, output)
        return output

    def project_to_tangent(self, p, v):
        """
        Projects the velocity v to the tangent space of point p. For each point in p, the reurning value is (I-pp^T)v.
        Function should be able to handle multiple points in p.

        Parameters:
        -------------
        p: torch.tensor float of dimension Nx3 or (3,)
            Defines the points which induced the tangent space
        v: torch.tensor of dimension Nx3 or (3,)
            velocity in the embedding space

        Returns:
        -----------
        projected_vec: torch.tensor of dimension Nx3 or (3,)
            Resulting tangent vector to tangent space at points p.

        Tip to handle outer product when N>1, look up torch.einsum.
        """
        # Add batch dimension
        if p.shape == (1, 3) or p.ndim == 1:
            p = torch.reshape(p, (1, 3))
            v = torch.reshape(v, (1, 3))

        ppT = torch.einsum("bi,bj->bij", p, p)  # cool way of calculating outer product
        I = torch.eye(3).expand_as(ppT)  # Identity matrix for each batch

        projection_matrix = I - ppT

        projected_vec = torch.einsum("bij,bj->bi", projection_matrix, v)

        return projected_vec.squeeze(0) if projected_vec.size(0) == 1 else projected_vec

    def exponential_map(self, p, v, timestep=1):
        """
        Maps from tangent space of current_point p to the manifold after taken a step in
        direction of velocity v with given timestep. I.e., return exp_p(timestep*v)

        Parameters:
        ------------
        p: torch.tensor float of dimension Nx3 or (3,)
            Defines the points which induced the tangent space
        v: torch.tensor of dimension Nx3 or (3,)
            velocity in the embedding space
        timestep: float >0
            how long a step when moving in tangent space

        Returns:
        ---------
        new_point: torch.tensor of dimension Nx3 or (3,)
            Point mapped down to the manifold after taking timestep*v steps in tangent space at points p
        """
        if p.shape == (1, 3) or p.ndim == 1:
            p = torch.reshape(p, (1, 3))
            v = torch.reshape(v, (1, 3))

        v_norm = torch.linalg.norm(v, dim=1, keepdim=True)
        new_point = (
            torch.cos(timestep * v_norm) * p + torch.sin(timestep * v_norm) * v / v_norm
        )

        if new_point.size(0) == 1:
            new_point = new_point.reshape(-1)

        return new_point


def compute_trajectory(
    initial_point, vector_field_model, timestep=1e-2, number_of_timesteps=20
):
    """
    Computes trajectory from initial_point, using SphereFC model vector_field_model to predict
    the vector field in the tangent space, and then use exponential map to take a step with given
    timestep, and predict the next steps number_of_timesteps.

    Should be able to handle many inital points and return list of length number_of_timesteps,
    where each entry is the next step, which is a tensor of dimension inital_point.shape.
    """

    points = [initial_point.reshape(-1)]

    x = initial_point
    for _ in range(number_of_timesteps):
        v = vector_field_model(x)
        next_point = vector_field_model.exponential_map(x, v, timestep)
        x = next_point

        points.append(next_point)

    return points
