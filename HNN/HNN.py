import torch
import torch.nn as nn


class HamiltonianNN(nn.Module):
    """Hamiltonian Neural Network model."""

    def __init__(self, hidden_dim=64):
        super(HamiltonianNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, q, p) -> torch.Tensor:
        """Perform a forward pass with the Hamiltonian neural network and compute the Hamiltonian.

        Args:
            q (torch.Tensor): Phase space coordinate (batch_size, 1).
            p (torch.Tensor): Phase space coordinate (batch_size, 1).

        Returns:
            torch.Tensor: Learned Hamiltonian (batch_size, 1).
        """
        X = torch.hstack((q, p))
        batch_size = q.shape[0]
        return self.model(X).reshape((batch_size, 1))
