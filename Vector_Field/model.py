import torch.nn as nn


# Define a neural network to approximate the vector field
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden_layers):
        super(NeuralNetwork, self).__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]

        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
