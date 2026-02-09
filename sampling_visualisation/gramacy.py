import torch

class Gramacy:
    def __init__(self):
        self.optimum = torch.tensor([[0.1954, 0.4044]], dtype=torch.double)
        self.x_dim = 2
        self.n_constraints = 2
        # all axis are between 0 and 1
        self.range = torch.zeros((self.x_dim, 2))
        self.range[:, 1] = 1
        self.name = 'LSQ function'
        self.cite = 'Gramacy et al., 2016'

    def __call__(self, x):
        result = torch.empty((x.shape[0], 3))
        x0, x1 = x[:, 0], x[:, 1]
        result[:, 0] = x0 + x1
        result[:, 1] = x0 + 2 * x1 + torch.sin(2 * torch.pi * (x0 ** 2 - 2 * x1)) / 2 - 1.5
        result[:, 2] = 1.5 - x0 ** 2 - x1 ** 2
        return result.to(x)