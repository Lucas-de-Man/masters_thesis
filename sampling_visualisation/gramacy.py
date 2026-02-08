import torch

class Gramecy:
    def __init__(self):
        pass

    def __call__(self, x):
        result = torch.empty((x.shape[0], 3))
        x0, x1 = x[:, 0], x[:, 1]
        result[:, 0] = x0 + x1
        result[:, 1] = x0 + 2 * x1 + torch.sin(2 * torch.pi * (x0 ** 2 - 2 * x1)) / 2
        result[:, 2] = 1.5 - x0 ** 2 - x1 ** 2
        return result