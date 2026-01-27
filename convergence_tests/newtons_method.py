import torch

class NewtonsMethod:
    """
        Class that holds a set of points and a surrogate.
        It's able to aply newton's method to those points on the surrogate.
        The surrogate just needs to be a differentiable function taking and returning torch tensors.
    """
    def __init__(self, surrogate, points = None):
        self.surrogate = surrogate
        self.points = torch.empty(0) if points is None else points
        # also store the initial points
        self.history = [self.points.detach().clone().requires_grad_(False)]
    
    def val_gradient_at(self, x):
        assert x.shape[1] == self.points.shape[1]
        # df/df = 1, we want the plain gradient of the inputs
        x.requires_grad_()
        out = self.surrogate(x)
        out.backward(torch.ones(out.shape), inputs=x)
        return out.detach(), x.grad
    
    def step(self):
        # get y values and gradients at all points
        y, y_grad = self.val_gradient_at(self.points)
        # n is dimensionality of the inputs
        n = self.points.shape[1]
        # multi dimensional newtons method using the moore penrose inverse of the gradient
        self.points = self.points - torch.mul(torch.div(1, n * y_grad), y)
        # save a copy of the new set of points
        self.history.append(self.points.detach().clone().requires_grad_(False))