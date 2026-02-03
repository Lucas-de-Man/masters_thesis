import torch

class NewtonsMethod:
    """
        Class that holds a set of points and a surrogate.
        It's able to aply newton's method to those points on the surrogate.
        The surrogate just needs to be a differentiable function taking and returning torch tensors.
    """
    def __init__(self, surrogate, points = None, epsilon=1e-9):
        self.surrogate, self.epsilon = surrogate, epsilon
        self.points = torch.empty((0, 1)) if points is None else points.detach().clone()
        # also store the initial points
        self.history = [self.points.detach().clone().requires_grad_(False)]
    
    # returns x range between which the method converges
    def converging_range(self, zero_point=torch.tensor([[0]])):
        def euclidean_dist(a, b):
            return torch.sqrt(torch.sum((a - b) ** 2, dim=1))
        min_index, max_index = self.points.shape[0], 0
        # find first that does converge
        for i in range(self.history.shape[1]):
            if euclidean_dist(self.history[0][i], zero_point) > euclidean_dist(self.history[1][i], zero_point):
                break
        min_index = i
        # find last that converges
        for i in range(min_index + 1, self.history.shape[1]):
            if euclidean_dist(self.history[0][i], zero_point) <= euclidean_dist(self.history[1][i], zero_point):
                break
        max_index = i - 1
        return self.history[0][min_index], self.history[0][max_index]

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
        # Find the sign of all gradient values
        sign_mask = ((y_grad > 0) * 2 - 1).type(y_grad.dtype)
        # set the sign of epsilon to match the value it is added to
        y_grad = y_grad + self.epsilon * sign_mask
        # n is dimensionality of the inputs
        n = self.points.shape[1]
        # multi dimensional newtons method using the moore penrose inverse of the gradient
        self.points = self.points - torch.mul(torch.div(1, n * y_grad), y)
        # save a copy of the new set of points
        self.history.append(self.points.detach().clone().requires_grad_(False))
    
    def take_n_steps(self, n):
        for _ in range(n):
            self.step()