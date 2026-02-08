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

    def val_gradient_at(self, x):
        assert x.shape[1] == self.points.shape[1]
        # df/df = 1, we want the plain gradient of the inputs
        x.requires_grad_()
        means, variances = self.surrogate.mean_and_variance(x)
        means.backward(torch.ones(means.shape), inputs=x)
        return means.detach(), variances, x.grad
    
    def step(self):
        # get y values and gradients at all points
        y, _, y_grad = self.val_gradient_at(self.points)
        # Find the sign of all gradient values
        sign_mask = ((y_grad > 0) * 2 - 1).type(y_grad.dtype)
        # set the sign of epsilon to match the value it is added to
        y_grad = y_grad + self.epsilon * sign_mask
        # n is dimensionality of the inputs
        n = self.points.shape[1]
        # multi-dimensional newtons method using the moore penrose inverse of the gradient
        self.points = self.points - torch.mul(torch.div(1, n * y_grad), y)
    
    def take_n_steps(self, n):
        for _ in range(n):
            self.step()
    
    # by default we use the 95% confidence interval (1.96 standard deviations from the mean)
    def noise_vector(self, points = None, stds = 1.96):
        # by default use the held points
        if points is None:
            points = self.points.detach().clone()
        # we assume the mean value to be 0 here
        _, variances, grad = self.val_gradient_at(points)
        # divide by |grad|^2, as the rate of change of the mean affects change in edge likelihood
        # we assume the variance to remain constant here
        magnitude = stds * torch.sqrt(variances) / (self.epsilon + torch.sum(grad ** 2, dim=1, keepdim=True))
        return grad * magnitude
    
    def noise_vector_normalised_grad(self, points = None, stds=1.96):
        # by default use the held points
        if points is None:
            points = self.points.detach().clone()
        # we assume the mean value to be 0 here too
        _, variances, grad = self.val_gradient_at(points)
        # divide by |grad|, so we assume both the mean and variance to remain constant when computing the noise magnitude
        magnitude = stds * torch.sqrt(variances) / (self.epsilon + torch.sqrt(torch.sum(grad ** 2, dim=1, keepdim=True)))
        return grad * magnitude