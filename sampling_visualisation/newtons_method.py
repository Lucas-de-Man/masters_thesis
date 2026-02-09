import torch

class NewtonsMethod:
    """
        Class that holds a set of points and a surrogate.
        It's able to aply newton's method to those points on the surrogate.
        The surrogate just needs to be a differentiable function taking and returning torch tensors.
    """
    def __init__(self, surrogate, points = None, epsilon=1e-6, lim=(0, 1)):
        self.surrogate, self.epsilon, self.lim = surrogate, epsilon, lim
        self.points = torch.empty((0, 1)) if points is None else points.detach().clone()

    def val_gradient_at(self, x):
        assert x.shape[1] == self.points.shape[1]
        # df/df = 1, we want the plain gradient of the inputs
        x.requires_grad_()
        posterior = self.surrogate.posterior(x)
        means = posterior.mean
        means.backward(torch.ones(means.shape), inputs=x)
        return means.detach(), posterior.variance, x.grad
    
    def step(self):
        # get y values and gradients at all points
        y, _, y_grad = self.val_gradient_at(self.points)
        # compute magnitudes of gradients, adding epsilon to not divide by zero 
        y_grad_mags = torch.sum(y_grad ** 2, dim=1, keepdim=True) + self.epsilon
        # multi-dimensional newtons method using the moore penrose inverse of the gradient (v^-1 = 1/|v|^2 * v^T)
        self.points = self.points - torch.mul(torch.div(1, y_grad_mags), y_grad) * y
    
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
    
    def get_edge_points(self, increment=0.01, scale_increment=False):
        # remove points wiht a nan value
        numaric_points = self.points[~torch.any(self.points.isnan(), dim=1)]
        # select points that are inside the legal hypercube
        valid_points = numaric_points[~torch.any(torch.logical_or(numaric_points < self.lim[0], numaric_points > self.lim[1]), dim=1)]
        # no valid points means we can end here
        if valid_points.shape[0] == 0:
            return valid_points
        # distance from 0
        values = self.surrogate.posterior(valid_points).mean ** 2
        # scale increment with a common value (median) if needed
        if scale_increment:
            increment *= torch.median(values)
        threshold = increment
        n_valid = torch.sum((values < threshold).type(torch.int32))
        # rounded up point where we exeed 5% of the total points
        five_percent = values.shape[0] // 20 + int((values.shape[0] % 20) > 0)
        while n_valid <= five_percent:
            threshold += increment
            n_valid = torch.sum((values <= threshold).type(torch.int32))
        return valid_points[(values < threshold).squeeze()]