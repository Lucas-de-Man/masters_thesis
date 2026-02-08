import torch
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize

class Surrogate:
    """

    """
    # expect y_points to be n x (1 + n_constraints)
    def __init__(self, x_points, y_points, normalize_inputs=False):
        self.normilisation = Normalize(d=x_points.shape[-1]) if normalize_inputs else None
        self.n_constraints = y_points.shape[1] - 1
        self.x_points, self.y_points = x_points, y_points
        self.model = None
        self.model, self.mll = self.update_model()
        fit_gpytorch_mll(self.mll)
        
    def update_model(self):
        assert self.x_points.shape[0] == self.y_points.shape[0]
        y_var = torch.ones((self.y_points.shape[0], 1)) * 0.0001
        models = []
        for i in range(self.y_points.shape[1]):
            models.append(SingleTaskGP(
                self.x_points,
                torch.unsqueeze(self.y_points[:, i], -1),
                y_var.to(self.x_points),
                input_transform=self.normilisation,
            ).to(self.x_points))
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        if self.model is not None:
            model.load_state_dict(self.model.state_dict())
        return model, mll

    def add_observations(self, x, y):
        self.x_points = torch.cat((self.x_points, x), dim=0)
        self.y_points = torch.cat((self.y_points, y), dim=0)
        print(self.y_points)
        self.model, self.mll = self.update_model()
        fit_gpytorch_mll(self.mll)

    def mean_and_variance(self, x):
        posterior = self.model.posterior(x)
        return posterior.mean, posterior.variance

    def __call__(self, x):
        return self.model.posterior(x).mean

x, y = torch.tensor([[0.], [1.]], dtype=torch.double), torch.tensor([[0., 0.], [1., 0.1]], dtype=torch.double)
test = Surrogate(x, y)
add_x, add_y = torch.tensor([[0.2], [0.8]], dtype=torch.double), torch.tensor([[0.2, 0.1], [0.8, 0.5]], dtype=torch.double)
test.add_observations(add_x, add_y)
print(test(torch.tensor([[0.2]], dtype=torch.double)))