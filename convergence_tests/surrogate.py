import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize

class Surrogate:
    """
        Simple container class for a SingleTaskGP.
        We just needit to interpolate over a set number of points, with the abilaty to choose a lenghtscale.
    """
    def __init__(self, x_points, y_points, set_ls=None, normalize_inputs=False):
        normilisation = Normalize(d=x_points.shape[-1]) if normalize_inputs else None
        y_var=torch.full_like(y_points, 1e-6)
        self.gp_model = SingleTaskGP(x_points, y_points, y_var, input_transform=normilisation)
        if set_ls is not None:
            self.gp_model.covar_module._set_lengthscale(set_ls)
    
    def __call__(self, x):
        return self.gp_model.posterior(x).mean