import numpy as np
from scipy.optimize import minimize

try:
    from .function_preprocessing import get_grad, get_Hess
except ImportError:
    from function_preprocessing import get_grad, get_Hess


def get_max_grad(f_expr, vars_, bounds):
    """Return the maximum of the 2-norm of the gradient on a rectangular domain."""
    grad_func = get_grad(f_expr, vars_)

    def norm_grad(x):
        grad = grad_func(x)
        return -np.linalg.norm(grad)

    x0 = [(b[0] + b[1]) / 2 for b in bounds]
    result = minimize(norm_grad, x0, bounds=bounds, method="L-BFGS-B")
    return -result.fun


def get_max_Hess(f_expr, vars_, bounds):
    """Return the maximum of the 2-norm of the Hessian on a rectangular domain."""
    Hess_func = get_Hess(f_expr, vars_)

    def norm_Hess(x):
        Hess = Hess_func(x)
        return -np.linalg.norm(Hess, ord=2)

    x0 = [(b[0] + b[1]) / 2 for b in bounds]
    result = minimize(norm_Hess, x0, bounds=bounds, method="L-BFGS-B")
    return -result.fun
