import numpy as np
from sympy import Matrix, diff, lambdify


def get_grad(f, vars_):
    """Return the gradient function of f."""
    grad_expr = Matrix([diff(f, v) for v in vars_])
    grad_func = lambdify(vars_, grad_expr, "numpy")
    return lambda x: np.array(grad_func(*x), dtype=np.float64).flatten()


def get_Hess(f, vars_):
    """Return the Hessian function of f."""
    grad_expr = Matrix([diff(f, v) for v in vars_])
    Hess_expr = grad_expr.jacobian(vars_)
    Hess_func = lambdify(vars_, Hess_expr, "numpy")
    return lambda x: np.array(Hess_func(*x), dtype=np.float64)


def make_function(f_expr, vars_):
    """Return a NumPy-compatible function for the given symbolic expression."""
    func = lambdify(vars_, f_expr, "numpy")
    return lambda x: func(*x)
