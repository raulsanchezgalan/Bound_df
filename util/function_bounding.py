from scipy.optimize import minimize
from util.function_preprocessing import get_grad, get_Hess
import numpy as np
from sympy import symbols, Matrix, diff, lambdify


def get_max_grad(f_expr, vars, bounds):
    """Returns the maximum of the 2-norm of the gradient of f over a rectangular domain."""
    grad_func = get_grad(f_expr, vars)  # Get the gradient function

    def norm_grad(x):
        grad = grad_func(x)  # Compute gradient
        return -np.linalg.norm(grad)  # Minimize negative norm to maximize

    # Initial guess: midpoint of bounds
    x0 = [(b[0] + b[1]) / 2 for b in bounds]

    # Optimize to find max gradient norm
    result = minimize(norm_grad, x0, bounds=bounds, method='L-BFGS-B')

    return -result.fun  # Return the maximum norm


def get_max_Hess(f_expr, vars, bounds):
    """Returns the maximum of the 2-norm of the Hessian of f over a rectangular domain."""
    Hess_func = get_Hess(f_expr, vars)  # Get the Hessian function

    def norm_Hess(x):
        Hess = Hess_func(x)  # Compute Hessian
        return -np.linalg.norm(Hess, ord=2)  # Minimize negative 2-norm

    # Initial guess: midpoint of bounds
    x0 = [(b[0] + b[1]) / 2 for b in bounds]

    # Optimize to find max Hessian norm
    result = minimize(norm_Hess, x0, bounds=bounds, method='L-BFGS-B')

    return -result.fun  # Return the maximum norm


# Example usage
bounds = [(0, 2), (0, 2)]  # Define the rectangular domain

x0, x1, x2 = symbols('x0 x1 x2')
f_expr = ((x0 ** 3 - x0 * x1 ** 2 + x1 + 1) ** 2 * (x0 ** 2 + x1 ** 2 - 1) + x1 ** 2 - 5)

max_grad_norm = get_max_grad(f_expr, [x0, x1], bounds)
max_Hess_norm = get_max_Hess(f_expr, [x0, x1], bounds)

print(max_grad_norm)  # Output: 1011.6557105230127
print(max_Hess_norm)  # Output: 3460.393166917154
