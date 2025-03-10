import numpy as np
from sympy import symbols, Matrix, diff, lambdify

def get_grad(f, vars):
    """Returns the gradient function of f."""
    grad_expr = Matrix([diff(f, v) for v in vars])  # Compute gradient
    grad_func = lambdify(vars, grad_expr, 'numpy')  # Convert to numpy function
    return lambda x: np.array(grad_func(*x), dtype=np.float64).flatten()

def get_Hess(f, vars):
    """Returns the Hessian function of f."""
    grad_expr = Matrix([diff(f, v) for v in vars])  # Compute gradient
    Hess_expr = grad_expr.jacobian(vars)  # Compute Hessian
    Hess_func = lambdify(vars, Hess_expr, 'numpy')  # Convert to numpy function
    return lambda x: np.array(Hess_func(*x), dtype=np.float64)

def make_function(f_expr, vars):
    """Returns a lambda function for the given symbolic expression."""
    func = lambdify(vars, f_expr, 'numpy')  # Convert to numpy-compatible function
    return lambda x: func(*x)

# Example usage
x0, x1 = symbols('x0 x1')
f_expr = ((x0**3 - x0*x1**2 + x1 + 1)**2 * (x0**2 + x1**2 - 1) + x1**2 - 5)

grad_f_func = get_grad(f_expr, [x0, x1])
Hess_f_func = get_Hess(f_expr, [x0, x1])

# Test at a sample point
test_point = [1.0, 2.0]
print(grad_f_func(test_point))  # Output: [0. 4.]
print(Hess_f_func(test_point))  # Output: [[ 8. 24.]
                                #          [24. 74.]]
