from algo.algo import algo
import numpy as np
from util.box import Box
from util.function_preprocessing import get_grad, get_Hess, make_function
from sympy import symbols, Matrix, diff
from util.function_bounding import get_max_grad, get_max_Hess
import datetime


if __name__ == '__main__':
    x0, x1, x2 = symbols('x0 x1 x2')
    vars = [x0, x1, x2]
    f_expr = x0**2 + x1**2 + x2**2 - 1

    grad_f = get_grad(f_expr, vars)
    Hess_f = get_Hess(f_expr, vars)
    bounds = [[-2, 2], [-2, 2], [-2, 2]]
    max_grad_norm = get_max_grad(f_expr, vars, bounds)
    max_Hess_norm = get_max_Hess(f_expr, vars, bounds)

    lower_bound = algo(f_expr, vars, grad_f, Hess_f, [-2, -2, -2], 4, [1, 1, 1], max_grad_norm, max_Hess_norm)
    print("Lower bound for |grad_f|_1:", lower_bound)
