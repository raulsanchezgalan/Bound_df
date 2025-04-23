from algo.algo import algo
import numpy as np
from util.box import Box
from util.function_preprocessing import get_grad, get_Hess, make_function
from sympy import symbols, Matrix, diff
from util.function_bounding import get_max_grad, get_max_Hess
import datetime


if __name__ == '__main__':
    x0, x1 = symbols('x0 x1')
    vars = [x0, x1]
    f_expr = ((x0 ** 3 - x0 * x1 ** 2 + x1 + 1) ** 2 * (x0 ** 2 + x1 ** 2 - 1) + x1 ** 2 - 5)

    grad_f = get_grad(f_expr, vars)
    Hess_f = get_Hess(f_expr, vars)
    bounds = [[-3, 2], [-3, 2]]
    max_grad_norm = get_max_grad(f_expr, vars, bounds)
    max_Hess_norm = get_max_Hess(f_expr, vars, bounds)
    print(max_grad_norm,max_Hess_norm)

    smallest_case_one, smallest_case_two, smallest_grad_bd, smallest_hess_bd, boxwise_min = algo(f_expr, vars, grad_f, Hess_f, [-2, -2], 4, [1, 1], max_grad_norm,
                                                max_Hess_norm, use_local_bounds=True, visualise=False, compute_boxwise_min=True)
    print(f'Smallest case one box: {smallest_case_one}; smallest case two: {smallest_case_two}')
    print(f'smallest hess: {smallest_hess_bd}')
    lower_bound = smallest_hess_bd * len(vars) ** (3 / 2) * smallest_case_two / 2
    print(f"Lower bound for |grad_f|_1 WITHOUT boxwise computation:", lower_bound)
    print(f"Lower bound for |grad_f|_1 WITH boxwise computation:", boxwise_min)
