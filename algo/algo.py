import numpy as np
import matplotlib.pyplot as plt
from util.box import Box
from util.function_preprocessing import get_grad, get_Hess, make_function
from sympy import symbols
from util.function_bounding import get_max_grad, get_max_Hess
import datetime


def get_smallest_box_length(boxes):
    return min(np.min(box.side_lengths) for box in boxes)


def algo(f_expr, vars, grad_f, Hess_f, bottom_left_vertex, box_length, box_numbers, M2, M3, use_local_bounds=True, visualise=False):
    '''
    An algorithm computing a lower bound for |grad_f|_1 on a subset of n-dimensional
    space defined by bottom_left_vertex, box_length, box_numbers.

    :param f_expr: The function f defining the variety M=Z(f) in n-dimensional space
    as a sympy polynomial
    :param vars: A list of variables x0, x1, ...
    :param grad_f: Gradient of f
    :param Hess_f: Hessian of f
    :param bottom_left_vertex: A point in n-dim space marking the bottom-left vertex
    of a large box on which a bound for grad_f is sought
    :param box_length: Length of initial boxes which cover the large box
    :param box_numbers: Number of initial boxes in each direction
    :param M2, M3: Constants defining algorithm requirements (these variables are not
    accessed if use_local_bounds=True)
    :param use_local_bounds: If true, use local bounds for grad and Hess
    :param visualise: If true, save png image after every step (only works in dimension 2)
    :return: A lower bound for |grad_f|_1 on M.
    '''

    f = make_function(f_expr, vars)

    N = len(bottom_left_vertex)

    initial_boxes = []
    for indices in np.ndindex(*box_numbers):
        vertex = bottom_left_vertex + np.array(indices) * box_length
        initial_boxes.append(Box(vertex, np.full(N, box_length)))

    NewBoxes = initial_boxes.copy()
    CaseOneBoxes = []
    CaseTwoBoxes = []

    steps = 0

    while NewBoxes:
        if steps % 100 == 0:
            print(f'{datetime.datetime.now()}. Steps: {steps}')
        steps += 1
        CurrentBox = NewBoxes.pop()
        m = CurrentBox.midpoint()
        epsilon = np.linalg.norm(CurrentBox.side_lengths)

        if use_local_bounds:
            vertices = CurrentBox.vertices()
            box_bounds = [(np.min(vertices[:, i]), np.max(vertices[:, i])) for i in range(N)]
            box_max_grad = get_max_grad(f_expr, symbols(' '.join(f'x{i}' for i in range(N))), box_bounds)
            box_max_Hess = get_max_Hess(f_expr, symbols(' '.join(f'x{i}' for i in range(N))), box_bounds)
        else:
            box_max_grad = M2
            box_max_Hess = M3

        grad_norm = np.linalg.norm(grad_f(m), ord=2)
        f_abs = abs(f(m))

        if f_abs > np.sqrt(N) * epsilon * box_max_grad:
            CaseOneBoxes.append(CurrentBox)
        elif grad_norm > N ** (3 / 2) * epsilon * box_max_Hess:
            CaseTwoBoxes.append(CurrentBox)
        else:
            NewBoxes.extend(CurrentBox.subdivide())

        if visualise and N == 2:
            fig, ax = plt.subplots(figsize=(8, 8))
            x = np.linspace(bottom_left_vertex[0], bottom_left_vertex[0] + box_length * box_numbers[0], 500)
            y = np.linspace(bottom_left_vertex[1], bottom_left_vertex[1] + box_length * box_numbers[1], 500)
            X, Y = np.meshgrid(x, y)
            Z = f([X, Y])
            ax.contour(X, Y, Z, levels=[0], colors='blue')

            for box, color in [(NewBoxes, 'gray'), (CaseOneBoxes, 'green'), (CaseTwoBoxes, 'red')]:
                for b in box:
                    bl = b.bottom_left_vertex
                    sl = b.side_lengths
                    rect = plt.Rectangle(bl, sl[0], sl[1], linewidth=1, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)

            ax.set_xlim(bottom_left_vertex[0], bottom_left_vertex[0] + box_length * box_numbers[0])
            ax.set_ylim(bottom_left_vertex[1], bottom_left_vertex[1] + box_length * box_numbers[1])
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f'Iteration {steps}')
            plt.savefig(f'algo_step_{steps:04}.png')
            plt.close(fig)

    min_grad_norm = np.inf
    for box in CaseTwoBoxes:
        m = box.midpoint()
        grad_norm = np.linalg.norm(grad_f(m), ord=1)
        if grad_norm < min_grad_norm:
            min_grad_norm = grad_norm

    print(f'Smallest CaseOneBoxes box length: {get_smallest_box_length(CaseOneBoxes)}')
    print(f'Smallest CaseTwoBoxes box length: {get_smallest_box_length(CaseTwoBoxes)}')

    return min_grad_norm


if __name__ == '__main__':
    x0, x1 = symbols('x0 x1')
    vars = [x0, x1]
    f_expr = x0**2 + x1**2 - 1

    grad_f = get_grad(f_expr, vars)
    Hess_f = get_Hess(f_expr, vars)
    bounds = [[-2, -2], [2, 2]]
    max_grad_norm = get_max_grad(f_expr, vars, bounds)
    max_Hess_norm = get_max_Hess(f_expr, vars, bounds)

    lower_bound = algo(f_expr, vars, grad_f, Hess_f, [-2, -2], 4, [1, 1], max_grad_norm, max_Hess_norm)
    print("Lower bound for |grad_f|_1:", lower_bound) # gives 1.75
