import numpy as np
import matplotlib.pyplot as plt
from util.box import Box
from util.function_preprocessing import get_grad, get_Hess, make_function
from sympy import symbols
from util.function_bounding import get_max_grad, get_max_Hess
import datetime


def get_smallest_box_length(boxes):
    return min(np.min(box.side_lengths) for box in boxes)


def algo(f_expr, vars, grad_f, Hess_f, bottom_left_vertex, box_length, box_numbers, M2, M3, use_local_bounds=False, visualise=False, compute_boxwise_min=False):
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
    :param compute_boxwise_min:
    :return: get_smallest_box_length(CaseOneBoxes), get_smallest_box_length(CaseTwoBoxes), min_grad_bound, min_hess_bound, boxwise_min (<- this is inf if compute_boxwise_min=False)
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

    min_grad_bound = np.inf
    min_hess_bound = np.inf

    boxwise_min = np.inf
    minpoint = None

    while NewBoxes:
        if steps % 100 == 0:
            print(f'{datetime.datetime.now()}. Steps: {steps}')
        steps += 1
        CurrentBox = NewBoxes.pop()
        m = CurrentBox.midpoint()
        epsilon = np.min(CurrentBox.side_lengths)

        if use_local_bounds:
            vertices = CurrentBox.vertices()
            box_bounds = [(np.min(vertices[:, i]), np.max(vertices[:, i])) for i in range(N)]
            box_max_grad = get_max_grad(f_expr, symbols(' '.join(f'x{i}' for i in range(N))), box_bounds)
            min_grad_bound = min(box_max_grad, min_grad_bound)
            box_max_Hess = get_max_Hess(f_expr, symbols(' '.join(f'x{i}' for i in range(N))), box_bounds)
            min_hess_bound = min(box_max_Hess, min_hess_bound)
        else:
            box_max_grad = M2
            box_max_Hess = M3

        grad_norm = np.linalg.norm(grad_f(m), ord=1)
        CurrentBox.grad_one_norm_at_midpoint = grad_norm
        f_abs = abs(f(m))

        if f_abs > np.sqrt(N) * epsilon * box_max_grad:
            CaseOneBoxes.append(CurrentBox)
        elif grad_norm > N ** (3 / 2) * epsilon * box_max_Hess:
            CurrentBox.grad_one_norm_at_midpoint = grad_norm
            CaseTwoBoxes.append(CurrentBox)
            if compute_boxwise_min:
                if box_max_Hess * len(vars) ** (3 / 2) * epsilon / 2 < boxwise_min:
                    minpoint = m
                    # print(f'until now boxwise_min: {boxwise_min}, minpoint: {minpoint}, box_max_Hess: {box_max_Hess}, epsilon: {epsilon}')
                boxwise_min = min(boxwise_min, box_max_Hess * len(vars) ** (3 / 2) * epsilon / 2)  #
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

    print(f'Boxwise min attained at box containing minpoint={minpoint}')
    print(f'steps: {steps}')
    return get_smallest_box_length(CaseOneBoxes), get_smallest_box_length(CaseTwoBoxes), min_grad_bound, min_hess_bound, boxwise_min


if __name__ == '__main__':
    x0, x1 = symbols('x0 x1')
    vars = [x0, x1]
    f_expr = x0**2 + x1**2 - 1

    grad_f = get_grad(f_expr, vars)
    Hess_f = get_Hess(f_expr, vars)
    bounds = [[-2, -2], [2, 2]]
    max_grad_norm = get_max_grad(f_expr, vars, bounds)
    max_Hess_norm = get_max_Hess(f_expr, vars, bounds)

    smallest_case_one, smallest_case_two, _, _, _ = algo(f_expr, vars, grad_f, Hess_f, [-2, -2], 4, [1, 1], max_grad_norm, max_Hess_norm, use_local_bounds=False, visualise=False)
    print(f'Smallest case one box: {smallest_case_one}; smallest case two: {smallest_case_two}')
    lower_bound = max_Hess_norm*len(vars)**(3/2)*smallest_case_two/2
    print(f"Lower bound for |grad_f|_1:", lower_bound)
