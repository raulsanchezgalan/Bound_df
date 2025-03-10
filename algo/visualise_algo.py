import numpy as np
import matplotlib.pyplot as plt
from util.box import Box
from util.function_preprocessing import get_grad, get_Hess, make_function
from sympy import symbols, Matrix, diff
from util.function_bounding import get_max_grad, get_max_Hess


def algo(f, grad_f, Hess_f, bottom_left_vertex, box_length, box_numbers, M2, M3):
    N = len(bottom_left_vertex)

    initial_boxes = []
    for indices in np.ndindex(*box_numbers):
        vertex = bottom_left_vertex + np.array(indices) * box_length
        initial_boxes.append(Box(vertex, np.full(N, box_length)))

    NewBoxes = initial_boxes.copy()
    CaseOneBoxes = []
    CaseTwoBoxes = []
    iteration = 0

    while NewBoxes:
        CurrentBox = NewBoxes.pop()
        m = CurrentBox.midpoint()
        epsilon = np.linalg.norm(CurrentBox.side_lengths)

        grad_norm = np.linalg.norm(grad_f(m), ord=2)
        f_abs = abs(f(m))
        hess_norm = np.linalg.norm(Hess_f(m), ord=2)

        if f_abs > np.sqrt(N) * epsilon * M2:
            CaseOneBoxes.append(CurrentBox)
        elif grad_norm > N ** (3/2) * epsilon * M3:
            CaseTwoBoxes.append(CurrentBox)
        else:
            NewBoxes.extend(CurrentBox.subdivide())

        # Visualization
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
        ax.set_title(f'Iteration {iteration}')
        plt.savefig(f'algo_step_{iteration:03}.png')
        plt.close(fig)

        iteration += 1

    min_grad_norm = np.inf
    for box in CaseTwoBoxes:
        m = box.midpoint()
        grad_norm = np.linalg.norm(grad_f(m), ord=1)
        if grad_norm < min_grad_norm:
            min_grad_norm = grad_norm

    return min_grad_norm

if __name__ == '__main__':
    f = lambda x: x[0]**2 + x[1]**2 - 1
    grad_f = lambda x: [2*x[0], 2*x[1]]
    Hess_f = lambda x: np.array([[2, 0], [0, 2]])

    bottom_left_vertex = [-2, -2]
    box_length = 4
    box_numbers = [1, 1]
    M2 = np.sqrt(8)
    M3 = np.linalg.norm(np.array([[2, 0], [0, 2]]), ord=2)

    lower_bound = algo(f, grad_f, Hess_f, bottom_left_vertex, box_length, box_numbers, M2, M3)
    print("Lower bound for |grad_f|_1:", lower_bound)

    # x0, x1 = symbols('x0 x1')
    # vars = [x0, x1]
    # f_expr = ((x0 ** 3 - x0 * x1 ** 2 + x1 + 1) ** 2 * (x0 ** 2 + x1 ** 2 - 1) + x1 ** 2 - 5)
    # grad_f = get_grad(f_expr, vars)
    # Hess_f = get_Hess(f_expr, vars)
    # bounds = [[-3,-3], [3,3]]
    # max_grad_norm = get_max_grad(f_expr, vars, bounds)
    # max_Hess_norm = get_max_Hess(f_expr, vars, bounds)
    #
    # lower_bound = algo(make_function(f_expr, vars), grad_f, Hess_f, [-3, -3], 6, [1,1], max_grad_norm, max_Hess_norm)
    # print("Lower bound for |grad_f|_1:", lower_bound)



