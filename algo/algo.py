import numpy as np
from util.box import Box

def algo(f, grad_f, Hess_f, bottom_left_vertex, box_length, box_numbers, M2, M3):
    '''
    An algorithm computing a lower bound for |grad_f|_1 on a subset of n-dimensional
    space defined by bottom_left_vertex, box_length, box_numbers.

    :param f: The function f defining the variety M=Z(f) in n-dimensional space
    :param grad_f: Gradient of f
    :param Hess_f: Hessian of f
    :param bottom_left_vertex: A point in n-dim space marking the bottom-left vertex
    of a large box on which a bound for grad_f is sought
    :param box_length: Length of initial boxes which cover the large box
    :param box_numbers: Number of initial boxes in each direction
    :param M2, M3: Constants defining algorithm requirements
    :return: A lower bound for |grad_f|_1 on M.
    '''
    N = len(bottom_left_vertex)

    initial_boxes = []
    for indices in np.ndindex(*box_numbers):
        vertex = bottom_left_vertex + np.array(indices) * box_length
        initial_boxes.append(Box(vertex, np.full(N, box_length)))

    NewBoxes = initial_boxes.copy()
    CaseOneBoxes = []
    CaseTwoBoxes = []

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

    min_grad_norm = np.inf
    for box in CaseTwoBoxes:
        m = box.midpoint()
        grad_norm = np.linalg.norm(grad_f(m), ord=1)
        if grad_norm < min_grad_norm:
            min_grad_norm = grad_norm

    return min_grad_norm


if __name__ == '__main__':
    f = lambda x: x[0]**2+x[1]**2-1
    grad_f = lambda x: [2*x[0], 2*x[1]]
    Hess_f = lambda x: np.array([[2,0], [0,2]])
    bottom_left_vertex = [-2, -2]
    box_length = 4
    box_numbers = [1, 1]
    M2 = np.sqrt(8)
    M3 = np.linalg.norm(np.array([[2,0], [0,2]]), ord=2)

    lower_bound = algo(f, grad_f, Hess_f, bottom_left_vertex, box_length, box_numbers, M2, M3)
    print("Lower bound for |grad_f|_1:", lower_bound)
