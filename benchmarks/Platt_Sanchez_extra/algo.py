import datetime

import numpy as np

try:
    from .box import Box
    from .function_bounding import get_max_grad, get_max_Hess
    from .function_preprocessing import make_function
except ImportError:
    from box import Box
    from function_bounding import get_max_grad, get_max_Hess
    from function_preprocessing import make_function


def get_smallest_box_length(boxes):
    if not boxes:
        return None
    return min(float(np.min(box.side_lengths)) for box in boxes)


def algo(
    f_expr,
    vars_,
    grad_f,
    bottom_left_vertex,
    box_length,
    box_numbers,
    M2,
    M3,
    use_local_bounds=False,
    visualise=False,
    compute_boxwise_min=False,
    verbose=0,
):
    """
    An algorithm computing a lower bound for |grad_f|_1 on a subset of
    n-dimensional space defined by bottom_left_vertex, box_length, and
    box_numbers.

    The region is covered by initial cubes whose common side length is
    box_length. Each cube is classified as either:
    - a CaseOne box, meaning the midpoint test certifies that f does not
      vanish on the box; or
    - a CaseTwo box, meaning the midpoint gradient test certifies a lower
      bound for |grad_f|_1 on the whole box.

    Boxes that satisfy neither test are subdivided until every remaining box
    falls into one of these two cases.

    :param f_expr: The function f defining the variety M = Z(f), given as a
        SymPy expression.
    :param vars_: A list of variables x0, x1, ... used in f_expr.
    :param grad_f: A callable returning the gradient of f at a point.
    :param bottom_left_vertex: The bottom-left vertex of the large box to be
        covered by the initial cubes.
    :param box_length: The side length of each initial cube.
    :param box_numbers: The number of initial cubes in each coordinate
        direction.
    :param M2: An upper bound for |grad_f|_2 on the whole region. This is only
        used when use_local_bounds=False.
    :param M3: An upper bound for |Hess f|_2 on the whole region. This is only
        used when use_local_bounds=False.
    :param use_local_bounds: If True, replace the global constants M2 and M3 by
        boxwise bounds computed on each current box.
    :param visualise: If True, save a PNG image after every step. This is only
        implemented for the two-dimensional case.
    :param compute_boxwise_min: If True, compute the best boxwise lower bound
        for |grad_f|_1 coming from the accepted CaseTwo boxes. If False, the
        corresponding return value is None.
    :param verbose: Integer verbosity flag. No progress output is printed when
        verbose=0. Any positive value enables the existing progress print every
        100 steps.
    :return: A dictionary with the following keys:
        - "smallest_case_one_box_length": smallest side length among accepted
          CaseOne boxes, or None if there are no CaseOne boxes.
        - "smallest_case_two_box_length": smallest side length among accepted
          CaseTwo boxes, or None if there are no CaseTwo boxes.
        - "min_local_grad_bound": smallest local gradient upper bound that was
          encountered when use_local_bounds=True, or None otherwise.
        - "min_local_hess_bound": smallest local Hessian upper bound that was
          encountered when use_local_bounds=True, or None otherwise.
        - "boxwise_grad_lower_bound": best lower bound for |grad_f|_1 obtained
          from the accepted CaseTwo boxes when compute_boxwise_min=True, or
          None otherwise.
        - "case_one_box_count": number of accepted CaseOne boxes.
        - "case_two_box_count": number of accepted CaseTwo boxes.
        - "steps": total number of subdivision steps performed.
    """
    f = make_function(f_expr, vars_)

    bottom_left_vertex = np.array(bottom_left_vertex, dtype=float)
    N = len(vars_)

    initial_boxes = []
    for indices in np.ndindex(*box_numbers):
        vertex = bottom_left_vertex + np.array(indices, dtype=float) * box_length
        initial_boxes.append(Box(vertex, np.full(N, box_length, dtype=float)))

    NewBoxes = initial_boxes.copy()
    CaseOneBoxes = []
    CaseTwoBoxes = []

    steps = 0
    min_grad_bound = np.inf
    min_hess_bound = np.inf
    boxwise_min = np.inf

    while NewBoxes:
        if verbose > 0 and steps % 100 == 0:
            print(f"{datetime.datetime.now()}. Steps: {steps}")

        steps += 1
        CurrentBox = NewBoxes.pop()
        m = CurrentBox.midpoint()
        epsilon = float(np.min(CurrentBox.side_lengths))

        if use_local_bounds:
            vertices = CurrentBox.vertices()
            box_bounds = [
                (float(np.min(vertices[:, i])), float(np.max(vertices[:, i])))
                for i in range(N)
            ]
            box_max_grad = get_max_grad(f_expr, vars_, box_bounds)
            min_grad_bound = min(box_max_grad, min_grad_bound)
            box_max_Hess = get_max_Hess(f_expr, vars_, box_bounds)
            min_hess_bound = min(box_max_Hess, min_hess_bound)
        else:
            box_max_grad = M2
            box_max_Hess = M3

        grad_norm = float(np.linalg.norm(grad_f(m), ord=1))
        f_abs = abs(float(f(m)))

        if f_abs > np.sqrt(N) * epsilon * box_max_grad:
            CaseOneBoxes.append(CurrentBox)
        elif grad_norm > N ** (3 / 2) * epsilon * box_max_Hess:
            CaseTwoBoxes.append(CurrentBox)
            if compute_boxwise_min:
                boxwise_min = min(
                    boxwise_min,
                    box_max_Hess * N ** (3 / 2) * epsilon / 2,
                )
        else:
            NewBoxes.extend(CurrentBox.subdivide())

        if visualise and N == 2:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))
            x = np.linspace(
                bottom_left_vertex[0],
                bottom_left_vertex[0] + box_length * box_numbers[0],
                500,
            )
            y = np.linspace(
                bottom_left_vertex[1],
                bottom_left_vertex[1] + box_length * box_numbers[1],
                500,
            )
            X, Y = np.meshgrid(x, y)
            Z = f([X, Y])
            ax.contour(X, Y, Z, levels=[0], colors="blue")

            for box_group, color in [
                (NewBoxes, "gray"),
                (CaseOneBoxes, "green"),
                (CaseTwoBoxes, "red"),
            ]:
                for box in box_group:
                    bl = box.bottom_left_vertex
                    sl = box.side_lengths
                    rect = plt.Rectangle(
                        bl,
                        sl[0],
                        sl[1],
                        linewidth=1,
                        edgecolor=color,
                        facecolor="none",
                    )
                    ax.add_patch(rect)

            ax.set_xlim(
                bottom_left_vertex[0],
                bottom_left_vertex[0] + box_length * box_numbers[0],
            )
            ax.set_ylim(
                bottom_left_vertex[1],
                bottom_left_vertex[1] + box_length * box_numbers[1],
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Iteration {steps}")
            plt.savefig(f"algo_step_{steps:04}.png")
            plt.close(fig)

    if verbose > 0:
        print(f"steps: {steps}")

    return {
        "smallest_case_one_box_length": get_smallest_box_length(CaseOneBoxes),
        "smallest_case_two_box_length": get_smallest_box_length(CaseTwoBoxes),
        "min_local_grad_bound": None if not np.isfinite(min_grad_bound) else float(min_grad_bound),
        "min_local_hess_bound": None if not np.isfinite(min_hess_bound) else float(min_hess_bound),
        "boxwise_grad_lower_bound": None if not np.isfinite(boxwise_min) else float(boxwise_min),
        "case_one_box_count": len(CaseOneBoxes),
        "case_two_box_count": len(CaseTwoBoxes),
        "steps": steps,
    }
