import numpy as np
from util.function_preprocessing import get_grad, make_function


def bound_reach_silent(
    f_expr,
    vars,
    bottom_left_vertex,
    box_length,
    box_numbers,
    *,
    M2,
    M3,
    return_debug=False,
):
    """
    Optimized silent subdivision algorithm for reach lower bound.

    Differences vs algo.py:
    - No Box objects
    - No printing
    - No plotting
    - No vertex generation
    - Minimal NumPy allocations
    """

    f_num = make_function(f_expr, vars)
    grad_f = get_grad(f_expr, vars)

    N = len(vars)
    bottom_left_vertex = np.asarray(bottom_left_vertex, dtype=float)

    # --------------------------------------------------
    # Build initial stack of boxes
    # Each box is represented as (origin_vector, sidelength)
    # --------------------------------------------------

    stack = []

    for idx in np.ndindex(*box_numbers):
        origin = bottom_left_vertex + np.array(idx, dtype=float) * box_length
        stack.append((origin, float(box_length)))

    case1_eps = np.inf
    case2_eps = np.inf

    steps = 0

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------

    while stack:
        steps += 1

        origin, eps = stack.pop()

        # midpoint (no new arrays if possible)
        m = origin + eps * 0.5

        # evaluate
        f_abs = abs(float(f_num(m)))

        g = grad_f(m)
        # manual L1 norm (faster than np.linalg.norm)
        grad1 = 0.0
        for i in range(N):
            grad1 += abs(float(g[i]))

        # --------------------------------------------------
        # Case 1
        # --------------------------------------------------
        if f_abs > np.sqrt(N) * eps * M2:
            if eps < case1_eps:
                case1_eps = eps
            continue

        # --------------------------------------------------
        # Case 2
        # --------------------------------------------------
        if grad1 > (N ** (3 / 2)) * eps * M3:
            if eps < case2_eps:
                case2_eps = eps
            continue

        # --------------------------------------------------
        # Subdivide
        # --------------------------------------------------
        half = eps * 0.5

        # generate 2^N children
        for corner in np.ndindex(*(2,) * N):
            new_origin = origin + np.array(corner, dtype=float) * half
            stack.append((new_origin, half))

    # --------------------------------------------------
    # Certified bound
    # --------------------------------------------------

    eps_union = min(case1_eps, case2_eps)

    C1 = M3
    C2 = C1 * (N ** (3 / 2)) * (eps_union / 2.0)

    tau_lb = C2 / (np.sqrt(N) * C1) if np.isfinite(C2) else 0.0

    if not return_debug:
        return float(tau_lb)

    return float(tau_lb), {
        "eps_union": eps_union,
        "eps_case1": case1_eps,
        "eps_case2": case2_eps,
        "steps": steps,
    }
