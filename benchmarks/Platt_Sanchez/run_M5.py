import os, sys, time
import numpy as np
from sympy import symbols

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from util.function_bounding import get_max_grad, get_max_Hess
from benchmarks.Platt_Sanchez.reach_algo_silent import bound_reach_silent


def main():
    x0, x1 = symbols("x0 x1")
    vars = [x0, x1]

    P30 = sum(x0**(2*k) * x1**(30 - 2*k) for k in range(16))
    f_expr = x0**2 + x1**2 - 1 + 1e-5 * P30

    L = 2.0
    bottom_left = np.array([-L, -L], dtype=float)
    box_length = 2 * L
    box_numbers = [1, 1]
    bounds = [[-L, -L], [L, L]]

    M2 = float(get_max_grad(f_expr, vars, bounds))
    M3 = float(get_max_Hess(f_expr, vars, bounds))

    t0 = time.perf_counter()

    tau_lb, info = bound_reach_silent(
        f_expr, vars,
        bottom_left, box_length, box_numbers,
        use_local_bounds=False,
        M2=M2, M3=M3,
        return_debug=True
    )

    t1 = time.perf_counter()

    print("\n=== Platt_Sanchez M5 (silent algo) ===")
    print(f"tau_lower_bound = {tau_lb}")
    print(f"eps_union = {info['eps_union']}, steps = {info['steps']}")
    print(f"wallclock_seconds = {(t1 - t0):.6f}")


if __name__ == "__main__":
    main()
