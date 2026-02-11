import os
import sys
import time
import numpy as np
from sympy import symbols


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from util.function_bounding import get_max_grad, get_max_Hess
from benchmarks.Platt_Sanchez.reach_algo_silent import bound_reach_silent


def run_example(name, f_expr, vars, L):
    bottom_left = np.array([-L] * len(vars), dtype=float)
    box_length = 2 * L
    box_numbers = [1] * len(vars)
    bounds = [[-L] * len(vars), [L] * len(vars)]

    M2 = float(get_max_grad(f_expr, vars, bounds))
    M3 = float(get_max_Hess(f_expr, vars, bounds))

    t0 = time.perf_counter()

    tau_lb, info = bound_reach_silent(
        f_expr, vars,
        bottom_left, box_length, box_numbers,
        use_local_bounds=False,
        M2=M2,
        M3=M3,
        return_debug=True
    )

    t1 = time.perf_counter()

    return {
        "name": name,
        "dim": len(vars),
        "tau_lb": tau_lb,
        "eps": info["eps_union"],
        "steps": info["steps"],
        "time": t1 - t0,
    }


def main():
    print("\nRunning Platt–Sanchez silent reach benchmarks\n")

    results = []

    # =====================
    # M1
    # =====================
    x0, x1 = symbols("x0 x1")
    f1 = x0**2 + x1**2 - 1
    results.append(run_example("M1", f1, [x0, x1], L=2.0))

    # =====================
    # M2
    # =====================
    f2 = (x0**3 - x0*x1**2 + x1 + 1)**2 * (x0**2 + x1**2 - 1) + x1**2 - 5
    results.append(run_example("M2", f2, [x0, x1], L=3.0))

    # =====================
    # M3
    # =====================
    f3 = x0**4 - x0**2*x1**2 + x1**4 - 4*x0**2 - 2*x1**2 - x0 - 4*x1 + 1
    results.append(run_example("M3", f3, [x0, x1], L=3.0))

    # =====================
    # M4 (3D)
    # =====================
    x2 = symbols("x2")
    f4 = (
        4*x0**2 + 7*x1**4 + 3*x2**4 - 3
        - 8*x0**3 + 2*x0**2*x1 - 4*x0**2
        - 8*x0*x1**2 - 5*x0*x1 + 8*x0
        - 6*x1**3 + 8*x1**2 + 4*x1
    )
    results.append(run_example("M4", f4, [x0, x1, x2], L=2.0))

    # =====================
    # M5
    # =====================
    P30 = sum(x0**(2*k) * x1**(30 - 2*k) for k in range(16))
    f5 = x0**2 + x1**2 - 1 + 1e-5 * P30
    results.append(run_example("M5", f5, [x0, x1], L=2.0))

    # =====================
    # M6
    # =====================
    f6 = (x0**2 + 2*x1**2 - 1) * (x0**2 + 2*x1**2 - 3 - 1e-5)
    results.append(run_example("M6", f6, [x0, x1], L=3.0))

    # =====================
    # Print summary table
    # =====================

    print("------------------------------------------------------------")
    print("{:<4} {:<3} {:<14} {:<12} {:<10} {:<10}".format(
        "ID", "N", "tau_lower_bound", "epsilon", "steps", "time(s)"
    ))
    print("------------------------------------------------------------")

    for r in results:
        print("{:<4} {:<3} {:<14.8e} {:<12.6e} {:<10d} {:<10.6f}".format(
            r["name"],
            r["dim"],
            r["tau_lb"],
            r["eps"],
            r["steps"],
            r["time"],
        ))

    print("------------------------------------------------------------")


if __name__ == "__main__":
    main()
