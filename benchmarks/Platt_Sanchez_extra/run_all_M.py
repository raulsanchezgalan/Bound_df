#!/usr/bin/env python3

import math
import time

import sympy as sp

try:
    from .algo import algo
    from .function_bounding import get_max_grad, get_max_Hess
    from .function_preprocessing import get_grad
except ImportError:
    from algo import algo
    from function_bounding import get_max_grad, get_max_Hess
    from function_preprocessing import get_grad


def _examples():
    x0, x1, x2 = sp.symbols("x0 x1 x2", real=True)

    return {
        "M1": {
            "f_expr": x0**2 + x1**2 - 1,
            "vars": [x0, x1],
            "bounds": [[-2.0, 2.0], [-2.0, 2.0]],
            "bottom_left_vertex": [-2.0, -2.0],
            "box_length": 4.0,
            "box_numbers": [1, 1],
        },
        "M2": {
            "f_expr": (x0**3 - x0 * x1**2 + x1 + 1) ** 2 * (x0**2 + x1**2 - 1) + x1**2 - 5,
            "vars": [x0, x1],
            "bounds": [[-3.0, 3.0], [-3.0, 3.0]],
            "bottom_left_vertex": [-3.0, -3.0],
            "box_length": 6.0,
            "box_numbers": [1, 1],
        },
        "M3": {
            "f_expr": x0**4 - x0**2 * x1**2 + x1**4 - 4 * x0**2 - 2 * x1**2 - x0 - 4 * x1 + 1,
            "vars": [x0, x1],
            "bounds": [[-3.0, 3.0], [-3.0, 3.0]],
            "bottom_left_vertex": [-3.0, -3.0],
            "box_length": 6.0,
            "box_numbers": [1, 1],
        },
        "M4": {
            "f_expr": (
                4 * x0**2
                + 7 * x1**4
                + 3 * x2**4
                - 3
                - 8 * x0**3
                + 2 * x0**2 * x1
                - 4 * x0**2
                - 8 * x0 * x1**2
                - 5 * x0 * x1
                + 8 * x0
                - 6 * x1**3
                + 8 * x1**2
                + 4 * x1
            ),
            "vars": [x0, x1, x2],
            "bounds": [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]],
            "bottom_left_vertex": [-2.0, -2.0, -2.0],
            "box_length": 4.0,
            "box_numbers": [1, 1, 1],
        },
        "M5": {
            "f_expr": x0**2 + x1**2 - 1 + sp.Rational(1, 100000) * sum(
                x0 ** (2 * k) * x1 ** (30 - 2 * k) for k in range(16)
            ),
            "vars": [x0, x1],
            "bounds": [[-2.0, 2.0], [-2.0, 2.0]],
            "bottom_left_vertex": [-2.0, -2.0],
            "box_length": 4.0,
            "box_numbers": [1, 1],
        },
        "M6": {
            "f_expr": (x0**2 + 2 * x1**2 - 1)
            * (((x0 - (2 + sp.Rational(1, 100000))) ** 2) + 2 * x1**2 - 1),
            "vars": [x0, x1],
            "bounds": [[-2.0, 5.0], [-3.0, 4.0]],
            "bottom_left_vertex": [-2.0, -3.0],
            "box_length": 7.0,
            "box_numbers": [1, 1],
        },
        "M7": {
            "f_expr": x0**2 + x1**2 + sp.Rational(1, 5) * (sp.cos(3 * x0) + sp.cos(3 * x1)) - 1,
            "vars": [x0, x1],
            "bounds": [[-2.0, 2.0], [-2.0, 2.0]],
            "bottom_left_vertex": [-2.0, -2.0],
            "box_length": 4.0,
            "box_numbers": [1, 1],
        },
    }


def _format_optional(value):
    if value is None:
        return "      None"
    return f"{value:10.3e}"


def run_all(
    *,
    use_local_bounds=True,
    compute_boxwise_min=True,
    visualise=False,
    verbose=0,
):
    examples = _examples()
    results = {}

    print(
        f"{'Name':<4} {'N':>2} {'M2':>10} {'M3':>10} "
        f"{'eps1':>10} {'eps2':>10} {'grad_lb':>10} {'reach_lb':>10} "
        f"{'t_bounds':>9} {'t_algo':>9} {'steps':>8}"
    )

    for name, example in examples.items():
        f_expr = example["f_expr"]
        vars_ = example["vars"]
        bounds = example["bounds"]

        grad_f = get_grad(f_expr, vars_)

        t0 = time.perf_counter()
        max_grad_norm = float(get_max_grad(f_expr, vars_, bounds))
        max_hess_norm = float(get_max_Hess(f_expr, vars_, bounds))
        t_bounds = time.perf_counter() - t0

        t1 = time.perf_counter()
        output = algo(
            f_expr,
            vars_,
            grad_f,
            example["bottom_left_vertex"],
            example["box_length"],
            example["box_numbers"],
            max_grad_norm,
            max_hess_norm,
            use_local_bounds=use_local_bounds,
            visualise=visualise,
            compute_boxwise_min=compute_boxwise_min,
            verbose=verbose,
        )
        t_algo = time.perf_counter() - t1

        grad_lower_bound = output["boxwise_grad_lower_bound"]
        reach_lower_bound = grad_lower_bound / (math.sqrt(len(vars_)) * max_hess_norm)

        results[name] = {
            "N": len(vars_),
            "M2": max_grad_norm,
            "M3": max_hess_norm,
            "eps1": output["smallest_case_one_box_length"],
            "eps2": output["smallest_case_two_box_length"],
            "grad_lb": grad_lower_bound,
            "reach_lb": reach_lower_bound,
            "t_bounds": t_bounds,
            "t_algo": t_algo,
            "steps": output["steps"],
            "raw_algo_output": output,
        }

        print(
            f"{name:<4} {len(vars_):>2d} "
            f"{max_grad_norm:>10.3e} {max_hess_norm:>10.3e} "
            f"{_format_optional(output['smallest_case_one_box_length'])} "
            f"{_format_optional(output['smallest_case_two_box_length'])} "
            f"{_format_optional(grad_lower_bound)} "
            f"{_format_optional(reach_lower_bound)} "
            f"{t_bounds:>9.4f} {t_algo:>9.4f} {output['steps']:>8d}"
        )

    return results


if __name__ == "__main__":
    run_all(
        use_local_bounds=True,
        compute_boxwise_min=True,
        visualise=False,
        verbose=0,
    )
