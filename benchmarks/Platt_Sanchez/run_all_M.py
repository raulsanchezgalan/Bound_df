#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone reach lower-bound benchmark (Platt–Sánchez style), repo-independent.

- Implements the subdivision-based algorithm (Case 1 / Case 2) to lower-bound |∇f|_1 on Z(f)
- Converts that into a reach lower bound:  tau >= C2 / (sqrt(N) * C1)
  where C2 is a proven lower bound for |∇f|_1 on boxes covering Z(f),
  and C1 is an upper bound for ||Hess f||_2 on the ambient bounding box.

Features:
- Silent (no per-iteration prints)
- Optional "local bounds" per box (sampling-based, fast-ish, robust)
- Fixes SymPy->NumPy Hessian broadcasting bug (important for 3D, e.g. M4)
- Runs examples M1–M6 and prints a summary table


"""

import time
import math
import numpy as np
import sympy as sp


# -----------------------------
# Numeric function construction
# -----------------------------

def _make_numeric_functions(f_expr, vars_):
    """
    Robustly lambdify f, grad f, Hess f.

    Returns:
      f_num(X): X shape (m,N) or (N,) -> (m,) or scalar
      grad_num(X): -> (m,N) or (N,)
      hess_num(X): -> (m,N,N) or (N,N)

    
    We lambdify Hessian entrywise to avoid inhomogeneous array errors for vector inputs.
    """
    N = len(vars_)

    grad_syms = [sp.diff(f_expr, v) for v in vars_]
    hess_syms = [[sp.diff(f_expr, vars_[i], vars_[j]) for j in range(N)] for i in range(N)]

    f_lam = sp.lambdify(vars_, f_expr, "numpy")
    g_lams = [sp.lambdify(vars_, g, "numpy") for g in grad_syms]
    H_lams = [[sp.lambdify(vars_, hess_syms[i][j], "numpy") for j in range(N)] for i in range(N)]

    def f_num(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return float(f_lam(*X))
        cols = [X[:, i] for i in range(N)]
        return np.asarray(f_lam(*cols), dtype=float)

    def grad_num(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return np.array([float(g(*X)) for g in g_lams], dtype=float)
        cols = [X[:, i] for i in range(N)]
        return np.stack([np.asarray(g(*cols), dtype=float) for g in g_lams], axis=1)  # (m,N)

    def hess_num(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            H = np.empty((N, N), dtype=float)
            for i in range(N):
                for j in range(N):
                    H[i, j] = float(H_lams[i][j](*X))
            return H
        cols = [X[:, i] for i in range(N)]
        m = X.shape[0]
        H = np.empty((m, N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                H[:, i, j] = np.asarray(H_lams[i][j](*cols), dtype=float)
        return H

    return f_num, grad_num, hess_num


# -----------------------------
# Norm helpers
# -----------------------------

def _spectral_norm_batch(H):
    """
    H: (m,N,N) -> returns (m,) spectral norms (operator 2-norm) for each matrix.
    """
    s = np.linalg.svd(H, compute_uv=False)
    return s[:, 0]


def _box_vertices(bounds):
    """
    bounds: list of (low, high) length N
    returns vertices array shape (2^N, N)
    """
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)
    N = len(bounds)
    verts = []
    for mask in range(1 << N):
        v = lows.copy()
        for i in range(N):
            if (mask >> i) & 1:
                v[i] = highs[i]
        verts.append(v)
    return np.array(verts, dtype=float)


# -----------------------------
# Bounding routines
# -----------------------------

def _global_bounds_grid(bounds, grad_num, hess_num, pts_per_dim):
    """
    Crude global upper bounds on ||grad||_2 and ||Hess||_2 over the bounding box via grid sampling.
    """
    N = len(bounds)
    axes = [np.linspace(bounds[i][0], bounds[i][1], pts_per_dim) for i in range(N)]
    meshes = np.meshgrid(*axes, indexing="xy")
    X = np.stack([m.reshape(-1) for m in meshes], axis=1)  # (m,N)

    G = grad_num(X)
    M2 = float(np.max(np.linalg.norm(G, axis=1)))

    H = hess_num(X)  # (m,N,N)
    M3 = float(np.max(_spectral_norm_batch(H)))

    return M2, M3


def _local_bounds_sample(box_bounds, grad_num, hess_num, rng, n_random_points=2):
    """
    Fast local upper bounds on ||grad||_2 and ||Hess||_2 over a box by sampling:
    - all vertices
    - midpoint
    - a few random points
    """
    verts = _box_vertices(box_bounds)
    mid = np.array([(a + b) / 2.0 for a, b in box_bounds], dtype=float)

    pts = [mid]
    pts.extend(list(verts))

    if n_random_points > 0:
        lows = np.array([a for a, _ in box_bounds], dtype=float)
        highs = np.array([b for _, b in box_bounds], dtype=float)
        r = lows + (highs - lows) * rng.random((n_random_points, len(box_bounds)))
        pts.extend(list(r))

    P = np.array(pts, dtype=float)

    G = grad_num(P)
    M2 = float(np.max(np.linalg.norm(G, axis=1)))

    H = hess_num(P)
    M3 = float(np.max(_spectral_norm_batch(H)))

    return M2, M3


# -----------------------------
# Core algorithm (silent)
# -----------------------------

def reach_algo_silent(
    f_expr,
    vars_,
    bounds,
    *,
    use_local_bounds=True,
    local_random_points=2,
    max_steps=600_000,
    global_grid_pts_per_dim=41,
    seed=0,
):
    """
    Runs the subdivision algorithm and returns metrics.

    bounds: list of [min,max] per variable, e.g. [[-2,2],[-2,2]]
    """

    t0 = time.perf_counter()
    N = len(vars_)
    rng = np.random.default_rng(seed)

    # Numeric oracles
    f_num, grad_num, hess_num = _make_numeric_functions(f_expr, vars_)

    # Global bounds (always computed for reporting + for non-local mode)
    tb0 = time.perf_counter()
    # Optional: reduce grid in 3D for speed while staying reasonably safe
    pts = global_grid_pts_per_dim
    if N >= 3:
        pts = min(pts, 21)
    M2_global, M3_global = _global_bounds_grid(bounds, grad_num, hess_num, pts_per_dim=pts)
    t_bounds = time.perf_counter() - tb0

    # Initial box = the whole bounds
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)
    side_lengths0 = highs - lows

    # We keep a stack of (bottom_left, side_lengths)
    NewBoxes = [(lows.copy(), side_lengths0.copy())]
    CaseOne_eps = np.inf
    CaseTwo_eps = np.inf

    # We track the best (smallest) certified lower bound for |grad|_1 across CaseTwo boxes.
    # For a CaseTwo box of min side length eps:  |grad|_1 >= (N^(3/2) * eps * M3_local)/2
    grad_lower_bound = np.inf

    steps = 0

    # Useful constants
    sqrtN = math.sqrt(N)
    N32 = N ** 1.5

    # Main loop
    while NewBoxes and steps < max_steps:
        steps += 1
        bl, sl = NewBoxes.pop()
        eps = float(np.min(sl))
        mid = bl + 0.5 * sl

        # Bounds for this box
        if use_local_bounds:
            box_bounds = [(float(bl[i]), float(bl[i] + sl[i])) for i in range(N)]
            M2_box, M3_box = _local_bounds_sample(
                box_bounds, grad_num, hess_num, rng, n_random_points=local_random_points
            )
        else:
            M2_box, M3_box = M2_global, M3_global

        f_abs = abs(f_num(mid))
        grad1 = float(np.sum(np.abs(grad_num(mid))))

        # Case 1: box contains no zero
        if f_abs > sqrtN * eps * M2_box:
            CaseOne_eps = min(CaseOne_eps, eps)
            continue

        # Case 2: certify a lower bound on |grad|_1 over the whole box
        if grad1 > N32 * eps * M3_box:
            CaseTwo_eps = min(CaseTwo_eps, eps)
            # certified boxwise lower bound (from inequality)
            gb = (N32 * eps * M3_box) / 2.0
            grad_lower_bound = min(grad_lower_bound, gb)
            continue

        # Else: subdivide into 2^N subboxes by halving each coordinate
        half = 0.5 * sl
        for mask in range(1 << N):
            child_bl = bl.copy()
            for i in range(N):
                if (mask >> i) & 1:
                    child_bl[i] += half[i]
            NewBoxes.append((child_bl, half.copy()))

    t_algo = time.perf_counter() - t0

    # If we never hit CaseTwo, grad_lower_bound stays inf => no reach bound
    if not np.isfinite(grad_lower_bound):
        reach_lb = 0.0
    else:
        # Use global Hess bound for C1 (as in corollary)
        # tau >= C2 / (sqrt(N)*C1)
        reach_lb = float(grad_lower_bound / (sqrtN * M3_global))

    return {
        "N": N,
        "M2": M2_global,
        "M3": M3_global,
        "eps1": 0.0 if not np.isfinite(CaseOne_eps) else CaseOne_eps,
        "eps2": 0.0 if not np.isfinite(CaseTwo_eps) else CaseTwo_eps,
        "grad_lb": grad_lower_bound,
        "reach_lb": reach_lb,
        "t_bounds": t_bounds,
        "t_algo": t_algo,
        "t_total": t_algo,  # kept for convenience
        "steps": steps,
        "terminated": (len(NewBoxes) == 0),
    }


# -----------------------------
# Examples M1–M6
# -----------------------------

def _examples():
    x, y, z = sp.symbols("x y z", real=True)

    # M1
    f1 = x**2 + y**2 - 1
    b1 = [[-2.0, 2.0], [-2.0, 2.0]]

    # M2
    f2 = (x**3 - x*y**2 + y + 1)**2 * (x**2 + y**2 - 1) + y**2 - 5
    b2 = [[-3.0, 3.0], [-3.0, 3.0]]

    # M3
    f3 = x**4 - x**2*y**2 + y**4 - 4*x**2 - 2*y**2 - x - 4*y + 1
    b3 = [[-3.0, 3.0], [-3.0, 3.0]]

    # M4 (3D)
    f4 = (4*x**2 + 7*y**4 + 3*z**4 - 3 - 8*x**3 + 2*x**2*y - 4*x**2
          - 8*x*y**2 - 5*x*y + 8*x - 6*y**3 + 8*y**2 + 4*y)
    b4 = [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]

    # M5
    P30 = sum(x**(2*k) * y**(30-2*k) for k in range(16))
    f5 = x**2 + y**2 - 1 + sp.Rational(1, 100000) * P30  # 10^-5
    b5 = [[-2.0, 2.0], [-2.0, 2.0]]

    # M6 
    f6 = (x**2 + 2*y**2 - 1) * (((x - (2 + sp.Rational(1, 100000)))**2) + 2*y**2 - 1)
    b6 = [[-2.0, 5.0], [-3.0, 3.0]]

    return {
        "M1": (f1, [x, y], b1),
        "M2": (f2, [x, y], b2),
        "M3": (f3, [x, y], b3),
        "M4": (f4, [x, y, z], b4),
        "M5": (f5, [x, y], b5),
        "M6": (f6, [x, y], b6),
    }


# -----------------------------
# Runner
# -----------------------------

def run_all(
    *,
    use_local_bounds=True,
    local_random_points=2,
    max_steps=600_000,
    global_grid_pts_per_dim=41,
    seed=0,
):
    examples = _examples()

    cols = ["Name", "N", "M2", "M3", "eps1", "eps2", "grad_lb", "reach_lb", "t_bounds", "t_algo", "t_total", "steps"]
    print(f"{cols[0]:<4} {cols[1]:>2} {cols[2]:>10} {cols[3]:>10} {cols[4]:>10} {cols[5]:>10} {cols[6]:>10} {cols[7]:>10} {cols[8]:>9} {cols[9]:>9} {cols[10]:>9} {cols[11]:>8}")

    for name, (f_expr, vars_, bounds) in examples.items():
        out = reach_algo_silent(
            f_expr,
            vars_,
            bounds,
            use_local_bounds=use_local_bounds,
            local_random_points=local_random_points,
            max_steps=max_steps,
            global_grid_pts_per_dim=global_grid_pts_per_dim,
            seed=seed,
        )
        print(
            f"{name:<4} {out['N']:>2d} "
            f"{out['M2']:>10.3e} {out['M3']:>10.3e} "
            f"{out['eps1']:>10.3e} {out['eps2']:>10.3e} "
            f"{out['grad_lb']:>10.3e} {out['reach_lb']:>10.3e} "
            f"{out['t_bounds']:>9.4f} {out['t_algo']:>9.4f} {out['t_total']:>9.4f} "
            f"{out['steps']:>8d}"
        )


if __name__ == "__main__":
    # Recommended defaults:
    # - local bounds help a lot on M2/M4/M5
    # - local_random_points=0 is fastest but less robust; 2 is a good compromise
    run_all(
        use_local_bounds=True,
        local_random_points=2,
        max_steps=600_000,
        global_grid_pts_per_dim=41,
        seed=0,
    )
