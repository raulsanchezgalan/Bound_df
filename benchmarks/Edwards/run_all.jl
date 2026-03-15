# run_all.jl
#
# Standalone benchmark runner for Edwards weak feature size code:
# - Wraps each example in try/catch so the script never stops early
# - Prints a summary table including an error/status column
#
# Run:
#   julia --depwarn=no run_all.jl
# or
#   julia run_all.jl

using HomologyInferenceWithWeakFeatureSize
using HomotopyContinuation
using Printf

# ============================================================
# Settings
# ============================================================

const MAX_ORDER = 2
const THRESH = 1e-8
const USE_MONODROMY = false

# ============================================================
# Helper function
# ============================================================

function run_example(name, F)
    start_time = time()
    dim = length(variables(F[1]))

    try
        wfs = compute_weak_feature_size(
            F;
            maximum_bottleneck_order = MAX_ORDER,
            threshold = THRESH,
            solve_with_monodromy = USE_MONODROMY,
        )
        elapsed = time() - start_time
        return (
            name = name,
            dim = dim,
            wfs = wfs,
            time = elapsed,
            status = "ok",
        )
    catch err
        elapsed = time() - start_time
        msg = sprint(showerror, err)
        msg = replace(msg, '\n' => ' ')
        if lastindex(msg) > 120
            msg = msg[1:120] * "…"
        end
        return (
            name = name,
            dim = dim,
            wfs = NaN,
            time = elapsed,
            status = msg,
        )
    end
end

# ============================================================
# Main benchmark
# ============================================================

function main()
    println("\nRunning Edwards weak feature size benchmarks (try/catch enabled)\n")

    @var x y z

    results = []

    # ------------------------------------------------------------
    # M1
    # ------------------------------------------------------------
    F1 = [x^2 + y^2 - 1]
    push!(results, run_example("M1", F1))

    # ------------------------------------------------------------
    # M2
    # ------------------------------------------------------------
    F2 = [(x^3 - x*y^2 + y + 1)^2 * (x^2 + y^2 - 1) + y^2 - 5]
    push!(results, run_example("M2", F2))

    # ------------------------------------------------------------
    # M3
    # ------------------------------------------------------------
    F3 = [x^4 - x^2*y^2 + y^4 - 4x^2 - 2y^2 - x - 4y + 1]
    push!(results, run_example("M3", F3))

    # ------------------------------------------------------------
    # M4 (3D)
    # ------------------------------------------------------------
    F4 = [4x^2 + 7y^4 + 3z^4 - 3
          - 8x^3 + 2x^2*y - 4x^2
          - 8x*y^2 - 5x*y + 8x
          - 6y^3 + 8y^2 + 4y]
    push!(results, run_example("M4", F4))

    # ------------------------------------------------------------
    # M5
    # ------------------------------------------------------------
    P30 = sum(x^(2k) * y^(30 - 2k) for k in 0:15)
    F5 = [x^2 + y^2 - 1 + 1e-5 * P30]
    push!(results, run_example("M5", F5))

    # ------------------------------------------------------------
    # M6 (shifted ellipse)
    # ------------------------------------------------------------
    F6 = [(x^2 + 2y^2 - 1) * (((x - 2 - 1e-5)^2) + 2y^2 - 1)]
    push!(results, run_example("M6", F6))

    # ============================================================
    # Print summary table
    # ============================================================

    println("------------------------------------------------------------")
    @printf("%-4s %-3s %-18s %-10s %-s\n",
            "ID", "N", "wfs_lower_bound", "time(s)", "status")
    println("------------------------------------------------------------")

    for r in results
        @printf("%-4s %-3d %-18.10e %-10.4f %-s\n",
                r.name,
                r.dim,
                r.wfs,
                r.time,
                r.status)
    end

    println("------------------------------------------------------------\n")
end

# Avoid Julia world-age issues when running as a script
Base.invokelatest(main)
