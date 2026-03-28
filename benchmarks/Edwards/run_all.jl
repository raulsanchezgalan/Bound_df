# run_all.jl
#
# Single-example runner for Edwards weak feature size code.
# Uncomment exactly one example at the bottom and run the file.

using HomologyInferenceWithWeakFeatureSize
using HomotopyContinuation

# ============================================================
# Settings
# ============================================================

const MAX_ORDER = 2
const USE_MONODROMY = false

# ============================================================
# Helper function
# ============================================================

function run_example(name, F; solve_with_monodromy = USE_MONODROMY)
    start_time = time()
    dim = length(variables(F[1]))

    try
        wfs = compute_weak_feature_size(
            F;
            maximum_bottleneck_order = MAX_ORDER,
            solve_with_monodromy = solve_with_monodromy,
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
    @var x y z

    F1 = [x^2 + y^2 - 1]
    F2 = [(x^3 - x*y^2 + y + 1)^2 * (x^2 + y^2 - 1) + y^2 - 5]
    F3 = [x^4 - x^2*y^2 + y^4 - 4x^2 - 2y^2 - x - 4y + 1]
    F4 = [4x^2 + 7y^4 + 3z^4 - 3
          - 8x^3 + 2x^2*y - 4x^2
          - 8x*y^2 - 5x*y + 8x
          - 6y^3 + 8y^2 + 4y]
    P30 = sum(x^(2k) * y^(30 - 2k) for k in 0:15)
    F5 = [x^2 + y^2 - 1 + 1e-5 * P30]
    F6 = [(x^2 + 2y^2 - 1) * (((x - 2 - 1e-5)^2) + 2y^2 - 1)]

    println(run_example("M1", F1))
    # println(run_example("M2", F2))
    # println(run_example("M3", F3))
    # println(run_example("M4", F4; solve_with_monodromy = true))
    # println(run_example("M5", F5; solve_with_monodromy = true))
    # println(run_example("M6", F6))
end

main()
