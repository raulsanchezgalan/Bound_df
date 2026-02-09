include("reach_core.jl")
using Printf

@polyvar x y
f = x^4 - x^2 * y^2 + y^4 - 4x^2 - 2y^2 - x - 4y + 1

# Fixed seeds for reproducibility (HomotopyContinuation has randomized internals).
seed_bn = 1
seed_curv = 1

t_total0 = time_ns()

t_bn0 = time_ns()
ρ_min = bottleneck_width_plane(f, [x, y]; seed = seed_bn)

# Note: curvature maxima can be seed-sensitive due to real-solution classification.
# Note: seed=7537 gives σ_max=4.991743168720899 with nreal=11 (misses the baseline max).
t_bn = (time_ns() - t_bn0) / 1e9

t_curv0 = time_ns()
σ_max = max_curvature_plane(f, [x, y]; seed = seed_curv)

t_curv = (time_ns() - t_curv0) / 1e9

t_total = (time_ns() - t_total0) / 1e9

println("M3  ρ_min = ", ρ_min) # 0.5026444029968232
println("M3  σ_max = ", σ_max) # 9.650392378427208
@printf("M3  time bottlenecks (s) = %.3f\n", t_bn) # 23.592
@printf("M3  time curvature   (s) = %.3f\n", t_curv) # 5.329
@printf("M3  time total       (s) = %.3f\n", t_total) # 28.922
