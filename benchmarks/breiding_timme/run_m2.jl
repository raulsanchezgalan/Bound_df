include("reach_core.jl")
using Printf

@polyvar x y
f = (x^3 - x*y^2 + y + 1)^2 * (x^2 + y^2 - 1) + y^2 - 5

t_total0 = time_ns()

t_bn0 = time_ns()
ρ_min = bottleneck_width_plane(f, [x, y]; seed = 577138)
t_bn = (time_ns() - t_bn0) / 1e9

# Note: seed=77 gives σ_max=1603.7738482346272 with nreal=23 (misses the true max).
t_curv0 = time_ns()
σ_max = max_curvature_plane(f, [x, y]; seed = 140163)
t_curv = (time_ns() - t_curv0) / 1e9

t_total = (time_ns() - t_total0) / 1e9

println("M2  ρ_min = ", ρ_min) # 0.13835123592621207
println("M2  σ_max = ", σ_max) # 2097.1664746610427

@printf("M2  time bottlenecks (s) = %.3f\n", t_bn) # 36.680
@printf("M2  time curvature   (s) = %.3f\n", t_curv) # 20.088
@printf("M2  time total       (s) = %.3f\n", t_total) # 56.768
