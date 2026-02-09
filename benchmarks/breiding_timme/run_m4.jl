include("reach_core.jl")
using Printf

@polyvar x y z
f = 4x^2 + 7y^4 + 3z^4 - 3 - 8x^3 + 2x^2*y - 4x^2 - 8x*y^2 - 5x*y + 8x - 6y^3 + 8y^2 + 4y

seed_bn = 1
seed_curv = 1

t_total0 = time_ns()

t_bn0 = time_ns()
ρ_min = bottleneck_width_hypersurface(f, [x, y, z]; seed = seed_bn)

t_bn = (time_ns() - t_bn0) / 1e9

t_curv0 = time_ns()
σ_max = max_curvature_surface(f, [x, y, z]; seed = seed_curv)

t_curv = (time_ns() - t_curv0) / 1e9

t_total = (time_ns() - t_total0) / 1e9

println("M4  ρ_min = ", ρ_min) # 0.8311307816041904
println("M4  σ_max = ", σ_max) # 33.41540960909607
@printf("M4  time bottlenecks (s) = %.3f\n", t_bn) # 24.639
@printf("M4  time curvature   (s) = %.3f\n", t_curv) # 1831.316
@printf("M4  time total       (s) = %.3f\n", t_total) # 1855.956
