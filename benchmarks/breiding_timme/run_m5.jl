include("reach_core.jl")
using Printf

@polyvar x y
pert = x^30 + x^28 * y^2 + x^26 * y^4 + x^24 * y^6 + x^22 * y^8 + x^20 * y^10 + x^18 * y^12 + x^16 * y^14 + x^14 * y^16 + x^12 * y^18 + x^10 * y^20 + x^8 * y^22 + x^6 * y^24 + x^4 * y^26 + x^2 * y^28 + y^30
f = x^2 + y^2 - 1 + 1e-5 * pert

seed_bn = 1
seed_curv = 1

t_total0 = time_ns()

t_bn0 = time_ns()
ρ_min = bottleneck_width_plane(f, [x, y]; seed = seed_bn)

t_bn = (time_ns() - t_bn0) / 1e9

t_curv0 = time_ns()
σ_max = max_curvature_plane(f, [x, y]; seed = seed_curv)

t_curv = (time_ns() - t_curv0) / 1e9

t_total = (time_ns() - t_total0) / 1e9

println("M5  ρ_min = ", ρ_min) # 1.9999900014746774
println("M5  σ_max = ", σ_max) # 1.0000645795890644
@printf("M5  time bottlenecks (s) = %.3f\n", t_bn) # 1130.416
@printf("M5  time curvature   (s) = %.3f\n", t_curv) # 30.466
@printf("M5  time total       (s) = %.3f\n", t_total) # 1160.883
