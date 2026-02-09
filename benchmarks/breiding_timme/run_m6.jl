include("reach_core.jl")
using Printf

@polyvar x y
f = (x^2 + 2y^2 - 1) * (x^2 + 2y^2 - 3 - 1e-5)

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

println("M6  ρ_min = ", ρ_min) # 0.5176401314447928
println("M6  σ_max = ", σ_max) # 1.1546986138831639
@printf("M6  time bottlenecks (s) = %.3f\n", t_bn) # 26.212
@printf("M6  time curvature   (s) = %.3f\n", t_curv) # 4.399
@printf("M6  time total       (s) = %.3f\n", t_total) # 30.612
