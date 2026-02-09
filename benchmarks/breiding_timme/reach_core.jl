using HomotopyContinuation
using LinearAlgebra: det, norm

"Build the Breiding–Timme bottleneck equations for a plane curve f(x,y)=0."
function build_bottleneck_system_plane(f, vars)
    @polyvar p[1:2] q[1:2]
    f_p = subs(f, vars => p)
    f_q = subs(f, vars => q)
    ∇_p = differentiate(f_p, p)
    ∇_q = differentiate(f_q, q)
    bn_eqs = [
        f_p;
        det([∇_p p - q]);
        f_q;
        det([∇_q p - q]);
    ]
    return bn_eqs, p, q
end

"Solve bottlenecks and return real nonsingular solutions (each solution is [p1,p2,q1,q2])."
function bottleneck_solutions_plane(f, vars)
    bn_eqs, p, q = build_bottleneck_system_plane(f, vars)
    res = solve(bn_eqs; start_system = :polyhedral)
    return real_solutions(nonsingular(res)), p, q
end

"Compute the narrowest bottleneck width, ignoring the diagonal p≈q."
function bottleneck_width_plane(f, vars; distinct_tol = 1e-9, seed::Union{Nothing,Integer} = nothing)
    bn_eqs, _, _ = build_bottleneck_system_plane(f, vars)
    res = seed === nothing ? solve(bn_eqs; start_system = :polyhedral) : solve(bn_eqs; start_system = :polyhedral, seed = UInt32(seed))
    sols = real_solutions(nonsingular(res))
    widths = Float64[]
    for s in sols
        p = s[1:2]
        q = s[3:4]
        w = norm(p - q)
        if w > distinct_tol
            push!(widths, w)
        end
    end
    return minimum(widths)
end

"Build the Breiding–Timme curvature-critical equations for a plane curve f(x,y)=0."
function build_curvature_critical_system_plane(f, vars)
    ∇ = differentiate(f, vars)
    H = differentiate(∇, vars)
    g = sum(∇ .^ 2)
    v = [-∇[2]; ∇[1]]
    h = v' * H * v
    dg = differentiate(g, vars)
    dh = differentiate(h, vars)
    ∇σ = g .* dh .- ((3 / 2) * h) .* dg
    F = [sum(v .* ∇σ); f]
    σ(pt) = h(pt) / g(pt)^(3 / 2)
    return F, σ
end

"Compute maximal curvature by solving critical equations."
function max_curvature_plane(f, vars)
    F, σ = build_curvature_critical_system_plane(f, vars)
    res = solve(F; start_system = :polyhedral)
    pts = real_solutions(nonsingular(res))
    return maximum(map(σ, pts))
end

function max_curvature_plane(f, vars; seed::Union{Nothing,Integer} = nothing)
    F, σ = build_curvature_critical_system_plane(f, vars)
    res = seed === nothing ? solve(F; start_system = :polyhedral) : solve(F; start_system = :polyhedral, seed = UInt32(seed))
    pts = real_solutions(nonsingular(res))
    return maximum(map(σ, pts))
end

"Build the Breiding–Timme bottleneck equations for a hypersurface f(x)=0 in R^n."
function build_bottleneck_system_hypersurface(f, vars)
    n = length(vars)
    @polyvar p[1:n] q[1:n] λ μ
    f_p = subs(f, vars => p)
    f_q = subs(f, vars => q)
    ∇_p = differentiate(f_p, p)
    ∇_q = differentiate(f_q, q)
    bn_eqs = [
        f_p;
        f_q;
        (p - q) .- λ .* ∇_p;
        (p - q) .- μ .* ∇_q;
    ]
    return bn_eqs, p, q
end

"Compute the narrowest bottleneck width for a hypersurface, ignoring the diagonal p≈q."
function bottleneck_width_hypersurface(f, vars; distinct_tol = 1e-9, seed::Union{Nothing,Integer} = nothing)
    n = length(vars)
    bn_eqs, _, _ = build_bottleneck_system_hypersurface(f, vars)
    res = seed === nothing ? solve(bn_eqs; start_system = :polyhedral) : solve(bn_eqs; start_system = :polyhedral, seed = UInt32(seed))
    sols = real_solutions(nonsingular(res))
    widths = Float64[]
    for s in sols
        p = s[1:n]
        q = s[(n + 1):(2n)]
        w = norm(p - q)
        if w > distinct_tol
            push!(widths, w)
        end
    end
    return minimum(widths)
end

"Build curvature-critical equations for a surface (hypersurface in R^3) f(x,y,z)=0.

We maximize κ^2 = (uᵀ H u)^2 / ||∇f||^2 over (x,u) with constraints:
  f(x)=0, ||u||=1, u⋅∇f(x)=0.
Criticality is enforced via Lagrange multipliers after clearing denominators.
Returns a polynomial system F and an evaluator κ(pt) (principal-curvature magnitude at the solution).
"
function build_curvature_critical_system_surface(f, vars)
    if length(vars) != 3
        error("build_curvature_critical_system_surface expects 3 variables")
    end

    @polyvar u[1:3] λ1 λ2 λ3
    xvars = vars
    allvars = [xvars; u]

    ∇f = differentiate(f, xvars)
    H = differentiate(∇f, xvars)
    g = sum(∇f .^ 2)

    # s = uᵀ H u
    Hu = H * u
    s = sum(u .* Hu)
    s2 = s^2

    c1 = f
    c2 = sum(u .^ 2) - 1
    c3 = sum(u .* ∇f)

    z = zero(f)
    z3 = [z, z, z]
    ∇c1 = [∇f; z3]
    ∇c2 = [z3; 2 .* u]
    ∇c3 = differentiate(c3, allvars)

    ∇s2 = differentiate(s2, allvars)
    ∇g = differentiate(g, allvars)

    # Polynomial KKT-like criticality after clearing denominators of κ^2 = s2/g.
    K = g .* ∇s2 .- s2 .* ∇g .- λ1 .* ∇c1 .- λ2 .* ∇c2 .- λ3 .* ∇c3

    F = [c1; c2; c3; K]
    κ(pt) = abs(s(pt[1:6])) / sqrt(g(pt[1:3]))
    return F, κ
end

"Compute maximal curvature magnitude for a surface f(x,y,z)=0 by solving curvature-critical equations."
function max_curvature_surface(f, vars; seed::Union{Nothing,Integer} = nothing)
    F, κ = build_curvature_critical_system_surface(f, vars)
    res = seed === nothing ? solve(F; start_system = :polyhedral) : solve(F; start_system = :polyhedral, seed = UInt32(seed))
    pts = real_solutions(nonsingular(res))
    return maximum(map(κ, pts))
end
