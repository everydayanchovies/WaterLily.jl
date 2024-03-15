"""
    AutoBody(sdf,map=(s,I,x,t)->x; compose=true) <: AbstractBody

  - `sdf(x::AbstractVector,t::Real)::Real`: signed distance function
  - `map(x::AbstractVector,t::Real)::AbstractVector`: coordinate mapping function
  - `compose::Bool=true`: Flag for composing `sdf=sdf∘map`

Implicitly define a geometry by its `sdf` and optional coordinate `map`. Note: the `map`
is composed automatically if `compose=true`, i.e. `sdf(s,I,x,t) = sdf(s,I,map(s,I,x,t),t)`.
Both parameters remain independent otherwise. It can be particularly heplful to set
`compose=false` when adding mulitple bodies together to create a more complex one.
"""
struct AutoBody{F1<:Function,F2<:Function} <: AbstractBody
    sdf::F1
    map::F2
    function AutoBody(sdf, map = (s, I, x, t) -> x; compose = true)
        comp(s, I, x, t) = compose ? sdf(s, I, map(s, I, x, t), t) : sdf(s, I, x, t)
        new{typeof(comp),typeof(map)}(comp, map)
    end
end

function Base.:+(a::AutoBody, b::AutoBody)
    map(s, I, x, t) =
        ifelse(a.sdf(s, I, x, t) < b.sdf(s, I, x, t), a.map(s, I, x, t), b.map(s, I, x, t))
    sdf(s, I, x, t) = min(a.sdf(s, I, x, t), b.sdf(s, I, x, t))
    AutoBody(sdf, map, compose = false)
end
function Base.:∩(a::AutoBody, b::AutoBody)
    map(s, I, x, t) =
        ifelse(a.sdf(s, I, x, t) > b.sdf(s, I, x, t), a.map(s, I, x, t), b.map(s, I, x, t))
    sdf(s, I, x, t) = max(a.sdf(s, I, x, t), b.sdf(s, I, x, t))
    AutoBody(sdf, map, compose = false)
end
Base.:∪(x::AutoBody, y::AutoBody) = x + y
Base.:-(x::AutoBody) = AutoBody((s, I, d, t) -> -x.sdf(s, I, d, t), x.map, compose = false)
Base.:-(x::AutoBody, y::AutoBody) = x ∩ -y

"""
    d = sdf(body::AutoBody,s,I,x,t) = body.sdf(s,I,x,t)
"""
sdf(body::AutoBody, s, I, x, t) = body.sdf(s, I, x, t)

"""
    Bodies(bodies, ops::AbstractVector)

  - `bodies::Vector{AutoBody}`: Vector of AutoBody
  - `ops::Vector{Function}`: Vector of operators for the superposition of multiple AutoBody

Superposes multiple `body::AutoBody` objects together according to the operators `ops`.
While this can be manually performed by the operators implemented for `AutoBody`, adding too many
bodies can yield a recursion problem of the `sdf and `map` functions not fitting in the stack.
This type implements the superposition of bodies by iteration instead of recursion, and the reduction of the sdf and map
functions is done on the `mesure` function, and not before.
The operators vector `ops` specifies the specific operation to call between two consecutive bodies in the vector of `bodies`.
Note that `+` (or the alias `∪`) is the only operation supported between `Bodies`.
"""
struct Bodies <: AbstractBody
    bodies::Vector{AutoBody}
    ops::Vector{Function}
    function Bodies(bodies, ops::AbstractVector)
        all(x -> x == Base.:+ || x == Base.:- || x == Base.:∩ || x == Base.:∪, ops) &&
            ArgumentError("Operations array not supported. Use only `ops ∈ [+,-,∩,∪]`")
        length(bodies) != length(ops) + 1 &&
            ArgumentError("length(bodies) != length(ops)+1")
        new(bodies, ops)
    end
end
Bodies(bodies) = Bodies(bodies, repeat([+], length(bodies) - 1))
Bodies(bodies, op::Function) = Bodies(bodies, repeat([op], length(bodies) - 1))
Base.:+(a::Bodies, b::Bodies) = Bodies(vcat(a.bodies, b.bodies), vcat(a.ops, b.ops))
Base.:∪(a::Bodies, b::Bodies) = a + b

"""
    sdf_map_d(ab::Bodies,s,I,x,t)

Returns the `sdf` and `map` functions, and the distance `d` (`d=sdf(s,I,x,t)`) for `::Bodies`.
If bodies are not actual `::AutoBody`, it recursively iterates in the nested bodies of the vector.
"""
function sdf_map_d(bodies, ops, s, I, x, t)
    sdf, map, d = bodies[1].sdf, bodies[1].map, bodies[1].sdf(s, I, x, t)
    for i ∈ eachindex(bodies)[begin+1:end]
        sdf2, map2, d2 = bodies[i].sdf, bodies[i].map, bodies[i].sdf(s, I, x, t)
        sdf, map, d = reduce_sdf_map(sdf, map, d, sdf2, map2, d2, ops[i-1], s, I, x, t)
    end
    return sdf, map, d
end
"""
    reduce_sdf_map(sdf_a,map_a,d_a,sdf_b,map_b,d_b,op,s,I,x,t)

Returns `sdf`, `map`, and `sdf(s,I,x,t)` between two (unpacked) `AutoBody`s.
"""
function reduce_sdf_map(sdf_a, map_a, d_a, sdf_b, map_b, d_b, op, s, I, x, t)
    (Base.:+ == op || Base.:∪ == op) && d_b < d_a && return (sdf_b, map_b, d_b)
    Base.:- == op && -d_b > d_a && return ((y, u) -> -sdf_b(y, u), map_b, -d_b)
    Base.:∩ == op && d_b > d_a && return (sdf_b, map_b, d_b)
    return sdf_a, map_a, d_a
end
"""
    d = sdf(a::Bodies,s,I,x,t)

Compute distance for `Bodies` type.
"""
sdf(a::Bodies, s, I, x, t) = sdf_map_d(a.bodies, a.ops, s, I, x, t)[end]

using ForwardDiff
"""
    d,n,V = measure(body::AutoBody,s,I,x,t)
    d,n,V = measure(body::Bodies,s,I,x,t)

Determine the implicit geometric properties from the `sdf` and `map`.
The gradient of `d=sdf(map(s,I,x,t))` is used to improve `d` for pseudo-sdfs.
The velocity is determined _solely_ from the optional `map` function.
"""
measure(body::AutoBody, s, I, x, t) = measure(body.sdf, body.map, s, I, x, t)
function measure(a::Bodies, s, I, x, t)
    sdf, map, _ = sdf_map_d(a.bodies, a.ops, s, I, x, t)
    measure(sdf, map, s, I, x, t)
end
function measure(sdf, map, s, I, x, t)
    # eval d=f(s,I,x,t), and n̂ = ∇f
    d = sdf(s, I, x, t)
    n = ForwardDiff.gradient(x -> sdf(s, I, x, t), x)
    any(isnan.(n)) && return (d, zero(x), zero(x))

    # correct general implicit fnc f(x₀)=0 to be a pseudo-sdf
    #   f(x) = f(x₀)+d|∇f|+O(d²) ∴  d ≈ f(x)/|∇f|
    m = √sum(abs2, n)
    d /= m
    n /= m

    # The velocity depends on the material change of ξ=m(s,I,x,t):
    #   Dm/Dt=0 → ṁ + (dm/dx)ẋ = 0 ∴  ẋ =-(dm/dx)\ṁ
    J = ForwardDiff.jacobian(x -> map(s, I, x, t), x)
    dot = ForwardDiff.derivative(t -> map(s, I, x, t), t)
    return (d, n, -J \ dot)
end

using LinearAlgebra: tr
"""
    curvature(A::AbstractMatrix)

Return `H,K` the mean and Gaussian curvature from `A=hessian(sdf)`.
`K=tr(minor(A))` in 3D and `K=0` in 2D.
"""
function curvature(A::AbstractMatrix)
    H, K = 0.5 * tr(A), 0
    if size(A) == (3, 3)
        K =
            A[1, 1] * A[2, 2] + A[1, 1] * A[3, 3] + A[2, 2] * A[3, 3] - A[1, 2]^2 -
            A[1, 3]^2 - A[2, 3]^2
    end
    H, K
end
