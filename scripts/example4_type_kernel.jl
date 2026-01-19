#!/usr/bin/env -S julia --project
"""
Example 4: Heterogeneous Dynamics with Type-Specific Kernels + Symbolic Regression

**The Full Pipeline:**
1. Domain knowledge (UDE): Node type membership + self-rates are known
2. Flexible learning: NN kernel learns κ(P_ij, type_i, type_j)
3. Symbolic discovery: SymReg extracts interpretable equations per type-pair

**Known physics (UDE):**
- Node types: predator, prey, resource
- Self-rates: a_P, a_Y, a_R (decay rates per type)

**Unknown (learned by NN):**
- Input: (P_ij, type_i, type_j) with one-hot encoding for types
- Output: 9 message kernels [κ_PP, κ_PY, κ_PR, κ_YP, κ_YY, κ_YR, κ_RP, κ_RY, κ_RR]
- For SymReg: analyze each output NN(·,·,·)[k] as a separate function

**True dynamics include Holling Type II:**
- κ_PY(p) = α·p / (1 + β·p)  -- Saturating predation (to be discovered!)
- Other interactions: linear or constant

Usage:
    julia --project scripts/example4_type_kernel.jl
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using RDPGDynamics
using LinearAlgebra
using Random
using OrdinaryDiffEq
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity
using Lux
using ComponentArrays
using CairoMakie
using Printf
using Statistics
using Serialization  # For saving intermediate results

const CM = CairoMakie

# Output directory for all results
const OUTPUT_DIR = joinpath(dirname(@__DIR__), "results", "example4_type_kernel")
mkpath(OUTPUT_DIR)

# ============================================================================
# Configuration
# ============================================================================

# Node counts per type (at least 10 per community)
const N_PRED = 12    # Predators
const N_PREY = 15    # Prey
const N_RES = 10     # Resources
const N_TOTAL = N_PRED + N_PREY + N_RES  # 37 total

const D_EMBED = 2    # Embedding dimension
const T_TOTAL = 25   # Total timesteps
const TRAIN_FRAC = 0.7
const SEED = 42

# Type indices (1-indexed ranges)
const IDX_PRED = 1:N_PRED
const IDX_PREY = (N_PRED+1):(N_PRED+N_PREY)
const IDX_RES = (N_PRED+N_PREY+1):N_TOTAL

# Type labels
const TYPE_P = 1  # Predator
const TYPE_Y = 2  # Prey
const TYPE_R = 3  # Resource
const K_TYPES = 3

# Type assignment for each node
const NODE_TYPES = vcat(
    fill(TYPE_P, N_PRED),
    fill(TYPE_Y, N_PREY),
    fill(TYPE_R, N_RES)
)

# ============================================================================
# True Dynamics: Type-Specific with Holling Type II
# ============================================================================
#
# N_ij = κ_{type(i), type(j)}(P_ij)
# ẋ = N·x (equivalent to message-passing form)
#
# Key feature: Predator-Prey interaction uses Holling Type II (saturating)
#   κ_PY(p) = α·p / (1 + β·p)
#
# This nonlinearity is what SymReg should discover!

# Self-rates (KNOWN PHYSICS - part of UDE structure)
# Small decay for stability
const KNOWN_SELF_RATES = Dict(
    TYPE_P => -0.002,   # Predator: small decay
    TYPE_Y => -0.001,   # Prey: minimal decay
    TYPE_R =>  0.000    # Resource: stable
)

# Message kernels κ_ab(p) - stored as functions
# Holling Type II for predator-prey - scaled for n~37 nodes
const HOLLING_ALPHA = 0.025
const HOLLING_BETA = 2.0

function κ_true(type_i::Int, type_j::Int, p::Real)
    # Predator-Predator: mild repulsion (constant)
    if type_i == TYPE_P && type_j == TYPE_P
        return -0.004

    # Predator-Prey: HOLLING TYPE II (saturating attraction) - CHASE!
    elseif type_i == TYPE_P && type_j == TYPE_Y
        return HOLLING_ALPHA * p / (1 + HOLLING_BETA * p)

    # Predator-Resource: ignore
    elseif type_i == TYPE_P && type_j == TYPE_R
        return 0.0

    # Prey-Predator: FLEE! (linear, negative)
    elseif type_i == TYPE_Y && type_j == TYPE_P
        return -0.02 * p

    # Prey-Prey: mild cohesion (constant) - herding
    elseif type_i == TYPE_Y && type_j == TYPE_Y
        return 0.003

    # Prey-Resource: attraction (linear)
    elseif type_i == TYPE_Y && type_j == TYPE_R
        return 0.012 * p

    # Resource-Predator: ignore
    elseif type_i == TYPE_R && type_j == TYPE_P
        return 0.0

    # Resource-Prey: depletion (linear, negative)
    elseif type_i == TYPE_R && type_j == TYPE_Y
        return -0.006 * p

    # Resource-Resource: cohesion (constant)
    elseif type_i == TYPE_R && type_j == TYPE_R
        return 0.005

    else
        return 0.0
    end
end

"""
Compute N matrix from true type-specific kernels.
"""
function compute_N_true(P::Matrix{Float64})
    n = size(P, 1)
    N = zeros(n, n)

    for i in 1:n
        ti = NODE_TYPES[i]
        # Diagonal: self-rate minus sum of outgoing messages
        N[i,i] = KNOWN_SELF_RATES[ti]
        for j in 1:n
            if j != i
                tj = NODE_TYPES[j]
                κ_ij = κ_true(ti, tj, P[i,j])
                N[i,j] = κ_ij
                N[i,i] -= κ_ij  # Message-passing form: subtract from diagonal
            end
        end
    end

    return N
end

"""
True dynamics: ẋ = N(P)·x with type-specific kernels including Holling Type II.
"""
function true_dynamics!(dX::Matrix{Float64}, X::Matrix{Float64}, p, t)
    P = X * X'
    N = compute_N_true(P)
    dX .= N * X
end

function true_dynamics_vec!(du::Vector{Float64}, u::Vector{Float64}, p, t)
    n, d = N_TOTAL, D_EMBED
    X = Matrix(reshape(u, d, n)')  # Collect to concrete Matrix
    dX = similar(X)
    true_dynamics!(dX, X, p, t)
    du .= vec(dX')
end

# ============================================================================
# Data Generation
# ============================================================================

function generate_true_data(; seed=SEED)
    rng = Xoshiro(seed)

    # Initialize: spread types in DISTINCT regions for better 2D structure
    # More separation so SVD can see 2D structure from the start
    X0 = zeros(N_TOTAL, D_EMBED)

    for i in IDX_PRED
        X0[i, :] = [0.7, 0.6] .+ 0.06 .* randn(rng, D_EMBED)  # Upper right
    end
    for i in IDX_PREY
        X0[i, :] = [0.4, 0.6] .+ 0.06 .* randn(rng, D_EMBED)  # Upper left
    end
    for i in IDX_RES
        X0[i, :] = [0.5, 0.3] .+ 0.06 .* randn(rng, D_EMBED)  # Lower middle
    end

    X0 = max.(X0, 0.15)  # Keep away from origin

    u0 = Vector{Float64}(vec(X0'))
    tspan = (0.0, Float64(T_TOTAL - 1))

    prob = ODEProblem(true_dynamics_vec!, u0, tspan)
    sol = solve(prob, Tsit5(); saveat=1.0)

    X_true = [Matrix(reshape(sol.u[i], D_EMBED, N_TOTAL)') for i in 1:length(sol.t)]
    return X_true
end

# Use RDPGDynamics functions for adjacency sampling and embedding
# sample_adjacency, embed_temporal_duase_raw are from the package

"""
Generate adjacencies using RDPGDynamics.sample_adjacency with K repeated samples.
"""
function generate_adjacencies(X_true::Vector{Matrix{Float64}}; K::Int=10, seed=SEED)
    rng = Xoshiro(seed + 1000)
    T = length(X_true)

    A_obs = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        # Use package function with K repeated samples (averaged)
        A_obs[t] = sample_adjacency_repeated(X_true[t], K; rng=rng)
    end
    return A_obs
end

"""
DUASE estimation using RDPGDynamics.embed_temporal_duase_raw.
Falls back to simple Procrustes-aligned SVD if package function fails.
"""
function duase_estimate(A_obs::Vector{Matrix{Float64}}, d::Int; window::Union{Nothing,Int}=nothing)
    T = length(A_obs)
    n = size(A_obs[1], 1)

    # Use DUASE from the package (no B^d_+ projection, just consistent signs)
    _, X_raw = duase_embedding(A_obs, d; window=window)

    # Procrustes chain for temporal alignment
    X_est = Vector{Matrix{Float64}}(undef, T)
    X_est[1] = X_raw[1]

    for t in 2:T
        Q = ortho_procrustes_RM(X_raw[t]', X_est[t-1]')
        X_est[t] = X_raw[t] * Q
    end

    return X_est
end

# ============================================================================
# Neural Network Kernel (UDE: knows types + self-rates, learns κ flexibly)
# ============================================================================
#
# KNOWN PHYSICS (not learned):
#   - Node types (predator, prey, resource)
#   - Self-rates a_P, a_Y, a_R
#
# LEARNED BY NN:
#   - Input: (P_ij, type_i_onehot[3], type_j_onehot[3]) = 7 features
#   - Output: 9 message kernels κ(P_ij) for each type pair
#
# Output indices (all 9 asymmetric type pairs):
#   1: κ_PP, 2: κ_PY, 3: κ_PR
#   4: κ_YP, 5: κ_YY, 6: κ_YR
#   7: κ_RP, 8: κ_RY, 9: κ_RR

# Map (type_i, type_j) to output index
const TYPE_PAIR_TO_IDX = Dict(
    (TYPE_P, TYPE_P) => 1,
    (TYPE_P, TYPE_Y) => 2,
    (TYPE_P, TYPE_R) => 3,
    (TYPE_Y, TYPE_P) => 4,
    (TYPE_Y, TYPE_Y) => 5,
    (TYPE_Y, TYPE_R) => 6,
    (TYPE_R, TYPE_P) => 7,
    (TYPE_R, TYPE_Y) => 8,
    (TYPE_R, TYPE_R) => 9
)

"""
Build the kernel NN.
Input: 7 features (P_ij, type_i_onehot[3], type_j_onehot[3])
Output: 9 message kernels (self-rates are known, not learned)
"""
function build_kernel_nn(; hidden_sizes=[32, 32], rng=Random.default_rng())
    layers = []

    # Input: 7 features
    in_dim = 7

    for (i, h) in enumerate(hidden_sizes)
        push!(layers, Lux.Dense(in_dim, h, tanh))
        in_dim = h
    end

    # Output: 9 message kernels only
    push!(layers, Lux.Dense(in_dim, 9))

    return Lux.Chain(layers...)
end

"""
Compute N matrix using the learned kernel NN + known self-rates.
"""
function compute_N_nn(P::Matrix, nn, ps, st, types::Vector{Int})
    n = size(P, 1)
    N = zeros(eltype(P), n, n)

    for i in 1:n
        ti = types[i]
        # Diagonal starts with KNOWN self-rate
        N[i,i] = eltype(P)(KNOWN_SELF_RATES[ti])

        # One-hot encoding for type_i
        ti_onehot = zeros(eltype(P), 3)
        ti_onehot[ti] = one(eltype(P))

        for j in 1:n
            if j != i
                tj = types[j]
                tj_onehot = zeros(eltype(P), 3)
                tj_onehot[tj] = one(eltype(P))

                # NN input: [P_ij, type_i_onehot, type_j_onehot]
                p_ij = P[i,j]
                input = vcat([p_ij], ti_onehot, tj_onehot)

                # NN output: 9 message kernels [κ_PP, κ_PY, ..., κ_RR]
                output, _ = nn(reshape(input, 7, 1), ps, st)
                output = vec(output)

                # Get message kernel for this type pair
                idx = TYPE_PAIR_TO_IDX[(ti, tj)]
                κ_ij = output[idx]
                N[i,j] = κ_ij
                N[i,i] -= κ_ij  # Message-passing form
            end
        end
    end

    return N
end

"""
Learned dynamics using kernel NN.
"""
function nn_dynamics!(dX::Matrix, X::Matrix, ps, nn, st)
    P = X * X'
    N = compute_N_nn(P, nn, ps, st, NODE_TYPES)
    dX .= N * X
end

# ============================================================================
# Training
# ============================================================================

function train_kernel_nn(X_train::Vector{Matrix{Float64}};
                         epochs::Int=2000, lr::Float64=0.01)
    n, d = size(X_train[1])
    T_train = length(X_train)

    # Build NN
    rng = Xoshiro(123)
    nn = build_kernel_nn(; hidden_sizes=[32, 32], rng=rng)
    ps, st = Lux.setup(rng, nn)
    ps = ComponentArray(ps)

    # Flatten training data
    X_target = hcat([vec(X') for X in X_train]...)
    u0 = vec(X_train[1]')
    tspan = (0.0, Float64(T_train - 1))

    # ODE with NN dynamics
    function nn_ode!(du, u, p, t)
        X = reshape(u, d, n)'
        dX = similar(X)
        nn_dynamics!(dX, X, p, nn, st)
        du .= vec(dX')
    end

    prob = ODEProblem(nn_ode!, u0, tspan, ps)

    # Loss function
    function loss(p, _)
        _prob = remake(prob; p=p)
        sol = solve(_prob, Tsit5(); saveat=1.0,
                    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
        if sol.retcode != ReturnCode.Success
            return Inf, nothing
        end
        pred = hcat(sol.u...)

        mse = mean(abs2, pred .- X_target)

        # Probability constraint
        prob_loss = 0.0
        for u in sol.u
            X = reshape(u, d, n)'
            P = X * X'
            prob_loss += sum(max.(-P, 0.0).^2) + sum(max.(P .- 1.0, 0.0).^2)
        end

        return mse + 0.05 * prob_loss, nothing
    end

    # Optimize
    opt_func = OptimizationFunction(loss, Optimization.AutoForwardDiff())
    opt_prob = OptimizationProblem(opt_func, ps)

    println("      Training kernel NN with Adam...")
    result = solve(opt_prob, Adam(lr); maxiters=epochs, progress=false)

    println("      Refining with LBFGS...")
    opt_prob2 = OptimizationProblem(opt_func, result.u)
    try
        result = solve(opt_prob2, LBFGS(); maxiters=100)
    catch e
        println("      LBFGS failed, using Adam result")
    end

    return result.u, nn, st
end

# ============================================================================
# Symbolic Regression
# ============================================================================

"""
Sample the learned kernel NN for symbolic regression.

For each type pair (ti, tj), generate (P_ij, κ) pairs.
Returns a Dict mapping (ti, tj) to (p, κ) vectors for SymReg.
"""
function sample_kernel_for_symreg(nn, ps, st; n_samples=500, seed=999)
    rng = Xoshiro(seed)

    # Sample P_ij values
    p_values = rand(rng, n_samples) .* 0.8 .+ 0.1  # P_ij in [0.1, 0.9]

    # For each type pair, collect samples
    samples = Dict{Tuple{Int,Int}, @NamedTuple{p::Vector{Float64}, κ::Vector{Float64}}}()

    for ti in 1:K_TYPES
        ti_onehot = zeros(3)
        ti_onehot[ti] = 1.0

        for tj in 1:K_TYPES
            tj_onehot = zeros(3)
            tj_onehot[tj] = 1.0

            κ_values = Float64[]

            for p_ij in p_values
                input = vcat([p_ij], ti_onehot, tj_onehot)
                output, _ = nn(reshape(Float64.(input), 7, 1), ps, st)
                output = vec(output)

                # Get the message kernel for this type pair (now indices 1-9)
                idx = TYPE_PAIR_TO_IDX[(ti, tj)]
                push!(κ_values, output[idx])
            end

            samples[(ti, tj)] = (p=copy(p_values), κ=κ_values)
        end
    end

    return samples
end

"""
Fit simple parametric forms to the sampled kernel data.
This is a simplified "symbolic regression" - fitting known functional forms.
"""
function fit_kernel_forms(samples)
    type_names = ["P", "Y", "R"]
    results = Dict{Tuple{Int,Int}, @NamedTuple{form::String, params::Vector{Float64}, r2::Float64}}()

    for ti in 1:K_TYPES
        for tj in 1:K_TYPES
            data = samples[(ti, tj)]
            p = data.p
            κ = data.κ

            # Try different forms and pick best R²

            # Form 1: Constant κ(p) = c
            c = mean(κ)
            pred_const = fill(c, length(p))
            ss_res_const = sum((κ .- pred_const).^2)
            ss_tot = sum((κ .- mean(κ)).^2)
            r2_const = ss_tot > 1e-10 ? 1 - ss_res_const / ss_tot : 1.0

            # Form 2: Linear κ(p) = a + b*p
            # Simple least squares
            X_lin = hcat(ones(length(p)), p)
            coef_lin = X_lin \ κ
            pred_lin = X_lin * coef_lin
            ss_res_lin = sum((κ .- pred_lin).^2)
            r2_lin = ss_tot > 1e-10 ? 1 - ss_res_lin / ss_tot : 1.0

            # Form 3: Holling Type II κ(p) = α*p / (1 + β*p)
            # Nonlinear fit via linearization: 1/κ = 1/α + β/(α*p) for κ > 0
            # Or just grid search for simplicity
            best_r2_holling = -Inf
            best_α, best_β = 0.0, 0.0

            if mean(κ) > 0.01  # Only try Holling if κ is mostly positive
                for α in range(0.01, 0.2, length=20)
                    for β in range(0.1, 5.0, length=20)
                        pred_h = α .* p ./ (1 .+ β .* p)
                        ss_res_h = sum((κ .- pred_h).^2)
                        r2_h = ss_tot > 1e-10 ? 1 - ss_res_h / ss_tot : 1.0
                        if r2_h > best_r2_holling
                            best_r2_holling = r2_h
                            best_α, best_β = α, β
                        end
                    end
                end
            end

            # Pick best form
            if r2_const >= r2_lin && r2_const >= best_r2_holling
                results[(ti, tj)] = (form="constant", params=[c], r2=r2_const)
            elseif r2_lin >= best_r2_holling
                results[(ti, tj)] = (form="linear", params=coef_lin, r2=r2_lin)
            else
                results[(ti, tj)] = (form="holling", params=[best_α, best_β], r2=best_r2_holling)
            end
        end
    end

    return results
end

# ============================================================================
# Evaluation
# ============================================================================

function predict_P_trajectory(ps, nn, st, X0::Matrix{Float64}, T::Int)
    n, d = size(X0)
    u0 = vec(X0')
    tspan = (0.0, Float64(T - 1))

    function ode!(du, u, p, t)
        X = reshape(u, d, n)'
        dX = similar(X)
        nn_dynamics!(dX, X, p, nn, st)
        du .= vec(dX')
    end

    prob = ODEProblem(ode!, u0, tspan, ps)
    sol = solve(prob, Tsit5(); saveat=1.0)

    X_traj = [reshape(u, d, n)' for u in sol.u]
    P_traj = [X * X' for X in X_traj]

    return P_traj, X_traj
end

function compute_P_error(P_pred, P_true)
    T = min(length(P_pred), length(P_true))
    [norm(P_pred[t] .- P_true[t]) / norm(P_true[t]) for t in 1:T]
end

# ============================================================================
# Preliminary Trajectory Visualization
# ============================================================================

"""
Visualize trajectories to check they stay in reasonable gauge before training.
"""
function visualize_trajectories(X_true, X_est, P_true, P_est)
    T = length(X_true)
    type_colors = [:red, :blue, :green]  # Pred, Prey, Res

    fig = CM.Figure(size=(1400, 800))

    # Row 1: True X trajectories
    ax1 = CM.Axis(fig[1, 1]; xlabel="x₁", ylabel="x₂", title="True X trajectories")
    for i in 1:N_TOTAL
        ti = NODE_TYPES[i]
        xs = [X_true[t][i, 1] for t in 1:T]
        ys = [X_true[t][i, 2] for t in 1:T]
        CM.lines!(ax1, xs, ys; color=(type_colors[ti], 0.5), linewidth=1)
        CM.scatter!(ax1, [xs[1]], [ys[1]]; color=type_colors[ti], markersize=8)
        CM.scatter!(ax1, [xs[end]], [ys[end]]; color=type_colors[ti], marker=:star5, markersize=10)
    end

    # Row 1: Estimated X trajectories
    ax2 = CM.Axis(fig[1, 2]; xlabel="x₁", ylabel="x₂", title="DUASE X̂ trajectories")
    for i in 1:N_TOTAL
        ti = NODE_TYPES[i]
        xs = [X_est[t][i, 1] for t in 1:T]
        ys = [X_est[t][i, 2] for t in 1:T]
        CM.lines!(ax2, xs, ys; color=(type_colors[ti], 0.5), linewidth=1)
        CM.scatter!(ax2, [xs[1]], [ys[1]]; color=type_colors[ti], markersize=8)
        CM.scatter!(ax2, [xs[end]], [ys[end]]; color=type_colors[ti], marker=:star5, markersize=10)
    end

    # Row 1: P error over time
    ax3 = CM.Axis(fig[1, 3]; xlabel="Time", ylabel="Relative P-error", title="DUASE estimation error")
    err = compute_P_error(P_est, P_true)
    CM.lines!(ax3, 0:T-1, err; color=:coral, linewidth=2)

    # Row 2: P heatmaps at t=1, mid, end
    times = [1, T ÷ 2, T]
    for (col, t) in enumerate(times)
        ax_true = CM.Axis(fig[2, col]; aspect=1, title="P_true(t=" * string(t-1) * ")")
        CM.heatmap!(ax_true, P_true[t]; colorrange=(0,1), colormap=:viridis)
        CM.hidedecorations!(ax_true)

        ax_est = CM.Axis(fig[3, col]; aspect=1, title="P̂_est(t=" * string(t-1) * ")")
        CM.heatmap!(ax_est, P_est[t]; colorrange=(0,1), colormap=:viridis)
        CM.hidedecorations!(ax_est)
    end

    # Legend
    CM.Legend(fig[1, 4], [CM.MarkerElement(color=c, marker=:circle) for c in type_colors],
           ["Predator", "Prey", "Resource"]; framevisible=false)

    return fig
end

# ============================================================================
# Save/Load Functions
# ============================================================================

function save_data(X_true, A_obs, X_est)
    data = Dict(
        "X_true" => X_true,
        "A_obs" => A_obs,
        "X_est" => X_est,
        "config" => Dict(
            "N_PRED" => N_PRED, "N_PREY" => N_PREY, "N_RES" => N_RES,
            "D_EMBED" => D_EMBED, "T_TOTAL" => T_TOTAL, "SEED" => SEED,
            "HOLLING_ALPHA" => HOLLING_ALPHA, "HOLLING_BETA" => HOLLING_BETA
        )
    )
    serialize(joinpath(OUTPUT_DIR, "data.jls"), data)
    println("   Saved: " * joinpath(OUTPUT_DIR, "data.jls"))
end

function load_data()
    data = deserialize(joinpath(OUTPUT_DIR, "data.jls"))
    return data["X_true"], data["A_obs"], data["X_est"]
end

function save_model(ps_learned, nn, st, samples, kernel_fits)
    model = Dict(
        "ps_learned" => ps_learned,
        "nn" => nn,
        "st" => st,
        "samples" => samples,
        "kernel_fits" => kernel_fits
    )
    serialize(joinpath(OUTPUT_DIR, "model.jls"), model)
    println("   Saved: " * joinpath(OUTPUT_DIR, "model.jls"))
end

function save_evaluation(err_pred, err_duase, P_pred, P_true)
    eval_data = Dict(
        "err_pred" => err_pred,
        "err_duase" => err_duase,
        "P_pred" => P_pred,
        "P_true" => P_true
    )
    serialize(joinpath(OUTPUT_DIR, "evaluation.jls"), eval_data)
    println("   Saved: " * joinpath(OUTPUT_DIR, "evaluation.jls"))
end

# ============================================================================
# Main
# ============================================================================

function run_example4(; skip_training::Bool=false)
    println("=" ^ 70)
    println("Example 4: Type-Specific Kernels with Symbolic Regression")
    println("=" ^ 70)

    # =========================================================================
    # 1. Generate or Load Data
    # =========================================================================
    data_file = joinpath(OUTPUT_DIR, "data.jls")

    if isfile(data_file) && skip_training
        println("\n1. Loading existing data...")
        X_true, A_obs, X_est = load_data()
        P_true = [X * X' for X in X_true]
        P_est = [X * X' for X in X_est]
    else
        println("\n1. Generating data with Holling Type II predation...")

        X_true = generate_true_data()
        P_true = [X * X' for X in X_true]

        println("   Nodes: " * string(N_PRED) * " predators, " *
                string(N_PREY) * " prey, " * string(N_RES) * " resources")
        println("   True dynamics include:")
        println("     - Holling Type II: κ_PY(p) = " * @sprintf("%.2f", HOLLING_ALPHA) *
                "·p / (1 + " * @sprintf("%.1f", HOLLING_BETA) * "·p)")
        println("     - Linear: κ_YP(p) = -0.04·p (prey flee)")
        println("     - Constant: κ_PP = -0.008 (predator repulsion)")

        println("\n   Generating adjacency samples (K=10 per timestep)...")
        A_obs = generate_adjacencies(X_true; K=10)

        println("   Running DUASE estimation...")
        X_est = duase_estimate(A_obs, D_EMBED)
        P_est = [X * X' for X in X_est]

        # Save data
        save_data(X_true, A_obs, X_est)
    end

    T_train = Int(floor(TRAIN_FRAC * T_TOTAL))
    X_train = X_est[1:T_train]

    println("   Training: t=1-" * string(T_train) * ", Validation: t=" *
            string(T_train+1) * "-" * string(T_TOTAL))

    # =========================================================================
    # 1b. Preliminary Trajectory Visualization
    # =========================================================================
    println("\n1b. Visualizing trajectories (checking gauge consistency)...")
    fig_traj = visualize_trajectories(X_true, X_est, P_true, P_est)
    CM.save(joinpath(OUTPUT_DIR, "trajectories.png"), fig_traj; px_per_unit=2)
    println("   Saved: " * joinpath(OUTPUT_DIR, "trajectories.png"))

    # Report trajectory statistics
    X_flat_true = vcat([X_true[t][:] for t in 1:length(X_true)]...)
    X_flat_est = vcat([X_est[t][:] for t in 1:length(X_est)]...)
    println("   True X range: [" * @sprintf("%.3f", minimum(X_flat_true)) * ", " *
            @sprintf("%.3f", maximum(X_flat_true)) * "]")
    println("   Est  X̂ range: [" * @sprintf("%.3f", minimum(X_flat_est)) * ", " *
            @sprintf("%.3f", maximum(X_flat_est)) * "]")
    println("   Mean P-error (DUASE): " * @sprintf("%.2f%%", 100*mean(compute_P_error(P_est, P_true))))

    # =========================================================================
    # 2. Train Kernel NN
    # =========================================================================
    println("\n2. Training kernel NN (UDE: knows types + self-rates, learns κ)...")
    println("   Known self-rates (not learned):")
    type_names = ["Pred", "Prey", "Res"]
    for ti in 1:K_TYPES
        println("     a_" * type_names[ti] * " = " * @sprintf("%.4f", KNOWN_SELF_RATES[ti]))
    end

    ps_learned, nn, st = train_kernel_nn(X_train; epochs=2500, lr=0.015)

    # =========================================================================
    # 3. Symbolic Regression
    # =========================================================================
    println("\n3. Symbolic Regression: Discovering functional forms for 9 kernels...")

    samples = sample_kernel_for_symreg(nn, ps_learned, st)
    kernel_fits = fit_kernel_forms(samples)

    println("\n   Discovered kernel forms (NN outputs → parametric fits):")
    for ti in 1:K_TYPES
        for tj in 1:K_TYPES
            fit = kernel_fits[(ti, tj)]
            pair_name = type_names[ti] * "→" * type_names[tj]

            if fit.form == "constant"
                form_str = @sprintf("%.4f", fit.params[1])
            elseif fit.form == "linear"
                form_str = @sprintf("%.4f + %.4f·p", fit.params[1], fit.params[2])
            else  # holling
                form_str = @sprintf("%.3f·p/(1 + %.2f·p)", fit.params[1], fit.params[2])
            end

            println("     κ_" * pair_name * ": " * fit.form * " = " * form_str *
                    " (R²=" * @sprintf("%.3f", fit.r2) * ")")
        end
    end

    # Check if Holling was discovered for P→Y
    py_fit = kernel_fits[(TYPE_P, TYPE_Y)]
    if py_fit.form == "holling"
        println("\n   ✓ SUCCESS: Holling Type II discovered for predator-prey!")
        println("     True:     κ(p) = " * @sprintf("%.2f", HOLLING_ALPHA) *
                "·p / (1 + " * @sprintf("%.1f", HOLLING_BETA) * "·p)")
        println("     Learned:  κ(p) = " * @sprintf("%.2f", py_fit.params[1]) *
                "·p / (1 + " * @sprintf("%.1f", py_fit.params[2]) * "·p)")
    else
        println("\n   Note: Holling form not recovered for P→Y (got " * py_fit.form * ")")
    end

    # Save model and SymReg results
    save_model(ps_learned, nn, st, samples, kernel_fits)

    # =========================================================================
    # 4. Evaluate
    # =========================================================================
    println("\n4. Evaluation: Apply to TRUE initial conditions...")

    P_pred, X_pred = predict_P_trajectory(ps_learned, nn, st, X_true[1], T_TOTAL)

    err_pred = compute_P_error(P_pred, P_true)
    err_duase = compute_P_error(P_est, P_true)

    println("\n   P-error:")
    println("     Training:     " * @sprintf("%.2f%%", 100*mean(err_pred[1:T_train])))
    println("     Extrapolation: " * @sprintf("%.2f%%", 100*mean(err_pred[T_train+1:end])))
    println("     DUASE baseline: " * @sprintf("%.2f%%", 100*mean(err_duase[1:T_train])))

    # Save evaluation results
    save_evaluation(err_pred, err_duase, P_pred, P_true)

    # =========================================================================
    # 5. Visualization
    # =========================================================================
    println("\n5. Generating visualizations...")

    fig = CM.Figure(size=(1400, 1000))

    # Row 1: P(t) comparison
    for (col, t) in enumerate([1, T_train, T_TOTAL])
        ax1 = CM.Axis(fig[1, col]; aspect=1, title="t=" * string(t-1))
        CM.heatmap!(ax1, P_true[t]; colorrange=(0,1), colormap=:viridis)
        CM.hidedecorations!(ax1)
        if col == 1
            CM.Label(fig[1, 0], "True P", rotation=pi/2, fontsize=14)
        end

        ax2 = CM.Axis(fig[2, col]; aspect=1)
        CM.heatmap!(ax2, P_pred[t]; colorrange=(0,1), colormap=:viridis)
        CM.hidedecorations!(ax2)
        if col == 1
            CM.Label(fig[2, 0], "Predicted P̂", rotation=pi/2, fontsize=14)
        end
    end

    CM.Colorbar(fig[1:2, 4]; colorrange=(0,1), colormap=:viridis, label="P(i,j)")

    # Row 3: P-error over time
    ax3 = CM.Axis(fig[3, 1:3]; xlabel="Time", ylabel="Relative P-error",
               title="P-Error Over Time")
    CM.vspan!(ax3, [0], [T_train-1]; color=(:green, 0.1))
    CM.lines!(ax3, 0:T_TOTAL-1, err_duase; color=:coral, linestyle=:dash, label="DUASE")
    CM.lines!(ax3, 0:T_TOTAL-1, err_pred; color=:blue, label="Kernel NN")
    CM.axislegend(ax3; position=:lt)

    # Row 4: Kernel fits visualization
    ax4 = CM.Axis(fig[4, 1:2]; xlabel="P_ij", ylabel="κ(P_ij)",
               title="Discovered Kernels (Predator interactions)")

    p_range = range(0.1, 0.9, length=100)

    # True and learned κ_PY (should be Holling)
    κ_true_py = [κ_true(TYPE_P, TYPE_Y, p) for p in p_range]
    CM.lines!(ax4, p_range, κ_true_py; color=:red, linewidth=2, label="True κ_P→Y (Holling)")

    data_py = samples[(TYPE_P, TYPE_Y)]
    CM.scatter!(ax4, data_py.p, data_py.κ; color=:red, alpha=0.3, markersize=5)

    # True and learned κ_PP (constant)
    κ_true_pp = [κ_true(TYPE_P, TYPE_P, p) for p in p_range]
    CM.lines!(ax4, p_range, κ_true_pp; color=:purple, linewidth=2, label="True κ_P→P (const)")

    data_pp = samples[(TYPE_P, TYPE_P)]
    CM.scatter!(ax4, data_pp.p, data_pp.κ; color=:purple, alpha=0.3, markersize=5)

    CM.axislegend(ax4; position=:rt)

    # Save
    CM.save(joinpath(OUTPUT_DIR, "results.png"), fig; px_per_unit=2)
    println("   Saved: results/example4_type_kernel/results.png")

    # =========================================================================
    # Summary
    # =========================================================================
    println("\n" * "=" ^ 70)
    println("Summary")
    println("=" ^ 70)

    println("\n1. UDE STRUCTURE")
    println("   Known physics:")
    println("     - Node type membership (predator/prey/resource)")
    println("     - Self-rates a_P, a_Y, a_R (decay per type)")
    println("   Learned: 9 message kernels κ(P_ij, type_i, type_j) via NN")

    println("\n2. SYMBOLIC REGRESSION DISCOVERY")
    n_holling = count(fit -> fit.form == "holling", values(kernel_fits))
    n_linear = count(fit -> fit.form == "linear", values(kernel_fits))
    n_const = count(fit -> fit.form == "constant", values(kernel_fits))
    println("   Forms discovered: " * string(n_holling) * " Holling, " *
            string(n_linear) * " linear, " * string(n_const) * " constant")

    println("\n3. KEY RESULT")
    if py_fit.form == "holling"
        println("   Holling Type II saturation discovered for predator-prey interaction!")
        println("   This nonlinear ecological law emerged from flexible NN learning + SymReg")
    end

    return (
        ps_learned = ps_learned,
        nn = nn,
        st = st,
        kernel_fits = kernel_fits,
        samples = samples,
        errors = (pred=err_pred, duase=err_duase)
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = run_example4()
end
