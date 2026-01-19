"""
Example 5: Gauge-Consistent N(P)X vs Standard Neural ODE

Compares the standard Neural ODE architecture with gauge-consistent formulations:
- Standard NN: Ẋ = f(X) where f is a generic neural network
- Polynomial N(P)X: Ẋ = (α₀I + α₁P + α₂P²)X with learned αₖ
- Kernel N(P)X: Ẋ = N(P)X where N_ij = κ(P_ij) with learned kernel κ

The N(P)X form is theoretically motivated:
1. Symmetric N eliminates gauge ambiguity (no invisible rotational dynamics)
2. Polynomial form is highly parsimonious (3 parameters vs thousands)
3. Structure preserves probability interpretation

This script generates data using pairwise dynamics (which has the N(P)X form)
to provide a fair comparison where the correct inductive bias should help.
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using RDPGDynamics
using LinearAlgebra
using Random
using OrdinaryDiffEq
using ComponentArrays
using CairoMakie
using Printf

# ============================================================================
# Generate Synthetic Data
# ============================================================================

"""
Generate pairwise dynamics data: Ẋ = (αI + βP)X

This is exactly the polynomial N(P)X form with degree 1, so the polynomial
architecture should have perfect inductive bias.
"""
function generate_pairwise_data(;
    n::Int=30,
    d::Int=2,
    α::Float64=-0.02,  # Diagonal term (slight contraction)
    β::Float64=0.001,  # Pairwise attraction
    T::Float64=30.0,
    dt::Float64=1.0,
    seed::Int=42
)
    rng = Xoshiro(seed)

    # Initialize embeddings in B_d^+ (positive orthant of unit ball)
    X0 = rand(rng, n, d) .* 0.5 .+ 0.1  # Start in [0.1, 0.6]

    # Normalize rows to stay in unit ball
    for i in 1:n
        row_norm = norm(X0[i, :])
        if row_norm > 0.9
            X0[i, :] .*= 0.9 / row_norm
        end
    end

    # ODE dynamics
    function pairwise_ode!(dX, X, p, t)
        P = X * X'  # Probability matrix
        N = α * I(n) + β * P
        dX .= N * X
    end

    # Solve ODE
    prob = ODEProblem(pairwise_ode!, X0, (0.0, T))
    sol = solve(prob, Tsit5(); saveat=dt)

    # Extract trajectory as vector of matrices
    L_data = [Matrix{Float64}(sol.u[i]) for i in 1:length(sol.t)]

    return L_data, (α=α, β=β)
end

# ============================================================================
# Standard Neural ODE (baseline)
# ============================================================================

"""
Train standard Neural ODE for comparison.
"""
function train_standard_nn(L_data::Vector{Matrix{Float64}};
                           n::Int, d::Int, epochs::Int=500, verbose::Bool=true)
    config = RDPGConfig(
        n=n, d=d,
        hidden_sizes=[64, 64, 32],
        datasize=length(L_data),
        epochs_adam=epochs,
        epochs_bfgs=0,
        learning_rate=0.01,
        constraint_weight=1.0,
        seed=1234
    )

    if verbose
        println("\nTraining Standard Neural ODE")
        println("  Parameters: ~" * string(n*d*64 + 64*64 + 64*32 + 32*n*d))
    end

    params = train_rdpg_node(L_data, config; verbose=verbose)
    return params, config
end

# ============================================================================
# Main Comparison
# ============================================================================

function run_comparison(;
    n::Int=30,
    d::Int=2,
    epochs::Int=500,
    save_results::Bool=true
)
    println("=" ^ 60)
    println("Example 5: Gauge-Consistent N(P)X vs Standard Neural ODE")
    println("=" ^ 60)

    # Generate data
    println("\n1. Generating pairwise dynamics data...")
    α_true, β_true = -0.02, 0.001
    L_data, true_params = generate_pairwise_data(n=n, d=d, α=α_true, β=β_true, T=30.0)
    println("  n=" * string(n) * " nodes, d=" * string(d) * " dimensions")
    println("  T=" * string(length(L_data)) * " timesteps")
    println("  True parameters: α=" * string(α_true) * ", β=" * string(β_true))

    datasize = length(L_data)
    u0 = Float32.(vec(L_data[1]'))

    # Results storage
    results = Dict{String, NamedTuple}()

    # =========================================================================
    # 1. Standard Neural ODE
    # =========================================================================
    println("\n2. Training models...\n")
    println("  [1/3] Standard Neural ODE")
    println("  " * "-" ^ 40)

    config_std = RDPGConfig(
        n=n, d=d,
        hidden_sizes=[64, 64, 32],
        datasize=datasize,
        epochs_adam=epochs,
        epochs_bfgs=0,
        learning_rate=0.01,
        constraint_weight=1.0,
        seed=1234
    )

    # Count parameters
    std_params = 2*64 + 64 + 64*64 + 64 + 64*32 + 32 + 32*(n*d) + n*d
    println("  Parameters: " * string(std_params))

    params_std = train_rdpg_node(L_data, config_std; verbose=true)

    # =========================================================================
    # 2. Polynomial N(P)X (degree 1 matches true dynamics)
    # =========================================================================
    println("\n  [2/3] Polynomial N(P)X (degree=1)")
    println("  " * "-" ^ 40)

    config_poly = PolynomialNConfig(
        n=n, d=d,
        degree=1,  # N = α₀I + α₁P (matches true dynamics!)
        datasize=datasize,
        learning_rate=0.01,
        epochs=epochs,
        constraint_weight=1.0,
        seed=1234
    )

    poly_params_count = config_poly.degree + 1
    println("  Parameters: " * string(poly_params_count) * " (α₀, α₁)")

    prob_poly, params_poly_init = build_polynomial_N_ode(config_poly)
    params_poly = train_gauge_ude(L_data, prob_poly, params_poly_init;
                                   config=config_poly, verbose=true)

    println("  Learned coefficients: α₀=" * @sprintf("%.5f", params_poly.α[1]) *
            ", α₁=" * @sprintf("%.5f", params_poly.α[2]))
    println("  True coefficients:    α=" * @sprintf("%.5f", α_true) *
            ", β=" * @sprintf("%.5f", β_true))

    # =========================================================================
    # 3. Kernel N(P)X
    # =========================================================================
    println("\n  [3/3] Kernel N(P)X")
    println("  " * "-" ^ 40)

    config_kernel = KernelNConfig(
        n=n, d=d,
        kernel_hidden=[16, 16],
        datasize=datasize,
        learning_rate=0.01,
        epochs=epochs,
        constraint_weight=1.0,
        seed=1234
    )

    # Count kernel NN parameters
    kernel_params_count = 1*16 + 16 + 16*16 + 16 + 16*1 + 1  # kernel network
    kernel_params_count += 1*8 + 8 + 8*1 + 1  # diagonal network
    println("  Parameters: " * string(kernel_params_count))

    prob_kernel, params_kernel_init, _ = build_kernel_N_ode(config_kernel)
    params_kernel = train_gauge_ude(L_data, prob_kernel, params_kernel_init;
                                     config=config_kernel, verbose=true)

    # =========================================================================
    # 3. Evaluate all models
    # =========================================================================
    println("\n3. Evaluating predictions...\n")

    # Compute prediction errors
    tspan = (0.0f0, Float32(datasize - 1))
    tsteps = range(tspan[1], tspan[2]; length=datasize)

    # Ground truth
    u_true = Float32.(hcat([vec(L') for L in L_data]...))

    # Standard NN prediction
    prob_std, _ = build_neural_ode(config_std)
    prob_std = remake(prob_std; u0=u0, tspan=tspan)
    # Need to get the chain and state from training - for now use predict_trajectory
    pred_std = predict_trajectory(prob_std, params_std, u0, datasize)
    mse_std = sum(abs2, u_true .- pred_std) / length(u_true)

    # Polynomial prediction
    prob_poly_pred = remake(prob_poly; u0=u0, tspan=tspan)
    pred_poly = predict_gauge_trajectory(prob_poly_pred, params_poly, u0, datasize)
    mse_poly = sum(abs2, u_true .- pred_poly) / length(u_true)

    # Kernel prediction
    prob_kernel_pred = remake(prob_kernel; u0=u0, tspan=tspan)
    pred_kernel = predict_gauge_trajectory(prob_kernel_pred, params_kernel, u0, datasize)
    mse_kernel = sum(abs2, u_true .- pred_kernel) / length(u_true)

    # Probability constraint violations
    function count_violations(pred, n, d)
        violations = 0
        for col in eachcol(pred)
            X = reshape(col, d, n)'
            P = X * X'
            violations += sum(P .< -0.01) + sum(P .> 1.01)
        end
        return violations
    end

    viol_std = count_violations(pred_std, n, d)
    viol_poly = count_violations(pred_poly, n, d)
    viol_kernel = count_violations(pred_kernel, n, d)

    # Print results
    println("  " * "-" ^ 55)
    println("  Model                  | MSE        | Params | Violations")
    println("  " * "-" ^ 55)
    @printf("  Standard Neural ODE    | %.6f   | %5d  | %d\n", mse_std, std_params, viol_std)
    @printf("  Polynomial N(P)X (d=1) | %.6f   | %5d  | %d\n", mse_poly, poly_params_count, viol_poly)
    @printf("  Kernel N(P)X           | %.6f   | %5d  | %d\n", mse_kernel, kernel_params_count, viol_kernel)
    println("  " * "-" ^ 55)

    # =========================================================================
    # 4. Visualizations
    # =========================================================================
    println("\n4. Generating visualizations...")

    fig = Figure(size=(1200, 800))

    # Row 1: Trajectory comparisons (first 3 nodes, dimension 1)
    ax1 = Axis(fig[1, 1], title="Node 1 Trajectory", xlabel="Time", ylabel="X₁")
    ax2 = Axis(fig[1, 2], title="Node 2 Trajectory", xlabel="Time", ylabel="X₁")
    ax3 = Axis(fig[1, 3], title="Node 3 Trajectory", xlabel="Time", ylabel="X₁")

    times = collect(0:datasize-1)

    for (ax, node_idx) in zip([ax1, ax2, ax3], [1, 2, 3])
        # Extract trajectories for this node (first dimension)
        true_traj = [u_true[(node_idx-1)*d + 1, t] for t in 1:datasize]
        std_traj = [pred_std[(node_idx-1)*d + 1, t] for t in 1:datasize]
        poly_traj = [pred_poly[(node_idx-1)*d + 1, t] for t in 1:datasize]
        kernel_traj = [pred_kernel[(node_idx-1)*d + 1, t] for t in 1:datasize]

        lines!(ax, times, true_traj, color=:black, linewidth=2, label="True")
        lines!(ax, times, std_traj, color=:red, linewidth=1.5, linestyle=:dash, label="Standard NN")
        lines!(ax, times, poly_traj, color=:blue, linewidth=1.5, linestyle=:dot, label="Polynomial")
        lines!(ax, times, kernel_traj, color=:green, linewidth=1.5, linestyle=:dashdot, label="Kernel")
    end

    # Legend
    Legend(fig[1, 4], ax1, framevisible=false)

    # Row 2: Probability matrix comparisons at final time
    t_final = datasize

    X_true_final = reshape(u_true[:, t_final], d, n)'
    P_true_final = X_true_final * X_true_final'

    X_std_final = reshape(pred_std[:, t_final], d, n)'
    P_std_final = X_std_final * X_std_final'

    X_poly_final = reshape(pred_poly[:, t_final], d, n)'
    P_poly_final = X_poly_final * X_poly_final'

    X_kernel_final = reshape(pred_kernel[:, t_final], d, n)'
    P_kernel_final = X_kernel_final * X_kernel_final'

    ax_p1 = Axis(fig[2, 1], title="True P(T)", aspect=1)
    ax_p2 = Axis(fig[2, 2], title="Standard NN P(T)", aspect=1)
    ax_p3 = Axis(fig[2, 3], title="Polynomial P(T)", aspect=1)
    ax_p4 = Axis(fig[2, 4], title="Kernel P(T)", aspect=1)

    heatmap!(ax_p1, P_true_final, colorrange=(0, 1))
    heatmap!(ax_p2, P_std_final, colorrange=(0, 1))
    heatmap!(ax_p3, P_poly_final, colorrange=(0, 1))
    heatmap!(ax_p4, P_kernel_final, colorrange=(0, 1))

    # Row 3: Error heatmaps
    ax_e1 = Axis(fig[3, 1], title="Reference", aspect=1)
    ax_e2 = Axis(fig[3, 2], title="Standard NN Error", aspect=1)
    ax_e3 = Axis(fig[3, 3], title="Polynomial Error", aspect=1)
    ax_e4 = Axis(fig[3, 4], title="Kernel Error", aspect=1)

    error_std = abs.(P_std_final .- P_true_final)
    error_poly = abs.(P_poly_final .- P_true_final)
    error_kernel = abs.(P_kernel_final .- P_true_final)

    max_err = max(maximum(error_std), maximum(error_poly), maximum(error_kernel), 0.1)

    heatmap!(ax_e1, zeros(n, n), colorrange=(0, max_err))
    heatmap!(ax_e2, error_std, colorrange=(0, max_err))
    heatmap!(ax_e3, error_poly, colorrange=(0, max_err))
    heatmap!(ax_e4, error_kernel, colorrange=(0, max_err))

    # Add colorbar
    Colorbar(fig[3, 5], limits=(0, max_err), label="|P_pred - P_true|")

    # Save figure
    output_dir = joinpath(dirname(@__DIR__), "outputs")
    mkpath(output_dir)
    output_path = joinpath(output_dir, "example5_gauge_comparison.png")
    save(output_path, fig, px_per_unit=2)
    println("  Saved: " * output_path)

    # =========================================================================
    # 5. Parameter recovery analysis (polynomial model)
    # =========================================================================
    println("\n5. Parameter Recovery Analysis (Polynomial Model)")
    println("  " * "-" ^ 40)
    println("  True dynamics: Ẋ = (αI + βP)X")
    println("  Learned:       Ẋ = (α₀I + α₁P)X")
    println()
    @printf("  α (true): %.5f  |  α₀ (learned): %.5f  |  error: %.2f%%\n",
            α_true, params_poly.α[1], 100*abs(params_poly.α[1] - α_true)/abs(α_true))
    @printf("  β (true): %.5f  |  α₁ (learned): %.5f  |  error: %.2f%%\n",
            β_true, params_poly.α[2], 100*abs(params_poly.α[2] - β_true)/abs(β_true))

    # Summary
    println("\n" * "=" ^ 60)
    println("Summary")
    println("=" ^ 60)
    println("\nKey findings:")
    println("1. Polynomial N(P)X recovers true parameters with " *
            @sprintf("%.1f%%", 100 - 100*abs(params_poly.α[1] - α_true)/abs(α_true)) * " accuracy")
    println("2. Parameter reduction: " * string(std_params) * " → " * string(poly_params_count) *
            " (" * @sprintf("%.0f", 100*(1 - poly_params_count/std_params)) * "% fewer)")
    println("3. MSE comparison: Standard=" * @sprintf("%.4f", mse_std) *
            ", Polynomial=" * @sprintf("%.4f", mse_poly))

    return (
        mse=(standard=mse_std, polynomial=mse_poly, kernel=mse_kernel),
        params_learned=(polynomial=params_poly, kernel=params_kernel),
        params_true=true_params
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_comparison(n=30, d=2, epochs=500)
end
