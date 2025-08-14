import MLJ: machine, fit!, predict, report
using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
      OptimizationOptimisers, Random, Plots
using JSON
# Dataset with two named features:
# Aim: f(x)=x'
# Ie NN(x)=y
using Serialization
import SymbolicRegression: Options, equation_search
u = deserialize("models/1_community_oscillation/big-NN-07-08-2025-L.jls")
data = JSON.parsefile("data/1_community_oscillation.json")

datasize = 10
tspan = (0.0f0, 9.0f0)

n = 5
d = 2
u0 = reshape(hcat(data["L_series"][1]...), n*d, 1)

L_data = [Float64.([L[1]; L[2]])[:,:] for L in data["L_series"]]


tsteps = range(tspan[1], tspan[2]; length = datasize)


rng = Xoshiro(0)

dudt2 = Chain(x -> x, Dense(n*d, 256, celu), Dense(256, 128, celu), Dense(128, 128, celu), Dense(128, n*d))
p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(); saveat = tsteps)



X = L_data
test = dudt2(X[1], u, st)

Y = [dudt2(x, u, st)[1] for x in X]

X̂ = hcat(X...)
Ŷ = hcat(Y...)

options = Options(
    populations=25,
    binary_operators=[+, -, *],
    unary_operators=[cos, sin],
    should_optimize_constants=true,
    should_simplify=true,
)

hall_of_fame = equation_search(
    X̂, Ŷ, niterations=50, options=options,
    parallelism=:multithreading
)

import SymbolicRegression: calculate_pareto_frontier

dominating = calculate_pareto_frontier.(hall_of_fame)
trees = [d[5].tree for d in dominating]


import SymbolicRegression: eval_tree_array

function dudt2_symb(X)::Matrix{Float64}
    output = eval_tree_array.(trees, [X], [options])
    Y = collect(hcat([o[1] for o in output]...)')
    return Y

end

dudt2_symb(X[10])
Ŷ[:,10]


using DifferentialEquations
f(u, p, t) = dudt2_symb(u)
u0 = X̂[:,1:1]
tspan = (0.0, 29.0)
tsteps = range(tspan[1], tspan[2]; length = 30)

prob = ODEProblem(f, u0, tspan)
sol = solve(prob, Tsit5(), saveat = tsteps)
soln = Array(sol)[:, 1, :]

trace = soln[1,:]

plot(sol.t[1:15],trace[1:15], linewidth = 5, title = "Solution to the linear ODE with a thick line",
    xaxis = "Time (t)", yaxis = "u(t) (in μm)", label = "My Thick Line!") # legend=false

