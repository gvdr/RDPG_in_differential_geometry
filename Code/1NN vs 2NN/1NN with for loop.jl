using CSV, DataFrames, LinearAlgebra, Revise, CUDA
using DifferentialEquations, Lux, SciMLSensitivity, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL, Optimisers
using Random
using Zygote
import DiffEqFlux: NeuralODE

# CUDA.allowscalar(false)
# User needs to provide:
#   u0
#   datasize
#   tspan
#   ode_data
#   input_data_length (length of one input vector)
#   output_data_length

# Things that are important:
# embedding dim, initial vec for SVD, amount of data
CUDA.allowscalar(false)
include("../_Modular Functions/helperFunctions.jl")
include("../_Modular Functions/loadGlobalConstants.jl")
include("../_Modular Functions/constructNN.jl")
include("../_Modular Functions/NODEproblem.jl")

global_consts("longTail", (182,2))
device = Lux.cpu
rng = Random.default_rng()
nn = Lux.Chain(x -> x,
      Lux.Dense(input_data_length, 64, tanh),
      Lux.Dense(64, 8, tanh),
      Lux.Dense(8, 8, tanh),
      Lux.Dense(8, output_data_length))

p, st = Lux.setup(rng, nn) 
p = p |> ComponentArray |> device
st = st |> device

function dudt_pred(u,p,t)
  
  M = train_data[t]
  subtract_func(m) = m.-u
  direction_vecs = subtract_func.(eachrow(M))
  û = vcat(partialsort(direction_vecs,1:k, by=x->sum(abs2, x))...)
  nn(û, p, st)[1]
end


dudt_pred_(u,p,t) = dudt_pred(u,p,t)

# prob_neuralode = ODEProblem{false}(, u0, tspan, p)
prob_neuralode = ODEProblem(dudt_pred_, u0, tspan, p)




TNode_data = targetNode(t_data[1:datasize],1)



# temp_func(i) = exp(-i/5+2)
# temp_func(i) = 0.0001
# result = Optimization.solve(optprob,
#                             SimulatedAnnealing(),
#                             callback = callback,
#                             maxiters = 50)
# optprob = remake(optprob, u0=result.u)
# # println("ping")

adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x,p)->loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p)
#optprob = remake(optprob, u0=result.u)
result = Optimization.solve(optprob,
                            Optimisers.Adam(0.001, (0.9, 0.99)),
                            callback = callback,
                            maxiters = 100)
optprob = remake(optprob, u0=result.u)
println("ping")


result = Optimization.solve(optprob,
                            Optimisers.Adam(0.0000001, (0.0, 0.0)),
                            callback = callback,
                            maxiters = 100)
optprob = remake(optprob, u0=result.u)
println("ping")


result = Optimization.solve(optprob,
                            Optim.BFGS(initial_stepnorm=0.01),
                            callback = callback,
                            maxiters = 50)
optprob = remake(optprob, u0=result.u)
println("ping")




# global train_data = withoutNode(t_data,1)
global tsteps = tspan[1]:0.01:tspan[2]
prob_neuralode = ODEProblem{false}(dudt_pred_, u0, tspan, p)

function predict_neuralode(θ)
  Array(solve(prob_neuralode, Tsit5(), saveat = tsteps, p=θ))#|>Lux.gpu
end
# prob_neuralode, p, st, nn = constructNN();
optprob = NODEproblem();
sol = predict_neuralode(result.u)

# loss_neuralode(result.u)
using DelimitedFiles

writedlm("./Code/Solutions/$net_name big net.csv", sol, ",")

function predict_neuralode(θ)
  Array(solve(prob_neuralode, Tsit5(), saveat = tsteps, p=θ))#|>Lux.gpu
end

global train_data = withoutNode(t_data[1+datasize:end],1)
global u0 = vec(targetNode(t_data,1)[1+datasize][:AL])
global tspan = (1.0, Float64(length(t_data[1+datasize:end])))
global tsteps = range(tspan[1], tspan[2], length = length(train_data))
prob_neuralode = ODEProblem{false}(dudt_pred_, u0, tspan, p)
optprob = NODEproblem();
sol = predict_neuralode(result.u)

using DelimitedFiles

writedlm("./Code/Solutions/$net_name big net test only.csv", sol, ",")


