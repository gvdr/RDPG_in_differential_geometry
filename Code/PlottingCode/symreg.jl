
using SymbolicUtils
using DelimitedFiles
using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using DataDrivenSparse
using LinearAlgebra

include("../_Modular Functions/helperFunctions.jl")
include("getSymbRegEqn.jl")

sltn = readdlm("./Code/Solutions/$net_name big net.csv", ',')
sltnt = readdlm("./Code/Solutions/$net_name big net test only.csv", ',')
grad_sltn = sltn[:,2:end].-sltn[:,1:end-1]



test_range = eachindex(1.0:0.01:Float64(datasize))


train_data = withoutNode(t_data,1)
length(train_data)





function dists(u,t)
    M = train_data[t]
    
    subtract_func(m) = m-u
    direction_vecs = [subtract_func(m) for m in eachrow(M)]

    û = vcat(partialsort(direction_vecs,1:k, by=x->sum(abs2, x))...)

end

iter_sltn = [sltn[:,i] for i in 1:size(sltn)[2]]
data = dists.(iter_sltn, 1.0:0.01:Float64(datasize))
data = hcat(data...)

# true_sltn = targetNode(t_data,1).AL[1,:,1:datasize]
# true_iter_sltn = [true_sltn[:,i] for i in 1:size(true_sltn)[2]]
# true_data = dists.(true_iter_sltn, 1.0:Float64(datasize))
# true_data = hcat(true_data[1:end]...)

# true_data[1]

# using DataDrivenDiffEq
# using ModelingToolkit
# using LinearAlgebra
# using DataDrivenSparse


# problem = ContinuousDataDrivenProblem(data[:,1:end-1], grad_sltn)


# @variables u[1:45]

# basis = Basis(monomial_basis(u, 2), u)
# println(basis) # hide

# res = solve(problem, basis, STLSQ())
# println(res)

# println(get_basis(res))
# res(data)

# system = get_basis(res)
# params = get_parameter_map(system)
# println(system) # hide
# println(params) # hide

# ODEProblem(true_data, true_sltn; U=system, p=params)
# remake(problem,p=params)

# println(system(data,problem))

# using Plots
# Plots.plot()

# get_results(res)
# remake(res.internal_problem; u0=[0 0 0])

# og_data = train_data[1:datasize].AL





options = SymbolicRegression.Options(
        binary_operators=[+, *, -],
        npopulations=50
    )

dominatings = getSymbRegEqn(data[:,1:end-1], grad_sltn, options);

eqns = [
    dominatings[i][end].tree
    for i in 1:dims[2]
]
function next_step(u0, t, eq)
    
    d = dists(u0, t+datasize)[:,:]

    u0+eq(d)
end

sltn1 = zeros(Float64, (dims[2],length(1.0:0.01:size(sltnt)[2])))
global u0 = targetNode(t_data,1)[1+datasize]
sltn1[:,1] = u0[:AL]
iter_eq(d) = vcat([eqns[i](d) for i in 1:dims[2]]...)


for t in 2:size(sltn1)[2]
    sltn1[:,t] = next_step(sltn1[:,t-1], 1+(t-1)*0.01, iter_eq)
end
next_step(sltn1[:,2], 1+(2)*0.01, iter_eq)

sltn_sym_reg = sltn1[:, 1:100:size(sltn1)[2]]
# eqns = simplify.(PolyForm.(simplify.(node_to_symbolic.(eqns, [options]))))


# using GLM, DataFrames,


# dfs = [DataFrame(["x$i"=>data[i,1:end-1] for i in 1:size(data)[1]]...) for _ in 1:dims[2]]
# for i in 1:dims[2]
#     dfs[i][!,"Y"] = grad_sltn[i,:]
# end
# eqns_symb = [PolyForm(simplify(node_to_symbolic(eqn, options))) for eqn in eqns]
# simplify(eqns[2])

# length(test_data)
# grad_test_


# formulae = [
#     @formula(Y~x[34]&x[24]&x[30]+x34&x24+x34&x30+x20+x25+x34),
#     @formula(Y~x17&x36^2+x36^3+x36^2),
#     @formula(Y~x15&x12&((0.71219-x6)^(-1))+x12&x34&((0.71219-x6)^(-1))+ x18&x34&((0.71219-x6)^(-1))+x18&x15&((0.71219-x6)^(-1))+x22&x15&((0.71219-x6)^(-1))+x22&x34&((0.71219-x6)^(-1)))
# ]

# regressed = [lm(formulae[i], dfs[i]) for i in 1:dims[2]] 
# pred_grad = hcat(predict.(regressed)...)
# global u0 = targetNode(t_data,1)[1+datasize]

# pred_points = [u0[:AL]]

# for grad in eachrow(pred_grad)
#     push!(pred_points, pred_points[end]+grad')
# end

# pred_points = vcat(pred_points...)'




