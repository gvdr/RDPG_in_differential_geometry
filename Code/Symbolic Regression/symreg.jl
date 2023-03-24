using SymbolicRegression
using DelimitedFiles
include("../_Modular Functions/helperFunctions.jl")

sltn = readdlm("./Code/Solutions/$net_name.csv", ',')
sltnt = readdlm("./Code/Solutions/$net_name test only.csv", ',')
grad_sltn = sltn[:,2:end].-sltn[:,1:end-1]


trace = scatter(x=sltn[1,1:datasize*100],y=sltn[2,1:datasize*100], mode="markers", name="sltn from start")
trace2 =  scatter(x=[t_data[i][1,1] for i in 1:datasize],y=[t_data[i][2,1] for i in 1:datasize], mode="markers", name="train")
trace3 =  scatter(x=[t_data[i+datasize][1,1] for i in 1:length(t_data)-datasize],y=[t_data[i+datasize][2,1] for i in 1:length(t_data)-datasize], mode="markers", name="test")
trace4 =  scatter(x=sltnt[1,1:end],y=sltnt[2,1:end], mode="markers", name="sltn from test")

plot([trace, trace2, trace3, trace4])

train_data = withoutNode(t_data,1)


function dists(u,t)
    M = train_data[t]
    
    subtract_func(m) = m-u
    direction_vecs = [subtract_func(m) for m in eachcol(M)]

    û = vcat(partialsort(direction_vecs,1:k, by=x->sum(abs2, x))...)

end

iter_sltn = [sltn[:,i] for i in 1:size(sltn)[2]]
data = dists.(iter_sltn, 1.0:0.01:Float64(length(t_data)))
data = hcat(data...)

options = SymbolicRegression.Options(
    binary_operators=[+, *, /, -],
    npopulations=20
)
halls = [
    EquationSearch(
    data[:,1:datasize*100], grad_sltn[i,1:datasize*100], niterations=40, options=options,
    parallelism=:multithreading
    )
    for i in 1:dims[2]
]



# dominatings = [
#     calculate_pareto_frontier(Float64.(data), grad_sltn[i,:], halls[i], options)
#     for i in 1:dims[2]
# ]



sltn1 = vcat([dominatings[i][end].tree(data)' for i in 1:dims[2]]...)


# sltn1 = vcat([sum([sltn1[:,j]' for j in 100*(i-1)+1:100*i]) for i in 1:2*datasize-1]...)'

function next_step(u0, t, tree_eq)
    d = dists(u0, t+datasize)
    u0+vcat([tree_eq(hcat(d))[i][end].tree(hcat(d))' for i in 1:dims[2]]...)
end
 
sltn1 = zeros(Float64, (dims[2],size(sltn1)[2]-100*datasize))
global u0 = targetNode(t_data,1)[1+datasize]
sltn1[:,1] = u0
for t in 2:size(sltn1)[2]-1
    pareto_func(d) = [calculate_pareto_frontier(Float64.(d), grad_sltn[:,t+100*datasize], halls[i], options) for i in 1:dims[2]]
    sltn1[:,t] = next_step(sltn1[:,t-1], 1+(t-1)*0.01, pareto_func)
end


sltn_sym_reg = sltn1[:, 1:100:size(sltn1)[2]]


# temp = zeros(Float64, (dims[2],datasize))
# global u0 = targetNode(t_data,1)[1+datasize]
# temp[:,1].=u0

# for i in 2:datasize
#     temp[:,i] = temp[:,i-1]+sltn_sym_reg[:,i-1]
# end

# sltn_sym_reg = temp



# trace = scatter(x=sltn[1,1:datasize],y=sltn[2,1:datasize], mode="markers")
# trace2 =  scatter(x=[t_data[i][1,1] for i in 1:datasize],y=[t_data[i][1,2] for i in 1:datasize], mode="markers")
# plot([trace, trace2])