using Serialization
using JSON
using ComponentArrays

using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
      OptimizationOptimisers, Random, Plots

include("create_forecast.jl")
n = 5
d = 2


function get_preds(data_index, data_path, model_path, n, d)
    data = JSON.parsefile(data_path)
    data = [[L[1] L[2]] for L in data[data_index]] 
    println(data[1])
    forecast, tsteps = create_forecast(model_path, Array(reshape(data[1]', n*d)), n, d)
    preds = collect.(Array.(reshape.(forecast, d,n))')
    return preds, data, tsteps
end

L_preds, L_data, tsteps = get_preds("L_series", "data/1_community_oscillation.json", "models/1_community_oscillation/big-NN-07-08-2025-L.jls", n, d)




loss_seq = []
for i in 1:30
    loss_matrix = L_preds[i]*(L_preds[i]*[1 0; 0 -1])'.-L_data[i]*(L_data[i]*[1 0; 0 -1])'
    # println(round.(abs.(loss_matrix[1,:]),sigdigits=2))
    push!(loss_seq, sum(abs.(loss_matrix)))
end




gr()

L_trace_d = hcat([d[1,:] for d in L_data]...)
L_trace_p = hcat([p[1,:] for p in L_preds]...)
plt = plot(tsteps,  L_trace_d[1,:]; label = "data", title="Long d1, moving point prediction", xaxis="Time Step")
plot!(plt, tsteps,  L_trace_p[1,:]; label = "prediction")
plot!([10; 10], [-.06; .05], lw=0.5, lc=:red;)
display(plot(plt))


plt = plt = plot(tsteps,  loss_seq; title="Long Tail total RDPG loss", xaxis="Time Step", yaxis="log10 loss")
plot!([10; 10], [0; 17], lw=0.5, lc=:red;)
