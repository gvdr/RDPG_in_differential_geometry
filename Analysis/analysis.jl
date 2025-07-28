using Serialization
using JSON

include("create_forecast.jl")


function get_preds(data_index, data_path, model_path)
    data = JSON.parsefile(data_path)
    data = [[L[1] L[2]] for L in data[data_index]] 

    forecast, tsteps = create_forecast(model_path, Array(reshape(data[1]', 10)))
    preds = collect.(Array.(reshape.(forecast, 2,5))')
    return preds, data, tsteps
end

L_preds, L_data, tsteps = get_preds("L_series", "data/1_community_oscillation.json", "models/1_community_oscillation/25-07-2025-L.jls")
R_preds, R_data, tsteps = get_preds("R_series", "data/1_community_oscillation.json", "models/1_community_oscillation/25-07-2025-R.jls")


for i in 1:30
    loss_matrix = L_preds[i]*R_preds[i]'.-L_data[i]*R_data[i]'
    println(round.(abs.(loss_matrix[1,:]).+abs.(loss_matrix[:,1]),sigdigits=2))
    println("")
end

L_trace_d = [d[1,2] for d in L_data]
L_trace_p = [p[1,2] for p in L_preds]
plt = scatter(tsteps, L_trace_d; label = "data")
scatter!(plt, tsteps, L_trace_p'; label = "prediction")
display(plot(plt))

R_trace_d = [d[1,2] for d in R_data]
R_trace_p = [p[1,2] for p in R_preds]
plt = scatter(tsteps, R_trace_d; label = "data")
scatter!(plt, tsteps, R_trace_p'; label = "prediction")
display(plot(plt))