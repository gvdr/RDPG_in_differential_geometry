
include("../_Modular Functions/helperFunctions.jl")

sltn = readdlm("./Code/Solutions/$net_name big net.csv", ',')
sltnt = readdlm("./Code/Solutions/$net_name big net test only.csv", ',')
grad_sltn = sltn[:,2:end].-sltn[:,1:end-1]

test_range = eachindex(1.0:0.01:Float64(datasize))
p1, p2 = (1,2)

trace = PlotlyJS.scatter(x=sltn[p1,:],y=sltn[p2,:], mode="markers", name="sltn from start")
trace2 =  PlotlyJS.scatter(x=[t_data[i,:AL][1,p1] for i in 1:datasize],y=[t_data[i,:AL][1,p2] for i in 1:datasize], mode="markers", name="train")
trace3 =  PlotlyJS.scatter(x=[t_data[i+datasize,:AL][1,p1] for i in 1:length(t_data)-datasize],y=[t_data[i+datasize,:AL][1,p2] for i in 1:length(t_data)-datasize], mode="markers", name="test")
trace4 =  PlotlyJS.scatter(x=sltnt[p1,1:end],y=sltnt[p2,1:end], mode="markers", name="sltn from test")

PlotlyJS.plot([trace, trace2, trace3, trace4])