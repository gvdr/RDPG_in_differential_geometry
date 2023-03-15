using PlotlyJS
using ColorSchemes
include("../_Modular Functions/loadGlobalConstants.jl")

include("../_Modular Functions/pca.jl")
include("../Symbolic Regression/symreg.jl")




function get_embedding(B, pred)
    mid = convert(Int, dims[1]/2)
    L = B[1:mid,:]

    if dims[2] > 2
        return principle_components([pred'; L])
    else
        return [pred'; L]
    end
end

using DelimitedFiles


sltn = readdlm("./Code/Solutions/$net_name test only.csv", ',')

for i in 1:datasize
    pts = t_data[i+datasize]'#get_embedding([sltn_sym_reg[:,i]'; t_data[i+datasize]], sltn[:,i])
    mid = convert(Int, dims[1]/2)
    traces0 = PlotlyJS.scatter(x=[sltn[1,i]], y=[sltn[2,i]], mode="markers", name="Neural Network Pred", marker_size=12)
    traces1 = PlotlyJS.scatter(x=[sltn_sym_reg[1,i]], y=[sltn_sym_reg[2,i]], mode="markers", name="Symbolic Regression Pred", marker_size=12)
    traces2 = PlotlyJS.scatter(x=[pts[1,1]], y=[pts[1,2]], mode="markers", name="Target Node", marker_size=12)
    
    traces3 = PlotlyJS.scatter(x=pts[2:mid,1], y=pts[1:mid,2], mode="markers", name="Data")
    display(PlotlyJS.plot([traces0,traces1,traces2, traces3]))
    savefig(PlotlyJS.plot([traces0, traces1, traces2, traces3]), "./Code/Plots/Test Only/$net_name/$net_name $i.png")

end

# using Plots

# @userplot ModelPred
# @recipe function f(mp::ModelPred)
#     gr()
#     pts, i = mp.args
#     types = i>datasize ? ["Data" "Test Pred" "True"] : ["Data" "Train Pred" "True"]
#     aspect_ratio --> 1
#     label --> types
#     legend --> :bottomleft
#     seriestype --> :scatter
#     bg --> :linen
#     [pts[3:end,1],[pts[2,1]+0.01],[pts[1,1]+0.01]], [pts[3:end,2],[pts[2,2]], [pts[1,2]]]
# end

# n = length(train_data)
# anim = @animate for i ∈ 1:datasize
#     pts = get_embedding(t_data[i], sltn[:,i])
#     modelpred(pts, i)
# end
# gif(anim, "/home/connor/Thesis/Code/Plots/Test Only/$net_name.gif", fps = 1)


