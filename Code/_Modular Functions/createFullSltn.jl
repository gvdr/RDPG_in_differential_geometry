
# function createFullSltn(data, prob_neuralode, result)
#     prediction = []
#     for i in 1:size(data)[2]
#         point_preds = []
#         M = reshape(data[:,i], (dims[1],dims[2]))
#         for v in 1:1
#             u0 = includeKNNdists(M[v,:], M[vcat(1:mid.!=v, zeros(Bool, mid)...),:])
#             pnode = remake(prob_neuralode, u0=u0)


#             pred = predict_neuralode(result.u; pnode)[1:dims[2],2]
#             push!(point_preds, pred)
#         end
#         p = hcat(point_preds...)
#         push!(prediction, copy(p))
#     end

#     return prediction
# end


