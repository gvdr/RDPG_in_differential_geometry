using DataFrames
using LinearAlgebra
using Arpack




# function that does svd decomposition
function do_the_rdpg(A,d)
    L,Σ,R = svds(A; nsv=d, v0=[Float64(i%7) for i in 1:minimum(size(A))])[1]
    L̂ = L * diagm(.√Σ)
    R̂ = R * diagm(.√Σ)
    return (L̂ = L̂, R̂ = R̂)
end

# Create function do give neural net an input.

# Function that takes take vertex embedding, one vertex and returns all distances between that vertex and the k closest vertices

# function vertex_distances(M, k)
#     distances = []  # preallocate sizes
#     vertex_norm_distances = []

#     for i::Int8 in 1:n                                                 # find closest_vertices to v 
#         if i != v
#             distance = norm(M[:,v] - M[:,i], 1)
#             push!(vertex_norm_distances, (distance, i))
    
#         end 
#     end

#     closest_vertices_to_v = partialsort(vertex_norm_distances,1:k, by=x->x[1])

#     for i in 1:length(closest_vertices_to_v)
#         vertex_index = closest_vertices_to_v[i][2]           
#         for j in 1:size(M)[1]                                          # iterating over all columns in matrix
#             push!(distances, M[j, v] - M[j,vertex_index])
#         end
#     end

#     return closest_vertices_to_v
# end


# function vertex_distances_2(M::Matrix, v::T, k::Integer, distance_metric=cosine_dist) where {T<:AbstractArray}
#     cosine_distances = zeros(Float64, size(M)[1])
    
#     for i::Int in eachindex(cosine_distances)
#         distance::Float64 = distance_metric(v, M[i,:])  
#         cosine_distances[i] = isnan(distance) ? zero(distance) : distance
#     end
#     closest_cosine_distances= partialsort(cosine_distances,1:k, by=x->x)

#     return closest_cosine_distances
# end

function printall(iter)
    for i in iter
        println(i)
    end
end

