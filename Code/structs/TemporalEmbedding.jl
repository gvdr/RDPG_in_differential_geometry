"""
    TemporalNetworkEmbedding
    A: The raw embedding array dims = [d*n, :]
    n: The number of nodes in the network
    d: The dimension of the embedding

"""

struct TemporalNetworkEmbedding
    A::AbstractArray
    n::Int
    d::Int
end

@inline function Base.getindex(X::TemporalNetworkEmbedding, t::T) where {T<:AbstractFloat} 
    X.A[:,:,Int(floor(t))]*(1-t).+X.A[:,:,Int(ceil(t))]*(t) # uses linear interpolation between indices
end
# Base.getindex(X::TemporalNetworkEmbedding, t::T) where {T<:AbstractFloat} = reshape(X.A[:,Int(floor(t))], (X.n,X.d)) # this uses the floor of a float index
Base.getindex(X::TemporalNetworkEmbedding, t::T) where {T<:Int} = X.A[:,:,t]


Base.getindex(X::TemporalNetworkEmbedding, t::UnitRange{Int64})=TemporalNetworkEmbedding(X.A[:,:,t], X.n, X.d)
Base.lastindex(X::TemporalNetworkEmbedding) = size(X.A)[3]
not(t::Bool)=!t
withoutNode(X::TemporalNetworkEmbedding, t::Int) = TemporalNetworkEmbedding(X.A[:,not.(in.(1:X.n, [t])),:], X.n-1, X.d)

targetNode(X::TemporalNetworkEmbedding, t::Int) = TemporalNetworkEmbedding(X.A[:,t:t,:], 1, X.d)

Base.length(X::TemporalNetworkEmbedding) = size(X.A)[3]


Base.iterate(X::TemporalNetworkEmbedding)= [X[i] for i in 1:length(X)]

