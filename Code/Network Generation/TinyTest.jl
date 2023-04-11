using LinearAlgebra
using BlockDiagonals

k=40
m=35
n=30

l=25


A = Matrix(BlockDiagonal([zeros(1,1), ones(k,k), ones(m,m), ones(n,n)]))

A[1,2:l+1].=1
A[2:l+1,1].=1

A[1,2+k:k+l].=1
A[2+k:k+l,1].=1


A[1,2+k+m:k+m+l-1].=1
A[2+k+m:k+m+l-1,1].=1

series = [copy(A)]

for i in 1:2*l
    if i <= l-2
        A[1,1+k+m+i] = 0
        A[1+k+m+i,1] = 0
    elseif (i>l-2) & (i<=2*l-3)
        A[1,k+i+3-l] = 0
        A[k+i+3-l,1] = 0
    else
        A[1,i+4-2*l] = 0
        A[i+4-2*l,1] = 0
    end

    push!(series, copy(A)) 
end


using DelimitedFiles

for i in 1:length(series)
    writedlm(string("/home/connor/Thesis/Code/Graph Series/3community/step", i, ".csv"), series[i], ",")
end