function create_series(n::Int, tsteps::Int)
    A = Matrix(BlockDiagonal([zeros(1,1), ones(n,n)]))
    for j in 1:n+1
        A[j, j] = 0
    end

    series = [copy(A)]

    for i in 2:tsteps
        A[1, (i%n)+2] = (A[1, (i%n)+2]+1)%2
        A[(i%n)+2, 1] = (A[(i%n)+2, 1]+1)%2


        push!(series, copy(A)) 
    end

    return series
end