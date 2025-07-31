function create_series(n::Int, c::Int, tsteps::Int)
    dv = zeros(Int, n)
    ev = ones(Int, n-1)

    T = Matrix(Tridiagonal(ev ,dv, ev))

    A = Matrix(BlockDiagonal([zeros(1,1), ones(c,c), T]))


    A[1,c+2]=1
    A[c+1,c+2]=1

    A[c+2,1]=1
    A[c+2,c+1]=1

    series = [copy(A)]

    for i in 2:n
        
        A[1,i+c] = 0
        A[1,i+c+1] = 1

        A[i+c,1] = 0
        A[i+c+1,1] = 1
        push!(series, copy(A)) 
    end
    return series
end