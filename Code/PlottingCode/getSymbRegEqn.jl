using SymbolicRegression

function getSymbRegEqn(data, sltn, options)
    
    halls = [
        EquationSearch(
        data, sltn[i,:], niterations=40, options=options,
        parallelism=:multithreading
        )
        for i in 1:dims[2]
    ]


    dominatings = [
        calculate_pareto_frontier(Float64.(data), sltn[i,:], halls[i], options)
        for i in 1:dims[2]
    ]
    dominatings

end


