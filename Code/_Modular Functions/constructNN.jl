using Lux, Random

include("helperFunctions.jl")



function constructNN()
    # x is the point vector we are trying to predict
    # i is the index of the point we are trying to predict 
    rng = Random.default_rng()
    nn = Lux.gpu(Lux.Chain((x) -> x,
          Lux.Dense(input_data_length, 32, tanh),
          Lux.Dense(32, 8),
          Lux.Dense(8, output_data_length)))
    
    p, st = Lux.setup(rng, nn) 
    p = p |> Lux.ComponentArray |> Lux.gpu
    st = st |> Lux.gpu
    
    
    function dudt_pred(du,u,p,t)
      M = reshape(train_data.A[:,Int(floor(t))],(train_data.n,train_data.d))' |> cu
    
      subtract_func(m) = m-u
      direction_vecs = [subtract_func(m) for m in eachcol(M)]
    
      û = vcat(partialsort(direction_vecs,1:k, by=x->sum(abs2, x))...)
    
      du .= nn(û, p, st)[1]
    end
    
    prob_neuralode = ODEProblem{true}(dudt_pred, u0, tspan)
    return cu(prob_neuralode), (p), cu(st),cu(nn)
end