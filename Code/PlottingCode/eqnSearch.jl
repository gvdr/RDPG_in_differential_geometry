function numbersToParameters(eqn)

end


using Symbolics
@syms p x[1:45] e(X, p)
arguments.(eqns[1])[1]=p
p=2
x40=1
x20=1

e = eqns[1]
Symbolics.diff2term(Symbolics.value(eqns[1]))

func = eval(build_function(e, x, p))
func(x, p)
e(4,4)
