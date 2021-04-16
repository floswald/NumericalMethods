### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ a0829c7b-3843-436b-b52a-3db8b46d3cea
begin
	using PlutoUI
    using LinearAlgebra  # for norm()
    using Plots
    using DataFrames
    using CSV
    using StatsPlots
    using StatsBase  # counts
    using Optim
    using NLopt
end

# ╔═╡ 79dee14b-5c6c-4358-ba09-1490d23575bc
html"<button onclick='present()'>present</button>"

# ╔═╡ 43e3f61c-9bb0-11eb-35bc-ade31ecafbe8
md"

# ScPo Numerical Methods 2021

**Estimating the Rust Bus Model**

Florian Oswald

"

# ╔═╡ 48b5b838-c5b7-4992-b020-e653f0a8451b
md"

* in this notebook we continue with the Bus Engine Replacement model.
* We heard in class last week about the NFXP and MPEC methods.
* Both require us to formulate the likelihood function in this model class. (other models could be estimated with a GMM setup, all depends on what the model outputs).

## Model Recap

$$\begin{eqnarray}
T^*(EV)(x,d) &\equiv& \sum_{x' \in X} \log \big( \exp[v(x',0)] + \exp[v(x',1)] \big) \pi(x'|x,d) \\
v(x,d) &=& u(x,d) + \beta EV(x,d)
\end{eqnarray}$$


$$P(d|x) = \frac{\exp[v(x,d)]}{\sum_{d'\in \{0,1\}} \exp[v(x,d')]}$$


So, object $P$ (the ccp) depends on the solution of the structural model, since $v$ depends on $EV$. To make this clear, we will index $P(d|x, EV_\theta)$ from here on.

## Likelihood Function

This model outputs a conditional probability distribution - the $P$s. *Estimating* such a model means to use Maximum Likelihood to see how *likely* the observed data could have been generated from this particular model at hand.

The likelihood function in this problem looks like this:

$$\mathcal{L}_n(\theta, EV_\theta) = \prod_{i=1}^{162}\prod_{t=2}^{T_i} P(d_{i,t}|x_{i,t}, EV_\theta) \pi(x_{i,t}|x_{i,t-1},d_{i,t-1})$$

* We have unique 162 buses in our 4 groups
* Each bus has a different number of observations $T_i$
* The CCP (i.e. the result of the structural model) is $P(d|x, EV_\theta)$
* The *exogenous* state transition is$\pi(x_t|x_{t-1},d_{t-1})$ 

## Log Likelihood

* As in our homework, we work with the log of the likelihood function, i.e.

$$\ell_n(\theta, EV_\theta) = \sum_{i=1}^{162}\sum_{t=2}^{T_i} \log P(d_{i,t}|x_{i,t}, EV_\theta) + \log \pi(x_{i,t}|x_{i,t-1},d_{i,t-1})$$

**Estimation Problem**

In reality, this is an optimization problem. Given that the Bellman Operator $T$ has a unique fixed point for each $\theta$, we *just* have to choose $\theta$!

$$\max_{\theta} \ell_n(\theta, EV_\theta)$$

* This is an *unconstrained* problem.
* In practice, we again maximise the *average* log likelihood function, i.e. divide by $n$.

## Double Loop


* We heard last tiem that the nested fixed point (NFXP) approach has an inner (find $EV = T(EV)$, given \theta) and an outer loop (find $\theta = \arg \max \ell_n$)
* We set up all the code to compute $EV = T(EV)$ and the CCP function. So now we can proceed to build the data-interface for the estimation!

#### Components

1. log likelihood function: evaluates $\ell$ above.
1. *nested* log likelihood function: takes a candidate $\theta^(k)$, updates `Harold`, computes model solution, and evaluates log likelihood
1. `nfxp` a function that loads data, sets starting values, and calls a nonlinear optimizer on the nested likelihood function.

## Likelihood

* We will set this up such that we can estimate the parameters for the milage transition in $\pi$ inside or outside the structural model.

"

# ╔═╡ 387d9de3-9dc1-4e3c-8b9b-cbb7cc9cb85c
md"
#
"

# ╔═╡ d3221610-1791-4d87-acdd-4a5bd9a2b37a
md"
#
"

# ╔═╡ 62893e3f-195f-4581-aa07-7a8b9115560b
md"
#

* We run `nfxp` for a certain setup of model parameters. `n` for example is the number of grid points.
* Let's quickly look at the data do see what those choices imply.
"

# ╔═╡ 5168797e-01d0-4697-b796-c0e270550348
# a simple frequency count is sufficient to estimate how mileage bins increase
function partialMLE(dx1::Vector) 
	c = counts(dx1) ./ length(dx1)
	c[1:(end-1)]  # forget about smallest category
end

# ╔═╡ 57632a86-1479-4331-a8e8-94865193a4eb
@bind n Slider(50:200, show_value = true)

# ╔═╡ c6ed299c-b9a7-47e7-ab3a-959031aa7d0a
md"""

## Time to Roll!

* Time to run this now.
* We looked at the code to solve for the model last time, so that should be ok.
* But what is going to go a *success* for our estimation exercise? 
* Well, we want to get close to 👇 this! We use Groups 1,2,3,4.

$(LocalResource("./Rust-table9.png"))
"""

# ╔═╡ 2e555776-ce29-4791-9542-27093d132a0e
md"
#

* ok let's see!
"

# ╔═╡ 165022ec-6952-4844-bfc4-c6cac6a501b1
md"
🎉
(running with β=0.9999 as in Rust would get even closer but it takes too long!)
"

# ╔═╡ 57a1df74-f19f-41f2-b242-42d3b35a2971
md"
#

* What about MPEC?
* That's too much code to show in the notebook. let's go over it [on the repository](https://github.com/floswald/Zurcher.jl)!
"

# ╔═╡ efd9b30b-8dd9-4656-b95b-13a999a58111
md"""
## Larger State Space

* Rust then increase `n` to 175.
* The estimates slightly change, particularly for 

$(LocalResource("./Rust-table10.png"))

"""

# ╔═╡ 0ea31a68-74ea-4cbd-a920-9ac384cfc67a
md"
#
"

# ╔═╡ 1848a939-7c2b-436f-ac45-03f5371e6ce5
md"
library
"

# ╔═╡ c5edc6f2-380c-47a8-8eaa-e44868f83eb6
function make_trans(θ, n)
	transition = zeros(n, n);
	p = [θ ; 1 - sum(θ)]
	if any(p .< 0)
		@warn "negative probability in make_trans" maxlog=1
	end
	np = length(p)

	# standard block
	for i = 0:n - np
		transition[i + 1,(i + 1):(i + np)] = p
	end

	for k in 1:(np-1)
		transition[n - k,(n - k):n] = [p[1:k]...; 1 - sum(p[1:k])]
	end
	transition[n,n] = 1.0
	return transition
end

# ╔═╡ 684ac370-bb0d-4f42-8e3c-65f4f68329a8
mutable struct Harold
	# parameters
	n::Int
	maxmiles::Int
	RC::Float64 
	c::Float64 
	θ::Vector{Float64}
	β::Float64 

	# numerical settings
	tol::Float64

	# state space
	mileage::Vector{Float64}
	transition::Matrix{Float64}

	function Harold(;n = 175,maxmiles = 450, RC = 11.7257,c = 2.45569,
					  θ = [0.0937, 0.4475 ,0.4459, 0.0127],
					  β = 0.999, tol =  1e-12)

		this = new()   # create new empty Harold instance
		this.n = n
		this.maxmiles = maxmiles
		this.RC = RC
		this.c = c
		this.θ = θ
		this.β = β
		this.tol = tol

		# build state space
		this.mileage = collect(0.0:n - 1)
		this.transition = make_trans(θ, n)
		return this
	end
end

# ╔═╡ daf361e3-9e68-4f64-8e33-a51ce443d3fa
function loglikelihood(h::Harold,p_model,
					   replace_data::BitArray,
					   miles_data::Vector{Int64},
					   miles_increase_data::Vector{Int64};do_θ = false)

	# check data arrays are consistent
	nt = length(replace_data)
	@assert(nt == length(miles_data))
	@assert(nt == length(miles_increase_data))

	# 1. get model-implied Probability of replace at miles_data
	prob_replace = p_model[miles_data]  
	prob_keep    = 1 .- prob_replace 

	# 2. for each observed discrete choice (replace or not)
	# compute the model implied probability of that happening
	logL = log.(prob_keep .* (.!(replace_data)) .+ (prob_replace .* replace_data))

	# 3. compute likelihood of mileage transition
	# adjust \theta for last element
	# this is tricky because implicitly need to enforce constraint that all p sum to 1
	# can get negative numbers here for the last element.
	if do_θ
		p_miles = clamp!([h.θ ; 1 .- sum(h.θ)], 0.0, 1.0)  # the clamp!() constrains the values to [0,1] mechanically
		logL = logL .+ log.(p_miles[miles_increase_data .+ 1])
	end

	mean((-1) * logL)
end

# ╔═╡ 2bd5b2d5-7bc1-4986-89b0-88dea6d8bf74
function busdata(z::Harold; bustype = 4) 
	d = CSV.read(joinpath("buses.csv"), DataFrame, header = false)
	select!(d, 1,2,5,7)
	rename!(d, [:id, :bustype, :d1, :odometer])

	d = filter(x -> x.bustype .<= bustype, d)

	# discretize odometer
	transform!(d, :odometer => (x -> Int.(ceil.(x .* z.n ./ (z.maxmiles * 1000)))) => :x)

	# replacement indicator
	dd = [d.d1[2:end] ; 0]

	# mileage increases
	dx1 = d.x .- [0;d.x[1:(end-1)]]
	dx1 = dx1 .* (1 .- d.d1) .+ d.x .* d.d1

	# make new dataframe
	df = [select(d, :id, :x, :bustype) DataFrame(dx1 = dx1, d = BitArray(dd))]

	# get rid of first observation for each bus
	idx = df.id .== [0; df.id[1:end-1]]
	df = df[idx,:]
end

# ╔═╡ fe4345fd-3eb2-471a-9be6-0e7d39cb2d1a
bd = busdata(Harold())

# ╔═╡ 91b85c93-a2d4-45bc-b181-b51695bf6178
function dplots(n)
	dd = busdata(Harold(n = n))
	b2 = groupby(dd, [:x, :bustype])
	bh2 = combine(b2, :d => mean => :mean_replaced, nrow)

	p1 = @df bh2 bar(:x, :mean_replaced, group=:bustype, title="avg replacement indicator", bar_position=:dodge, ylab="share replaced", xlab="miles", alpha=0.9, legend = :topleft)
	p2 = @df bh2 groupedbar(:x, :nrow, group=:bustype, bar_position=:stack, xlab="mileage state", ylab="number of buses", title="mileage states by bus groups", bar_width=1)
	plot(p1,p2)
end

# ╔═╡ 1b0fb366-0557-4dc4-bf79-553f4d068737
dplots(n)

# ╔═╡ e0a0d3fc-b183-4b10-bead-a13170c61538
function bellman(h::Harold, ev0::Vector)
	maintainance = -0.001 .* h.mileage .* h.c  # StepRange of length n
	v0 = maintainance .+ h.β .* ev0  # vector of length n
	v1 = -h.RC + maintainance[1] + h.β * ev0[1]  #  a scalar. if you replace, you start at zero miles
	M  = maximum(vcat(v1, v0))  #  largest value in both vs
	logsum = M .+ log.(exp.(v0 .- M) .+ exp(v1 - M))
	ev1 = h.transition * logsum #  matrix multiplication
	ev1
end

# ╔═╡ bbc4c659-1577-4af0-a892-d980858800fc
function ccp(h::Harold, ev::Vector)
	maintainance = -0.001 .* h.mileage .* h.c  # StepRange of length n
	v0 = maintainance .+ h.β .* ev  # vector of length n
	v1 = -h.RC + maintainance[1] + h.β * ev[1] 
	1.0 ./ (exp.(v0 .- v1) .+ 1)
end

# ╔═╡ 8149b1eb-c401-45eb-8262-f0615336fcd9
function vfi(h::Harold)
	if h.β >= 1.0
		throw(ArgumentError("value function iteration will not converge with β >= $(h.β)"))
	end
	ev0 = zeros(h.n)  # starting value
	err = 100.0
	iters = 0

	while err > h.tol
		iters += 1
		ev1 = bellman(h, ev0)
		err = norm(abs.(ev0 .- ev1))
		ev0[:] .= ev1   #  [:] do not reallocate a new object
	end
	return (ev0, iters)
end

# ╔═╡ ec467235-26c1-45c7-886d-d3c5a572a784
function nested_likelihood(x::Vector{Float64}, h::Harold, d::DataFrame)
	# update Harold
	h.RC = x[1]
	h.c  = x[2]
	if length(x) > 2
		h.θ  = x[3:end]
		h.transition = make_trans(h.θ,h.n)
	end

	# compute structural model 
	sol, iters = vfi(h)
	pr  = ccp(h, sol)

	# evaluate likelihood function 
	loglikelihood(h, pr, d.d, d.x, d.dx1, do_θ = length(x) > 2)
end

# ╔═╡ 724b6d55-9757-4c58-94ef-9de0e8f64669
function nfxp(; n = 175, β = 0.9, is_silent = false, doθ = false)
	z = Harold(RC = 0.0, c = 0.0, β = β,n = n)

	# get data
	d = busdata(z)
	# get initial transition probabilities
	if doθ
		p_0 = partialMLE(d.dx1)
		x_0 = [0.0, 0.0, p_0...]  # starting value with thetas
	else
		x_0 = [0.0, 0.0]  # starting value with thetas
	end

	# optimize likelihood which calls the structural model

	r = Optim.optimize( x -> nested_likelihood(x, z, d), x_0 ,BFGS(),  Optim.Options(show_trace  = !is_silent ))
	o = Optim.minimizer(r)
	if doθ
		(RC = o[1], θc = o[2], θp = o[3:end])
	else
		(RC = o[1], θc = o[2])
	end

end

# ╔═╡ 06645c70-959c-490e-b3e5-c5eebe42268e
nfxp()

# ╔═╡ 31c0c8b6-3630-432a-97c6-f59842b7185c
nfxp(n = 90,β = 0.999)

# ╔═╡ ed23d7fd-80ac-4cff-a791-4535c9450db1
nfxp(n = 90,β = 0.999, doθ = true)

# ╔═╡ ca35cfb9-344a-4345-a560-76393364b863
nfxp(n = 175,β = 0.99, doθ = true)

# ╔═╡ Cell order:
# ╟─a0829c7b-3843-436b-b52a-3db8b46d3cea
# ╟─79dee14b-5c6c-4358-ba09-1490d23575bc
# ╟─43e3f61c-9bb0-11eb-35bc-ade31ecafbe8
# ╟─48b5b838-c5b7-4992-b020-e653f0a8451b
# ╠═daf361e3-9e68-4f64-8e33-a51ce443d3fa
# ╟─387d9de3-9dc1-4e3c-8b9b-cbb7cc9cb85c
# ╠═ec467235-26c1-45c7-886d-d3c5a572a784
# ╟─d3221610-1791-4d87-acdd-4a5bd9a2b37a
# ╠═724b6d55-9757-4c58-94ef-9de0e8f64669
# ╟─62893e3f-195f-4581-aa07-7a8b9115560b
# ╠═5168797e-01d0-4697-b796-c0e270550348
# ╠═57632a86-1479-4331-a8e8-94865193a4eb
# ╠═1b0fb366-0557-4dc4-bf79-553f4d068737
# ╠═fe4345fd-3eb2-471a-9be6-0e7d39cb2d1a
# ╟─c6ed299c-b9a7-47e7-ab3a-959031aa7d0a
# ╟─2e555776-ce29-4791-9542-27093d132a0e
# ╠═06645c70-959c-490e-b3e5-c5eebe42268e
# ╠═31c0c8b6-3630-432a-97c6-f59842b7185c
# ╠═ed23d7fd-80ac-4cff-a791-4535c9450db1
# ╟─165022ec-6952-4844-bfc4-c6cac6a501b1
# ╟─57a1df74-f19f-41f2-b242-42d3b35a2971
# ╟─efd9b30b-8dd9-4656-b95b-13a999a58111
# ╟─0ea31a68-74ea-4cbd-a920-9ac384cfc67a
# ╠═ca35cfb9-344a-4345-a560-76393364b863
# ╟─91b85c93-a2d4-45bc-b181-b51695bf6178
# ╟─1848a939-7c2b-436f-ac45-03f5371e6ce5
# ╠═2bd5b2d5-7bc1-4986-89b0-88dea6d8bf74
# ╠═c5edc6f2-380c-47a8-8eaa-e44868f83eb6
# ╠═684ac370-bb0d-4f42-8e3c-65f4f68329a8
# ╠═e0a0d3fc-b183-4b10-bead-a13170c61538
# ╠═bbc4c659-1577-4af0-a892-d980858800fc
# ╠═8149b1eb-c401-45eb-8262-f0615336fcd9
