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

# ╔═╡ 307997a8-98a9-11eb-345d-cb8acfdf7e4a
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

# ╔═╡ 967e4554-3a10-4661-8965-cd84a14ec1ca
begin
	struct TwoColumn{L, R}
		left::L
		right::R
	end
	struct TwoColumnW{L, R, lw, rw}
		left::L
		right::R
		leftwidth::lw
		rightwidth::rw
	end

	function Base.show(io, mime::MIME"text/html", tc::TwoColumn)
		write(io, """<div style="display: flex;"><div style="flex: 50%;">""")
		show(io, mime, tc.left)
		write(io, """</div><div style="flex: 50%;">""")
		show(io, mime, tc.right)
		write(io, """</div></div>""")
	end
	function Base.show(io, mime::MIME"text/html", tc::TwoColumnW)
		write(io, """<div style="display: flex;"><div style="flex: $(tc.leftwidth)%;">""")
		show(io, mime, tc.left)
		write(io, """</div><div style="flex: $(tc.rightwidth)%;">""")
		show(io, mime, tc.right)
		write(io, """</div></div>""")
	end
end

# ╔═╡ 2d376efe-49ce-4629-90a4-7bea521ffa97
html"<button onclick='present()'>present</button>"

# ╔═╡ 6692f61f-1aa9-4a7d-b22d-54cdbc072c0d
md"
# ScPo Numerical Methods 2021

**Dynamic Discrete Choice Models**

Florian Oswald
"

# ╔═╡ 6bf55a2d-f998-4866-850f-826044ebbfb8
md"
# Bus Engine Replacement Model, John Rust (ECTA, 1987)

* Model Recap
* Model Solution

## Plan and Credits

* In this notebook we implement the *solution to the dynamic program* in the Bus Engine Replacement Model you got to know in [the last homework](https://floswald.github.io/NumericalMethods/hw6/). 
* In the next notebook, we will see how to then *estimate* the model, i.e. how to choose the model parameters so that it replicates some key features of the observed data.

This notebook is again based on [Fedor Iskhakov's](https://github.com/fediskhakov/CompEcon) great material - check it out!

## Setup

Let's remember from the homework that we cast the model in *expected value function space* - i.e. we *integrate out the type 1 EV shock*. We called this object here

$$EV(x,d)$$

the **expected value function**, which was defined as

$$EV(x,d) \equiv \mathbb{E}\big[ V(x',\varepsilon')\big|x,d\big] =
\int_{X} \log \big( \exp[v(x',0)] + \exp[v(x',1)] \big) \pi(x'|x,d) dx'$$

#

To remind you, we arrived at this expression by writing the *normal* value function $V(x,\varepsilon)$ like this

$$V(x,\varepsilon) = \max_{d\in \{0,1\}} \big\{ \underbrace{u(x,d) + \beta
\int_{X} \Big( \int_{\Omega} V(x',\varepsilon') q(\varepsilon'|x') d\varepsilon'\Big)
\pi(x'|x,d) dx'}_{v(x,d)}
+ \varepsilon_d \big\}$$

One key assumption was that the utility shock $\varepsilon$ was additively separable, as well as *max-stable*, such that we could write

$$V(x',\varepsilon') = \max_{d\in \{0,1\}} \big\{ v(x',d) + \varepsilon'_d \big\}$$

#

We basically can make the replacement

$$V(x,\varepsilon) = \max_{d\in \{0,1\}} \big\{ u(x,d) + \beta
\int_{X} \Big( \underbrace{ \int_{\Omega} V(x',\varepsilon') q(\varepsilon'|x') d\varepsilon'\Big)}_{\log \big( \exp[v(x',0)] + \exp[v(x',1)] \big)}
\pi(x'|x,d) dx'
+ \varepsilon_d \big\}$$

such that we can rewrite the value function with this simpler *log-sum* formulation. Summing up, we write the term that multiplies $\beta$ as

$$\begin{eqnarray}
EV(x,d,\theta) &=& \sum_{x' \in X} \log \big( \exp[v(x',0)] + \exp[v(x',1)] \big) \pi(x'|x,d) \\
v(x,d) &=& u(x,d) + \beta EV(x,d)
\end{eqnarray}$$


"

# ╔═╡ 484255dc-63f1-4e98-b5a7-2ce00f544b78
md"

## Identification

* We had an interesting discussion on *identification* in this model class.
* Typically, we cannot estimate $\beta$, but need to set it.
* Why is that? Write $v$ indexed with $\theta$:
$$v(x,d,\theta) = u(x,d,\theta) + \beta EV_{\theta}(x,d)$$

* In simple terms, we have a *parameter multiplies parameter* problem in $\beta \times EV_{\theta}(x,d)$
* So, we use variation in $v(x,d,\theta)$, as we change $\theta$, to see how the model predictions change; But for any value $\beta$ we could find a value $EV_{\theta}(x,d)$ which would leave the second part unchanged.
* This is an area of ongoing research. The classic reference would be [Magnac & Thesmar](https://www.jstor.org/stable/2692293). There is a lot of work following up on [Shimotsu and Kasahara](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA6763), and many more. For example, [Jaap Abbring (Tilburg)](https://jaap.abbring.org/research/ddc) has an exciting research agenda that allows to separately identify $\beta$!
"



# ╔═╡ e9145508-1f60-4dba-af41-e1c779b1d44a
md"
# Let's Build It!

"

# ╔═╡ 651a8398-db65-423f-b2af-471e8508d684
TwoColumn(
Resource("https://upload.wikimedia.org/wikipedia/commons/d/df/We_Can_Do_It%21_NARA_535413_-_Restoration_2.jpg", :width => 320),
md"
### Checklist
1. Build Transition Matrix
1. Make a Datatype. `Harold` maybe?
1. Write down the Belllman Operator.
1. Write down a Value Function Iterator.
1. Write a `ccp` function to return *conditional choice probabilities*
1. Run, plot, analyze the resulting Value and `ccp` functions.
1. Simulate it.
1. Rejoice! 🎉
")

# ╔═╡ f705f38e-e7ed-4b8e-bd4e-e0e8748cb173
md"



## Transition matrix if *not* replacing

$$\Pi(d=0)_{n x n} =
\begin{pmatrix}
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
0 & \theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & 0 \\
0 & 0 &\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & 0 \\
\cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
0 & \cdot & \cdot & 0 & \theta_{20} & \theta_{21} & \theta_{22} & 0 \\
0 & \cdot & \cdot & \cdot & 0 & \theta_{20} & \theta_{21} & \theta_{22} \\
0 & \cdot & \cdot & \cdot & \cdot  & 0 & \theta_{20} & 1-\theta_{20} \\
0 & \cdot & \cdot & \cdot & \cdot & \cdot  & 0 & 1
\end{pmatrix}$$
"

# ╔═╡ fc74c10d-4349-42ff-b50e-cc3076b6ef14
function make_trans(θ, n)
	transition = zeros(n, n);
	p = [θ ; 1 - sum(θ)]  # if θ does not sum to 1, we add the remainder here
	if any(p .< 0)
		warn("negative probability")
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

# ╔═╡ 435368bc-5441-4ea4-99c3-01e9408c7ada
md"

## Transition matrix if replacing


"

# ╔═╡ 82d338d4-2f08-40f7-9ce2-81188889df1c
TwoColumnW(md"$$\Pi(d=1)_{n x n} =
\begin{pmatrix}
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
	\end{pmatrix}$$",
md"
### Notice

* The model has a *reset* structure
* After replacing an engine at any state $x$, the bus is *like new* (i.e. as if $x=0$)
* That implies that $$EV(x,\text{replace}) = EV(0,\text{keep})$$
* So we only need the one row of 👈 this matrix
* That also means that $EV(x,d)$ does not need to be $(n,2)$, but only $(n,1)$
",
0.7,0.3)

# ╔═╡ aa974dfe-5714-423d-8e69-17072372ed8c
md"
#
"

# ╔═╡ 7f7733dd-816e-4076-b4a1-0eb79f6bc598
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

	# constructor with `true` values from Rust, table X, col 3
	function Harold(;n = 175,maxmiles = 450, RC = 11.7257,c = 2.45569,
					  θ = [0.0937, 0.4475 ,0.4459, 0.0127],
					  β = 0.999, tol =  1e-12)

		this = new()   # create new empty instance of Harold
		this.n = n
		this.maxmiles = maxmiles  # used when reading raw data
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

# ╔═╡ 3a7bbc63-bf47-43fe-899d-b86b047d365d
md"

## The Bellman Operator

Remember the defintion

$$\begin{eqnarray}
T^*(EV)(x,d) &\equiv& \sum_{x' \in X} \log \big( \exp[v(x',0)] + \exp[v(x',1)] \big) \pi(x'|x,d) \\
v(x,d) &=& u(x,d) + \beta EV(x,d)
\end{eqnarray}$$

where to build the *choice specific value function* $v$, we need the utility function $u$:

$$u(x,d,\theta_1)=\left \{
\begin{array}{ll}
    -RC-c(0,\theta_1) & \text{if }d_{t}=\text{replace}=1 \\
    -c(x,\theta_1) & \text{if }d_{t}=\text{keep}=0
\end{array} \right.$$

### Avoiding Numerical Issues

* Rust is careful with numerical precision. computing `exp` for very large or very small numbers is an issue.
* So instead of computing the log-sum as
$$\log \big( \exp[v(x',0)] + \exp[v(x',1)] \big)$$

we compute it as 

$$M + \log \big( \exp[v(x',0)-M] + \exp[v(x',1)-M] \big)$$
where $M = \max (v(x',0),v(x',1))$.
"

# ╔═╡ a2bccdc5-acd3-4130-9ae5-fe123a46896b
md"
#
"

# ╔═╡ b05a9f0e-6966-4f0c-8ac5-acc219872848
md"
$$\begin{eqnarray}
T^*(EV)(x,d) &\equiv& \sum_{x' \in X} \log \big( \exp[v(x',0)] + \exp[v(x',1)] \big) \pi(x'|x,d) \\
v(x,d) &=& u(x,d) + \beta EV(x,d)
\end{eqnarray}$$
"

# ╔═╡ fb398266-3d14-4c77-aa00-90dd673b05b4
# This is operator T^*
# It takes the current EV and returns the next guess for EV
function bellman(h::Harold, ev0::Vector)
	maintainance = -0.001 .* h.mileage .* h.c  # length n
	v0 = maintainance .+ h.β .* ev0  # vector of length n
	v1 = -h.RC + maintainance[1] + h.β * ev0[1]  # a scalar. if you replace, you start at zero miles
	M  = maximum(vcat(v1, v0))  #  largest value in both vs
	logsum = M .+ log.(exp.(v0 .- M) .+ exp(v1 - M))  # center by largest value
	ev1 = h.transition * logsum #  matrix multiplication: that's the summation!
	ev1
end

# ╔═╡ e068ad7a-be3d-40e1-8082-9a96d6a42ad6
md"
## Value Function Iterator


1. Start with arbitrary guess for \$EV(x,d)\$  
1. Apply \$T^*\$ operator  
1. Check for convergence  
1. If not converged to a given level of tolerance, return to step 2, otherwise exit.  
"

# ╔═╡ 48b44af3-e854-4b06-ba50-eabcbdda6958
function vfi(h::Harold)
	if h.β >= 1.0
		throw(ArgumentError("VFI will not converge with β >= 1.0"))
	end
	ev0 = zeros(h.n)  # arbitrary starting value
	err = 100.0  # set high error
	iters = 0

	while err > h.tol
		iters += 1
		ev1 = bellman(h, ev0)  # apply T operator
		err = norm(abs.(ev0 .- ev1))
		ev0[:] .= ev1   #  [:] do not reallocate a new object
	end
	return (ev0, iters)
end

# ╔═╡ 53ebae3d-9caf-48b6-be21-d18696602d64
md"

## CCP function

* That's just repetition.
* Remember the expression:
$$P(d|x) = \frac{\exp[v(x,d)]}{\sum_{d'\in \{0,1\}} \exp[v(x,d')]}$$

* In our case with 2 alternatives only, that reduces to 
$$P(d=1|x) = \frac{1}{1 + \exp[v(x,0) - v(x,1)]}$$
"



# ╔═╡ 0dc836e1-b750-4130-ad73-22e44efbde0f
# probability of replace! 
function ccp(h::Harold, ev::Vector)
	maintainance = -0.001 .* h.mileage .* h.c  
	v0 = maintainance .+ h.β .* ev  
	v1 = -h.RC + maintainance[1] + h.β * ev[1] 
	1.0 ./ (exp.(v0 .- v1) .+ 1)
end

# ╔═╡ 5b25b9f2-a574-4a18-8290-238ef0b6634c
md"

## Run It!
"

# ╔═╡ b4ba36b9-d7f5-46aa-a39b-4f2965ed8252
function runit(;n=90, β=0.9999,c=2.4, θ=[0.3, 0.68, 0.02])
	z = Harold(n=n, β=β, c=c, θ=θ)
	sol, iters = vfi(z)
	pr  = ccp(z, sol)
	sol, pr, iters, z
end

# ╔═╡ 9a7ed619-5345-48e9-825d-33fff1fdd8fb
md"
## Plot it!
"

# ╔═╡ f9cbfa5f-9ca8-4cb4-a512-25cdc9c93147
@bind ns Slider(90:200, default=90, show_value=true)

# ╔═╡ 337a8853-95d5-405a-a071-253318fa6999
function plotit(;n=90, β=0.9)
	sol, pr, iters, z = runit(n=n, β=β)
	plot(plot(z.mileage, sol, title="Vfun: $iters Iterations"), 
			plot(z.mileage, pr,  title="Prob of Replacement"),
			xlab="Miles", leg=false)
end

# ╔═╡ 3f85eba8-5e38-4684-b91c-c2e817164321
md"
## Simulate the Model Solution
"

# ╔═╡ b662f496-c525-4038-b6ef-e412bdb5f927
TwoColumn(md"
### No Strict Need
* Notice that we do **not** need simulation to later on estimate the model.
* In fact, as Fedor says in the video, the way Rust connects the model solution to the data via Maximum Likelihood is one the more remarkable features of the paper.
* Still, it's interesting to see how the model would look if simulated!
",
md"
### Checklist for Simulation
1. Create `Harold`
1. Get Model Solution: Value function, and CCPs
1. Start off a single bus on first grid point of miles
1. set action to `0`, i.e. keep the engine
1. Iterate forward for `T` periods, updating the miles state, checking whether a replacement action can be triggered (draw random shock and compare to CCP at that mileage state)
1. If no replacement, move up the mileage state in accordance with the LOM
1. Plot.
")


# ╔═╡ bcbfa438-79a9-4d60-9661-e9275e1853db
md"
#
"

# ╔═╡ 637f4a07-417b-4bc2-b993-f56e9b9cdc4f
@bind β Slider(0:0.001:1, default=0.9, show_value=true)

# ╔═╡ 2a796e16-08be-458e-9695-dbac3236551c
plotit(n=ns,β=β)

# ╔═╡ b3abc72b-4dc9-47e1-b9ac-312e31c03402
function simit(; T=500,n=500, θ=[0.3,0.6,0.1], β = 0.9)
	z = Harold(n=n, θ=θ, β = β)  # need go higher with miles
	sol, iters = vfi(z)
	pr  = ccp(z, sol)

	P = cumsum(z.θ ./ sum(z.θ))

	miles = Int[]
	push!(miles, 1)  # start on first grid point of mileage
	a = Int[]  # 0/1 keep or replace
	push!(a, 0)  #  keep in first period

	for it in 2:T
		action = rand() < pr[miles[end]] ? 1 : 0
		push!(a, action)
			# update miles
		if action == 1
			push!(miles, 1)  # go back to first state
		else
			next_miles = findfirst(rand() .< P)  # index of first `true`
			push!(miles, miles[end] + next_miles)
		end
	end

	plot(1:T, miles, xlab="Period", ylab="miles",
				title="Simulating Harold Zurcher's Decisions",
				leg=false, lw=2)
end

# ╔═╡ 6c4fe6e0-6cb9-4758-ab9f-7bc9ec2898fc
simit(β = β)

# ╔═╡ 06ce067a-490d-4479-9830-f692a78e88a1
md"

# Further Resources and Thanks

- John Rust [https://en.wikipedia.org/wiki/John_Rust](https://en.wikipedia.org/wiki/John_Rust)  
- Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher [https://www.jstor.org/stable/1911259](https://www.jstor.org/stable/1911259)  
- Google scholar citing Rust (1987) [https://scholar.google.com/scholar?oi=bibs&hl=en&cites=16527795233338248687](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=16527795233338248687)  

Big Thanks again to [Fedor Iskhakov](https://github.com/fediskhakov/CompEcon) for sharing the notebooks!
"

# ╔═╡ Cell order:
# ╠═307997a8-98a9-11eb-345d-cb8acfdf7e4a
# ╟─967e4554-3a10-4661-8965-cd84a14ec1ca
# ╠═2d376efe-49ce-4629-90a4-7bea521ffa97
# ╠═6692f61f-1aa9-4a7d-b22d-54cdbc072c0d
# ╟─6bf55a2d-f998-4866-850f-826044ebbfb8
# ╟─484255dc-63f1-4e98-b5a7-2ce00f544b78
# ╟─e9145508-1f60-4dba-af41-e1c779b1d44a
# ╟─651a8398-db65-423f-b2af-471e8508d684
# ╟─f705f38e-e7ed-4b8e-bd4e-e0e8748cb173
# ╠═fc74c10d-4349-42ff-b50e-cc3076b6ef14
# ╟─435368bc-5441-4ea4-99c3-01e9408c7ada
# ╟─82d338d4-2f08-40f7-9ce2-81188889df1c
# ╟─aa974dfe-5714-423d-8e69-17072372ed8c
# ╠═7f7733dd-816e-4076-b4a1-0eb79f6bc598
# ╟─3a7bbc63-bf47-43fe-899d-b86b047d365d
# ╟─a2bccdc5-acd3-4130-9ae5-fe123a46896b
# ╠═b05a9f0e-6966-4f0c-8ac5-acc219872848
# ╠═fb398266-3d14-4c77-aa00-90dd673b05b4
# ╟─e068ad7a-be3d-40e1-8082-9a96d6a42ad6
# ╠═48b44af3-e854-4b06-ba50-eabcbdda6958
# ╠═53ebae3d-9caf-48b6-be21-d18696602d64
# ╠═0dc836e1-b750-4130-ad73-22e44efbde0f
# ╟─5b25b9f2-a574-4a18-8290-238ef0b6634c
# ╠═b4ba36b9-d7f5-46aa-a39b-4f2965ed8252
# ╟─9a7ed619-5345-48e9-825d-33fff1fdd8fb
# ╠═637f4a07-417b-4bc2-b993-f56e9b9cdc4f
# ╠═f9cbfa5f-9ca8-4cb4-a512-25cdc9c93147
# ╠═2a796e16-08be-458e-9695-dbac3236551c
# ╟─337a8853-95d5-405a-a071-253318fa6999
# ╟─3f85eba8-5e38-4684-b91c-c2e817164321
# ╟─b662f496-c525-4038-b6ef-e412bdb5f927
# ╟─bcbfa438-79a9-4d60-9661-e9275e1853db
# ╠═6c4fe6e0-6cb9-4758-ab9f-7bc9ec2898fc
# ╠═b3abc72b-4dc9-47e1-b9ac-312e31c03402
# ╟─06ce067a-490d-4479-9830-f692a78e88a1
