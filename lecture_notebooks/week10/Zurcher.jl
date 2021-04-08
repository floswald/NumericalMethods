### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ 307997a8-98a9-11eb-345d-cb8acfdf7e4a
begin
	using PlutoUI
	using ShortCodes
end

# ╔═╡ 6bf55a2d-f998-4866-850f-826044ebbfb8
md"
# Bus Engine Replacement Model 

## John Rust (ECTA, 1987)

Today we implement the Bus Engine Replacement model you got to know in [the last homework](https://floswald.github.io/NumericalMethods/hw6/). 

This notebook is again based on [Fedor Iskhakov's](https://github.com/fediskhakov/CompEcon) great material - check it out!

### Setup

Let's remember from the homework that we cast the model in *expected value function space* - i.e. we *integrate out the type 1 EV shock*. We called this object

$$EV(x,d)$$

the **expected value function**, which was defined as

$$EV(x,d) \equiv \mathbb{E}\big[ V(x',\varepsilon')\big|x,d\big] =
\int_{X} \log \big( \exp[v(x',0)] + \exp[v(x',1)] \big) \pi(x'|x,d) dx'$$

To remind you, we arrived at this expression by writing the *normal* value function $V(x,\varepsilon)$ like this

$$V(x,\varepsilon) = \max_{d\in \{0,1\}} \big\{ \underbrace{u(x,d) + \beta
\int_{X} \Big( \int_{\Omega} V(x',\varepsilon') q(\varepsilon'|x') d\varepsilon'\Big)
\pi(x'|x,d) dx'}_{v(x,d)}
+ \varepsilon_d \big\}$$

One key assumption was that the utility shock $\varepsilon$ was additively separable, as well as *max-stable*, such that we could write

$$V(x',\varepsilon') = \max_{d\in \{0,1\}} \big\{ v(x',d) + \varepsilon'_d \big\}$$

We basically can make the replacement

$$V(x,\varepsilon) = \max_{d\in \{0,1\}} \big\{ u(x,d) + \beta
\int_{X} \Big( \underbrace{ \int_{\Omega} V(x',\varepsilon') q(\varepsilon'|x') d\varepsilon'\Big)}_{\log \big( \exp[v(x',0)] + \exp[v(x',1)] \big)}
\pi(x'|x,d) dx'
+ \varepsilon_d \big\}$$

such that we can rewrite the value function with this simpler *log-sum* formulation. Summing up, we write the term that multiplies $\beta$ as

$$\begin{eqnarray}
EV(x,d) &=& \sum_{x' \in X} \log \big( \exp[v(x',0)] + \exp[v(x',1)] \big) \pi(x'|x,d) \\
v(x,d) &=& u(x,d) + \beta EV(x,d)
\end{eqnarray}$$


"

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
	p = [θ ; 1 - sum(θ)]
	if any(p .< 0)
		println("negative probability")
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

# ╔═╡ c8adbfd9-973b-4013-bb24-30a394b833b9
make_trans([0.1,0.5,0.2,0.1,0.05,0.05], 15)

# ╔═╡ 435368bc-5441-4ea4-99c3-01e9408c7ada
md"

## Transition matrix if replacing

$$\Pi(d=1)_{n x n} =
\begin{pmatrix}
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\theta_{20} & \theta_{21} & \theta_{22} & 0 & \cdot & \cdot & \cdot & 0 \\
\end{pmatrix}$$
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

# ╔═╡ 3cf44a77-7173-4662-a2a5-4c6069f743d9
Harold()

# ╔═╡ 728ea1d0-8f25-44a8-98a6-aa4820566773
Harold()

# ╔═╡ 3a7bbc63-bf47-43fe-899d-b86b047d365d
md"

## Payoff function

$$u(x,d,\theta_1)=\left \{
\begin{array}{ll}
    -RC-c(0,\theta_1) & \text{if }d_{t}=\text{replace}=1 \\
    -c(x,\theta_1) & \text{if }d_{t}=\text{keep}=0
\end{array} \right.$$
"

# ╔═╡ 00df9998-cb7a-47a3-a977-7b212865ad26
md"
#### Bellman operator

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
### VFI


1. Start with arbitrary guess for \$EV(x,d)\$  
1. Apply \$T^*\$ operator  
1. Check for (uniform) convergence  
1. If not converged to a given level of tolerance, return to step 2, otherwise finish.  
"

# ╔═╡ Cell order:
# ╠═307997a8-98a9-11eb-345d-cb8acfdf7e4a
# ╟─6bf55a2d-f998-4866-850f-826044ebbfb8
# ╠═3cf44a77-7173-4662-a2a5-4c6069f743d9
# ╟─f705f38e-e7ed-4b8e-bd4e-e0e8748cb173
# ╠═fc74c10d-4349-42ff-b50e-cc3076b6ef14
# ╠═c8adbfd9-973b-4013-bb24-30a394b833b9
# ╟─435368bc-5441-4ea4-99c3-01e9408c7ada
# ╠═7f7733dd-816e-4076-b4a1-0eb79f6bc598
# ╠═728ea1d0-8f25-44a8-98a6-aa4820566773
# ╟─3a7bbc63-bf47-43fe-899d-b86b047d365d
# ╟─00df9998-cb7a-47a3-a977-7b212865ad26
# ╠═fb398266-3d14-4c77-aa00-90dd673b05b4
# ╠═e068ad7a-be3d-40e1-8082-9a96d6a42ad6
