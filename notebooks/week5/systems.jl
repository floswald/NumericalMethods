### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ c3572e14-c058-455c-96c0-6324970af323
using NLsolve

# ╔═╡ 84674032-4848-11ec-3ead-092daf45ee6a
md"""
# Solving Systems of Equations

>This is a simplified version of our paper [Structural Change, Land Use and Urban Expansion](https://floswald.github.io/publication/landuse/) by Coeurdacier, Oswald and Teignier. I am using this only to illustrate the steps involved to solve a system of equations.

## A Model of Land Use and Housing

* There are 2 sectors, Urban and Rural, each producing one good. We take the urban good as the numeraire good and assign relative price $p$ to the rural good.
* There are $L$ workers: $L_r$ rural workers and $L_u$ urban workers.
* The wage is identical in both sectors.
* Land can be use for either Rural Production or to live on.
* People consume rural, urban and housing goods. The housing good is provided by a developer who makes zero profit. He buys land at unit price $\rho$ and combines it with urban good to produce housing services. Those get rented out to people at unit price $q$.
* There is *no notion of space*, i.e. people do *not* have to commute. Hence, a key mechanism from our full model is missing.

## preferences

$$U = c_r^{\nu (1-\gamma)} c_u^{(1-\nu) (1-\gamma)} h^\gamma$$

budget

$$p c_r + c_u + q h= w + r$$

where $w$ is wage, $r$ is land rents which are given back to each individual and $q$ is the rental price of housing.

## Expenditure

maximizing U wrt budget gives expenditure on each good

$$\begin{align}
c_u &=(1-\nu) (1-\gamma) (w + r)\\
c_r &=\nu (1-\gamma) (w + r) / q\\
h   &=\gamma (w + r) / q\\
\end{align}$$
"""

# ╔═╡ 607a4845-02f1-4bc0-8e44-6951d33c4367
md"""
## Production

$$\begin{align}
Y_u &=\theta_u L_u\\
Y_r &=\theta_r (L_r^\alpha S_r^{1-\alpha})
\end{align}$$

The Urban firm has profits $1 \times Y_u - w L_u= \theta_u L_u - w L_u$, hence has an optimality condition

$$\theta_u = w$$

The rural firm has profit $p Y_r - w L_r - \rho S_r$ which it maximizes. FOCs

$$\begin{align}
w &= p \alpha \theta_r L_r^{\alpha-1} S_r^{1-\alpha}\\
\rho &= p (1-\alpha) \theta_r L_r^{\alpha} S_r^{-\alpha}\\
\end{align}$$

From the first condition we obtain the relative price of rural good:
$$p = \frac{w}{\alpha \theta_r} \left( \frac{L_r}{S_r} \right)^{1-\alpha}$$

while by combingin both we get a condition for the amount of land used in rural production

$$S_r = \frac{w}{\rho} \frac{1-\alpha}{\alpha} L_r$$
"""

# ╔═╡ 4a6cbff6-3c1a-414a-a866-599cb5df177c
md"""
## Housing Supply

The housing developer will supply $H$ units of housing in any location at a cost of 

$$\frac{H^{1+1/\epsilon}}{1+1/\epsilon}$$

which they will pay for in terms of numeraire (urban) good. Their profits are thus

$$\pi = q H - \frac{H^{1+1/\epsilon}}{1+1/\epsilon} - \rho$$

Maximizing wrt $H$ yields optimal housing supply function

$$H^* = q^\epsilon$$

Free entry implies $\pi = 0$, hence we get

$$\begin{align}
q^{1+\epsilon} - \frac{q^{1+\epsilon}}{1+1/\epsilon} &= \\
 \frac{q^{1+\epsilon}}{1+\epsilon} &=\rho
\end{align}$$

This expression determines housing prices $q$ as a function of the land price $\rho$.
"""

# ╔═╡ c5255877-e38d-4614-9ea6-7678d13ebb06
md"""
## Housing and Land Market Clearing

* Both rural and urban workers consume housing space $h$ as per above.
* In both urban and rural area, we need that supply equals housing demand.

$$\begin{align}
\text{supply} & =  \text{demand} \\
 H^* = q^{\epsilon} & = L_u h =L_u \gamma (w + r) / q\\
q^{1+\epsilon} & = L_u \gamma (w + r) \\
\rho (1+\epsilon) & = L_u \gamma (w + r) \\
\end{align}$$
where the last equality follows from the previous section. Given that $L_u \gamma (w + r)$ represents total demand for housing space in the urban sector, and given that in urban we *only* have space for housing, this defines the size of the city $\phi$:

$$\phi = \frac{L_u \gamma_\epsilon (w + r)}{\rho}$$ where $\gamma_\epsilon = \gamma / (1+\epsilon)$. Similarly in the rural sector, we define space for housing as $S_{rh}$ and obtain

$$S_{rh} = \frac{L_r \gamma_\epsilon (w + r)}{\rho}$$
"""

# ╔═╡ 59fb80db-10c5-4cca-8f85-be7599320242
md"""
## Market Clearing and Feasibility Constraint

We have to markets to clear: the labor market and the housing/land market. The labor market is easy, we need to make sure that

$$L = L_u + L_r$$

Of course the total of land uses cannot exceed available land, $S$, so we need to impose that

$$S_r + S_{rh} + \phi = S$$

Finally, we need a condition that tells us how big the lump sum land rent disbursement will be. The total amount of land rent collected is $\rho  (S_r + S_{rh} + \phi)$, hence the per capita rebate will be

$$r = \rho  (S_r + S_{rh} + \phi) / L$$

The Feasibility constraint relates to the fact that we chose the urban good to serve as the numeraire good in this economy. So we need to make sure that there is enough urban good produced to satisfy all the demands for it in this world. Notice that we need urban good here for direct consumption ($c_u$) *and* to build houses (!). 

The construction cost to be paid for in numeraire by the developer *per unit of housing* is

$$\frac{H^{1+1/\epsilon}}{1+1/\epsilon}$$

and notice that $H^{1+1/\epsilon} = H H^{1/\epsilon}$, further that $q = H^{1/\epsilon}$ as per optimal housing supply, and thus 

$$\frac{H^{1+1/\epsilon}}{1+1/\epsilon} = \frac{Hq}{1+1/\epsilon} = \frac{\epsilon}{1+\epsilon} Hq = \frac{\epsilon}{1+\epsilon} q^{1+\epsilon} =\frac{\epsilon}{1+\epsilon} (1+\epsilon) \rho = \epsilon \rho.$$

That means that we have a feasiblity constraint of

$$Y_u/L = c_u + \rho \epsilon (S_{rh} + \phi) / L$$
 
"""

# ╔═╡ 72c5f230-0e15-4baf-bd2a-91fea0a9c434
md"""
## Solving This Model

The above model is described by the consumption rules for consmers, the first order conditions for the rural firm and the constraints we have placed on the solution arising from finite land use. 

How to solve this model?

One approach is to realize that given $L,S$ and parameters $\alpha, \gamma,\nu$, the only things that determine the outcomes here are 

1. the number of people in either sector, $L_r$, say
2. the value of land, $\rho$. 

If $L_r$ is smaller, we need more space for the city, but we have less output in the rural sector. If $\rho$ is larger, housing demand will go down, but rent rebates $r$ will go up, also, the city will be smaller. Notice that all other quantities in the model are implied once the pair $(\rho, L_r)$ have been chosen.

Therefore, our approach will be to propose two numbers $(\rho, L_r)$, determine all quantities in the model, and check whether we clear the markets. Notice, that we can focus only on the land market clearing and the feasibility constraint, as all others are embedded in those.
"""

# ╔═╡ c104e2ec-67f5-4e8c-9bfb-fba51b47cd59
par = (α = 0.1, γ = 0.3, ν = 0.2, θᵤ = 1.0, θᵣ = 1.0, L = 2.0, S = 1.0, ϵ = 4.0)

# ╔═╡ 4cd4c450-237c-464a-bd6b-0cd325a97a30
function compute_model!(F::Vector,x::Vector,p::NamedTuple)
	# notice we call p the parameters and pr the relative price of rural good.
	if any(x .< 0)
		F[:] .= 100.0
	else
		# define choice variables
		ρ = x[1]
		Lᵣ = x[2]
	
		# implied quantities
		Lᵤ = p.L - Lᵣ   # this clears the labor market
		w = p.θᵤ  # wage in both sectors equal to that
		Sᵣ = w / ρ * ((1-p.α)/p.α ) * Lᵣ  # from rural FOC
		pr = w / (p.α * p.θᵣ) * (Lᵣ / Sᵣ)^(1-p.α)  # also
		r = ρ * (p.S / p.L)  # total rent collected per capita
	
		# housing demand in each sector
		γϵ = par.γ / (1+p.ϵ)
		ϕ   = Lᵤ * γϵ * (w + r) / ρ  # space need for city housing
		Sᵣₕ = Lᵣ * γϵ * (w + r) / ρ  # space for rural housing

		# total construction costs
		ccost = p.ϵ * ρ * (Sᵣₕ + ϕ)
		
		# constraints to obey. each equation should be (close to) zero
		# 1. feasibilty on urban good:
		# (urban good consumption) + (construction cost) = urban production
		# notice this is in per capita terms
		F[1] = (1 - p.γ) * (1 - p.ν) * (w + r) + ccost / p.L - Lᵤ * p.θᵤ / p.L
		# 2. Land market clearing
		F[2] = Sᵣ + Sᵣₕ + ϕ - p.S
	end

	# returns nothing but modified F

end

# ╔═╡ 98c1305e-ad91-46a0-a910-47c48512e2c8
r = nlsolve( (F,x) -> compute_model!(F,x,par), [1.0, 0.5] );

# ╔═╡ 788fc4e1-836d-4da8-a3fc-8865a3442f31
function getresult(x,p)
	ρ = x[1]
	Lᵣ = x[2]
	γϵ = p.γ / (1+p.ϵ)

	# implied quantities
	Lᵤ = p.L - Lᵣ   # this clears the labor market
	w = p.θᵤ  # wage in both sectors equal to that
	Sᵣ = w / ρ * ((1-p.α)/p.α ) * Lᵣ  # from rural FOC
	pr = w / (p.α * p.θᵣ) * (Lᵣ / Sᵣ)^(1-p.α)  # also
	r = ρ * (p.S / p.L)  # total rent collected per capita

	# housing demand in each sector
	ϕ   = Lᵤ * γϵ * (w + r) / ρ  # space need for city housing
	Sᵣₕ = Lᵣ * γϵ * (w + r) / ρ  # space for rural housing

	# check land market clearing
	Yᵣ = p.θᵣ * (Lᵣ^p.α * Sᵣ^(1-p.α) )
	check = (1 - p.γ) * p.ν * (w + r) - pr * Yᵣ / p.L

	(ρ = ρ, Lᵤ = Lᵤ, Lᵣ = Lᵣ, ϕ = ϕ, Sᵣₕ = Sᵣₕ, Sᵣ = Sᵣ, p = p, pr = pr, walrascheck = check)

end

# ╔═╡ f8804a53-92e4-47ad-82c8-3914e49f00ed
getresult(r.zero,par)

# ╔═╡ aa32ac11-58b7-454d-bdec-7ca76d258468


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"

[compat]
NLsolve = "~4.5.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.1"
manifest_format = "2.0"
project_hash = "93f7b991e44b933f76d3fd91b5660258a7d76ca6"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cc37d689f599e8df4f464b2fa3870ff7db7492ef"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "a89acc90c551067cd84119ff018619a1a76c6277"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.2.1"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "89a9db8d28102b094992472d333674bd1a83ce2a"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.1"

    [deps.ConstructionBase.extensions]
    IntervalSetsExt = "IntervalSets"
    StaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "a4ad7ef19d2cdc2eff57abbbe68032b1cd0bd8f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.13.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "49eba9ad9f7ead780bfb7ee319f962c811c6d3b2"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.8"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "ed1b56934a2f7a65035976985da71b6a65b4f2cf"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.18.0"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NLsolve]]
deps = ["Distances", "LineSearches", "LinearAlgebra", "NLSolversBase", "Printf", "Reexport"]
git-tree-sha1 = "019f12e9a1a7880459d0173c182e6a99365d7ac1"
uuid = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
version = "4.5.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "7756ce473bd10b67245bdebdc8d8670a85f6230b"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.18"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─84674032-4848-11ec-3ead-092daf45ee6a
# ╟─607a4845-02f1-4bc0-8e44-6951d33c4367
# ╟─4a6cbff6-3c1a-414a-a866-599cb5df177c
# ╟─c5255877-e38d-4614-9ea6-7678d13ebb06
# ╟─59fb80db-10c5-4cca-8f85-be7599320242
# ╟─72c5f230-0e15-4baf-bd2a-91fea0a9c434
# ╠═c3572e14-c058-455c-96c0-6324970af323
# ╠═4cd4c450-237c-464a-bd6b-0cd325a97a30
# ╠═c104e2ec-67f5-4e8c-9bfb-fba51b47cd59
# ╠═98c1305e-ad91-46a0-a910-47c48512e2c8
# ╠═f8804a53-92e4-47ad-82c8-3914e49f00ed
# ╟─788fc4e1-836d-4da8-a3fc-8865a3442f31
# ╠═aa32ac11-58b7-454d-bdec-7ca76d258468
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
