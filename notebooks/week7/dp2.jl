### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 36610294-87dc-11eb-21e8-5b1dd06c6020
begin
	using PlutoUI
	using Plots
	using Optim
	using Interpolations
	using Roots
	using LaTeXStrings
end

# ╔═╡ 5fab5e80-87ce-11eb-111a-d5288227b97c
html"<button onclick='present()'>present</button>"

# ╔═╡ 705e96f2-87ce-11eb-3f5a-eb6cdb8c49d4
md"
# Dynamic Programming 2

> Florian Oswald, SciencesPo 2021


* In this notebook we'll add uncertainty to our simple setup from the last time.
* We'll stay in the finite time framework, but everything we say will be easily portable to infinite time. 
* We'll write code (of course). We won't worry about performance so far, but try to be as explicit as possible.
* We'll do something that looks reasonable but works terribly. 
* Then we'll fix it and celebrate! 🎉


"

# ╔═╡ 34403942-87d2-11eb-1172-7bd285bf7d75
md"
#

* A good place to start is a so called decision tree. 
![](https://upload.wikimedia.org/wikipedia/commons/4/48/DecisionCalcs.jpg)"

# ╔═╡ b9ed265e-87d2-11eb-1028-05b4c4ccbc74
md"

## Markov Decision Problems (MDPs)

* Assume a finite set of states $S$
* A finite set of actions $A$
* For state $s\in S$ and action $a \in A$ we can compute the *cost* or the *contribution* $C(s,a)$
* There is a *transition matrix* $p_t(S_{t+1} | S_{t}, a_t)$ which tells us with which probability we go from state $S_{t}$ to $S_{t+1}$ upen taking decision $a_t$
* Even though many problems have *continuous* state and action spaces, this is a very useful class of problems to study.
"

# ╔═╡ 07992abe-87d4-11eb-0fd7-1bef7fc2ada8
md"
## Optimality (Bellman) Equations

* But let's start with _no uncertainty_. If we are at state $S_t$, we know how $S_{t+1}$ will look like with certainty if we choose a certain action $a_t$ 
* Then we just choose the best action:
$$a_t^*(S_t) = \arg \max_{a_t \in A_t} \left(C_t(S_t,a_t) + \beta V_{t+1}(S_{t+1}(S_t,a_t)) \right)$$
* The associated *value* of this choice is then the same Bellman Equation we know from last time,
$$\begin{align}V_t(S_t) &= \max_{a_t \in A_t} \left(C_t(S_t,a_t) + \beta V_{t+1}(S_{t+1}(S_t,a_t)) \right) \\ &= C_t(S_t,a^*_t(S_t)) + \beta V_{t+1}(S_{t+1}(S_t,a^*_t(S_t)))\end{align}$$
"

# ╔═╡ 93148b06-87d2-11eb-2356-fb460acf8e66
md"

## Enter Uncertainty

* Uncertainty comes into play if we *don't know exactly what state* $S_{t+1}$ we'll end up in.
* We may have partial information on the likelihood of each state occuring, like a belief, or prior information.
* We have our transition matrix $p_t(S_{t+1} | S_{t}, a_t)$, and can define the *standard form of the Bellman Equation*:

$$V_t(S_t) = \max_{a_t \in A_t} \left(C_t(S_t,a_t) + \beta \sum_{s' \in S} p_t\left( S_{t+1} = s' | S_t, a_t \right)V_{t+1}(s') \right)$$
"

# ╔═╡ 3d97787a-87d2-11eb-3713-ab3dad6e5e8b
md"
#

* we can also write it more general without the summation
$$V_t(S_t) = \max_{a_t \in A_t} \left(C_t(S_t,a_t) + \beta \mathbb{E} \left[ V_{t+1}(S_{t+1}(S_t,a_t)) | S_t \right] \right)$$"

# ╔═╡ a41aff3a-87d8-11eb-00fe-e16c14870ba2
md"

## Let's Do it!

* Let's look at this specific problem where the *action* is *consumption*, and $u$ is a *utility function* and the state is $R_t$:
$$V_t(R_t) = \max_{0 \leq c_t \leq R_t} \left(u(c_t) + \beta \mathbb{E} \left[ V_{t+1}(R_{t+1}(R_t,c_t)) |R_t \right] \right)$$

* Here $R_t$ could be available cash on hand of the consumer.
* Let us specify a *law of motion* for the state as 
$$R_{t+1} = R_t - c_t + y_{t+1}$$ where $y_t$ is a random variable.
* We have to compute expectations wrt to $y$

"

# ╔═╡ c6ed3824-87d9-11eb-3e0a-b5a473fdb952
md"

## I.I.D. Shock

* Let's start with the simplest case: $y_t \in \{\overline{y}, \underline{y}\}, 0 < \overline{y} < \underline{y}$
* There is a probability π associated to $P(y = \overline{y}) = \pi$
* There is *no state dependence* in the $y$ process: the current $y_t$ does not matter for the future $y_{t+1}$

"

# ╔═╡ 51983542-87f9-11eb-226a-81fe383f0763
md"
#
"

# ╔═╡ 6ddfc91c-87da-11eb-3b5c-dfe006576f29
"""
	Bellman(grid::Vector,vplus::Vector,π::Float64,yvec::Vector)

Given a grid and a next period value function `vplus`, and a probability distribution
calculate current period optimal value and actions.
"""
function Bellman(grid::Vector,vplus::Vector,π::Float64,yvec::Vector{Float64},β::Float64)
	points = length(grid)
	w = zeros(points) # temporary vector for each choice or R'
	Vt = zeros(points) # optimal value in T-1 at each state of R
	ix = 0 # optimal action index in T-1 at each state of R
	at = zeros(points) # optimal action in T-1 at each state of R

	for (ir,r) in enumerate(grid) # for all possible R-values
		# loop over all possible action choices
		for (ia,achoice) in enumerate(grid)
			if r <= achoice   # check whether that choice is feasible
				w[ia] = -Inf
			else
				rlow = r - achoice + yvec[1] # tomorrow's R if y is low
				rhigh  = r - achoice + yvec[2] # tomorrow's R if y is high
				jlow = argmin(abs.(grid .- rlow))  # index of that value in Rspace
				jhigh = argmin(abs.(grid .- rhigh))  # index of that value in Rspace
				w[ia] = sqrt(achoice) + β * ((1-π) * vplus[jlow] + (π) * vplus[jhigh] ) # value of that achoice
			end
		end
		# find best action
		Vt[ir], ix = findmax(w) # stores Value und policy (index of optimal choice)
		at[ir] = grid[ix]  # record optimal action level
	end
	return (Vt, at)
end

# ╔═╡ aeb2b692-87db-11eb-3c9a-114acdc3d159
md"
#
"

# ╔═╡ b3f20522-87db-11eb-325d-7ba201c66452
begin
	points = 500
	lowR = 0.01
	highR = 10.0
	# more points towards zero to make nicer plot
	Rspace = exp.(range(log(lowR), stop = log(highR), length = points))
	aT = Rspace # consume whatever is left
	VT = sqrt.(aT)  # utility of that consumption
	yvec = [1.0, 3.0]
	nperiods = 10
	β = 1.0  # irrelevant for now
	π = 0.7
end

# ╔═╡ 6e8aec5e-87f9-11eb-2e53-bf5ba90fa750
md"
#
"

# ╔═╡ f00890ee-87db-11eb-153d-bb9244b46e89
# identical
function backwards(grid, nperiods, β, π, yvec)
	points = length(grid)
	V = zeros(nperiods,points)
	c = zeros(nperiods,points)
	V[end,:] = sqrt.(grid)  # from before: final period
	c[end,:] = collect(grid)

	for it in (nperiods-1):-1:1
		x = Bellman(grid, V[it+1,:], π, yvec, β)	
		V[it,:] = x[1]
		c[it,:] = x[2]
	end
	return (V,c)
end

# ╔═╡ 4da91e08-87dc-11eb-3238-433fbdda7df7
V,a = backwards(Rspace, nperiods, β, π, yvec);

# ╔═╡ 7a97ba6a-87f9-11eb-1c88-a999bcff4647
md"
#
"

# ╔═╡ ca3e6b12-87dc-11eb-31c6-8d3be744def0
let
	cg = cgrad(:viridis)
    cols = cg[range(0.0,stop=1.0,length = nperiods)]
	pa = plot(Rspace, a[1,:], xlab = "R", ylab = "Action",label = L"a_1",leg = :topleft, color = cols[1])
	for it in 2:nperiods
		plot!(pa, Rspace, a[it,:], label = L"a_{%$(it)}", color = cols[it])
	end
	pv = plot(Rspace, V[1,:], xlab = "R", ylab = "Value",label = L"V_1",leg = :bottomright, color = cols[1])
	for it in 2:nperiods
		plot!(pv, Rspace, V[it,:], label = L"V_{%$(it)}", color = cols[it])
	end
	plot(pv,pa, layout = (1,2))
end

# ╔═╡ 027875e0-87e0-11eb-0ca5-317b58b0f8fc
PlutoUI.Resource("https://i.ytimg.com/vi/wAbnNZDhYrA/maxresdefault.jpg")

# ╔═╡ 48cc5fa2-87dd-11eb-309b-9708941fd8d5
md"
#

* Oh wow! That's like...total disaster!
* 😩
* This way of interpolating the future value function does *not work very well*
* Let's use proper interpolation instead!
* Instead of `vplus[jlow]` we want to have `vplus[Rprime]` where `Rprime` $\in \mathbb{R}$!

#

"

# ╔═╡ 04b6c4b8-87df-11eb-21dd-15a443042bd1
"""
	Bellman(grid::Vector,vplus::Vector,π::Float64,yvec::Vector)

Given a grid and a next period value function `vplus`, and a probability distribution
calculate current period optimal value and actions.
"""
function Bellman2(grid::Vector,vplus::Vector,π::Float64,yvec::Vector,β::Float64)
	points = length(grid)
	w = zeros(points) # temporary vector for each choice or R'
	Vt = zeros(points) # optimal value in T-1 at each state of R
	ix = 0 # optimal action index in T-1 at each state of R
	at = zeros(points) # optimal action in T-1 at each state of R
	
	# interpolator
	vitp = interpolate((grid,), vplus, Gridded(Linear()))
	vitp = extrapolate(vitp, Interpolations.Flat())

	for (ir,r) in enumerate(grid) # for all possible R-values
		# loop over all possible action choices
		for (ia,achoice) in enumerate(grid)
			if r <= achoice   # check whether that choice is feasible
				w[ia] = -Inf
			else
				rlow = r - achoice + yvec[1] # tomorrow's R if y is low
				rhigh  = r - achoice + yvec[2] # tomorrow's R if y is high
				w[ia] = sqrt(achoice) + (1-π) * vitp(rlow) + (π) * vitp(rhigh)  # value of that achoice
			end
		end
		# find best action
		Vt[ir], ix = findmax(w) # stores Value und policy (index of optimal choice)
		at[ir] = grid[ix]  # record optimal action level
	end
	return (Vt, at)
end

# ╔═╡ 99b19ed4-92c8-11eb-2daa-2df3db8736bd
md"
#
"

# ╔═╡ 43539482-87e0-11eb-22c8-b9380ae1ebdb
function backwards2(grid, nperiods, π, yvec, β)
	points = length(grid)
	V = zeros(nperiods,points)
	a = zeros(nperiods,points)
	V[end,:] = sqrt.(grid)  # from before: final period
	a[end,:] = collect(grid)

	for it in (nperiods-1):-1:1
		x = Bellman2(grid, V[it+1,:], π, yvec, β)	
		V[it,:] = x[1]
		a[it,:] = x[2]
	end
	return (V,a)
end

# ╔═╡ a361963a-92c8-11eb-0543-7decc61720a8
md"
#
"

# ╔═╡ 5f764692-87df-11eb-3b9d-e5823eb4d1b3
let
	V,a = backwards2(Rspace,nperiods,π,yvec,β)
	cg = cgrad(:viridis)
    cols = cg[range(0.0,stop=1.0,length = nperiods)]
	pa = plot(Rspace, a[1,:], xlab = "R", ylab = "Action",label = L"a_1",leg = :topleft, color = cols[1])
	for it in 2:nperiods
		plot!(pa, Rspace, a[it,:], label = L"a_{%$(it)}", color = cols[it])
	end
	pv = plot(Rspace, V[1,:], xlab = "R", ylab = "Value",label = L"V_1",leg = :bottomright, color = cols[1])
	for it in 2:nperiods
		plot!(pv, Rspace, V[it,:], label = L"V_{%$(it)}", color = cols[it])
	end
	plot(pv,pa, layout = (1,2))
end

# ╔═╡ 8eb731f6-87e0-11eb-35dd-61ba070afc8b
md"

#

* Well at least a little bit better. 
* Still, what about those steps in there? What's going on?
"

# ╔═╡ 02d26410-87e2-11eb-0278-99ee0bbd2923
md"

## Continuous Choice

* In fact our action space is continuous here:
$$V_t(R_t) = \max_{0 \leq a_t \leq R_t} \left(u(a_t) + \beta \mathbb{E} \left[ V_{t+1}(R_{t+1}(R_t,a_t)) |R_t \right] \right)$$

* So let's treat it as such. We could direct optimization for the Bellman Operator!

#

"

# ╔═╡ 3b1f5120-87e2-11eb-1b10-85900b6fbeb6
function bellman3(grid,v0,β::Float64, π, yvec)
    n = length(v0)
    v1  = zeros(n)     # next guess
    pol = zeros(n)     # consumption policy function

    Interp = interpolate((collect(grid),), v0, Gridded(Linear()) ) 
    Interp = extrapolate(Interp,Interpolations.Flat())

    # loop over current states
    # of current resources
    for (i,r) in enumerate(grid)

        objective(c) = - (sqrt.(c) + β * (π * Interp(r - c + yvec[2]) + (1-π) * Interp(r - c + yvec[1])) )
        res = optimize(objective, 1e-6, r)  # search in [1e-6,r]
        pol[i] = res.minimizer
        v1[i] = -res.minimum
    end
    return (v1,pol)   # return both value and policy function
end


# ╔═╡ c6990bb0-92c8-11eb-2523-9f8a7ec1cd4a
md"
#
"

# ╔═╡ 0b83e920-87e3-11eb-0792-479eb843b429
function backwards3(grid, nperiods,β, π, yvec)
	points = length(grid)
	V = zeros(nperiods,points)
	a = zeros(nperiods,points)
	V[end,:] = sqrt.(grid)  # from before: final period
	a[end,:] = collect(grid)

	for it in (nperiods-1):-1:1
		x = bellman3(grid, V[it+1,:],β,  π, yvec)	
		V[it,:] = x[1]
		a[it,:] = x[2]
	end
	return (V,a)
end

# ╔═╡ d391139e-92c8-11eb-0888-b325a75fb10f
md"
#
"

# ╔═╡ 7003b4e8-87e3-11eb-28a9-f7e3668beac3
let
	V,a = backwards3(Rspace,nperiods,β,π,yvec)
	cg = cgrad(:viridis)
    cols = cg[range(0.0,stop=1.0,length = nperiods)]
	pa = plot(Rspace, a[1,:], xlab = "R", ylab = "Action",label = L"a_1",leg = :topleft, color = cols[1])
	for it in 2:nperiods
		plot!(pa, Rspace, a[it,:], label = L"a_{%$(it)}", color = cols[it])
	end
	pv = plot(Rspace, V[1,:], xlab = "R", ylab = "Value",label = L"V_1",leg = :bottomright, color = cols[1])
	for it in 2:nperiods
		plot!(pv, Rspace, V[it,:], label = L"V_{%$(it)}", color = cols[it])
	end
	plot(pv,pa, layout = (1,2))
end

# ╔═╡ 4504d988-87e4-11eb-05d5-c9f9f215f785
md"

#

* It's getting there!
* Much fewer wiggles in the consumption function.
* But we can do even better than that.
* Let's do *Policy Iteration* rather than *Value Iteration*!
"

# ╔═╡ 6828b538-87e4-11eb-3cdd-71c31dad5a6e
md"

## Policy Iteration

* In most economic contexts, there is more structure on a problem.
* In our consumption example, we know that the Euler Equation must hold.
* Sometimes there are other *optimality conditions* that one could leverage.
* In our case, we have this equation at an optimal policy $c^*(R_t)$:
$$\begin{align}
u'(c^*(R_t)) & = \beta \mathbb{E} \left[ u'(c(R_{t+1})) \right] \\
             & = \beta \mathbb{E} \left[ u'(R_t - c^*(R_t) + y)   \right]
\end{align}$$
* So: We *just* have to find the function $c^*(R_t)$!
"

# ╔═╡ ba19759c-87eb-11eb-1bec-51de9ac10e31
md"

## *Just*?

* There is a slight issue here. The Euler Equation in itself does not enforce the contraint that we cannot borrow any $R$. 
* In particular, $R_{t} \geq 0$ is missing from the above equation.
* look back above at the last plot of the consumption function. It has a kink!
* The Euler Equation applies only to the *right* of that kink!
* If tomorrow's consumption is low (we are saving a lot), today's consumption is relatively high, implying that we consume more than we have in terms of $R_t$.
* Satisfying the Euler Equation would require to set $R_t <0$ sometimes, which is ruled out.


"

# ╔═╡ a4018f28-87ec-11eb-07f0-af86523dd26e
md"
#

* So if we want to use the Euler Equation, we need to manually adjust for that.
* 😩
* Alright, let's do it. It's not too bad in the end.

#

* First, let's put down the marginal utility function and it's inverse:

"

# ╔═╡ 027fda74-87f1-11eb-1441-55d6e410bf4c
u_prime(c) = 0.5 .* c.^(-0.5)

# ╔═╡ 540d8814-87f1-11eb-0b8c-23357c46f93c
u_primeinv(u) = (2 .* u).^(-2)

# ╔═╡ d2788ffe-92c4-11eb-19e7-4b41d9f9ebdd
md"

#

* We need the _inverse_ of the marginal utility function?
* Yes. Here is why. For any candidate consumption level $c_t$
$$\begin{align}
u'(c_t) & = \beta \mathbb{E} \left[ u'(c(R_{t+1})) \right] \\
u'(c_t) & = \text{RHS}
\end{align}$$
* So, if we have computed the RHS, current period optimal consumption is just
$$c_t  = \left( u' \right)^{-1}\left(\text{RHS}\right)\hspace{1cm}(\text{EEresid})$$

where $\left( u' \right)^{-1}(z)$ denotes the _inverse_ of function $u'(x)$. 

#

### Example

* Suppose tomorrow's consumption function _does_ have the kink at $R_{t+1} = 1.5$
* I.e. $c_{t+1}^*(R_{t+1}) = R_{t+1},\forall R_{t+1} \leq 1.5$
* Let $y_{t+1} = 1.5, R_t = 1$. Now try out consumption choice $c_t = 1$!

$$\begin{align}
R_{t+1} &= R_t - c_t+ y_{t+1} \\
        &= 1 - 1 + 1.5 = 1.5 \\
\Rightarrow c_{t+1}^*(1.5) &= 1.5
\end{align}$$ 

because right at the kink! 
* Then, the Euler equation says: 
$$\begin{align}
u'(c_t) & = u'(c_{t+1}) \\
c_t & = (u')^{-1}\left(u'(c_{t+1})\right) \\
c_t & = c_{t+1} = 1.5
\end{align}$$
* Well, that in turn means that $R_t - c_t = 1 - 1.5 < 0$, i.e. we would have to borrow 0.5 units of $R$ in order to satisfy that Euler Equation!

#
"

# ╔═╡ c75eed74-87ee-11eb-3e9a-3b893294baec
function EEresid(ct::Number,        # candidate current (t) consumption choice 
		         Rt::Number,        # current level of resources
				 cfunplus::Vector,  # next period's consumption *function*
		         grid::Vector,      # grid values on which cfunplus is defined
				 yvec::Vector,      # vector of values for shock
		         β::Number,         # discount factor
				 π::Number)         # prob of high y
	
	# next period resources, given candidate choice ct
	Rplus = Rt - ct .+ yvec  # a (2,1) vector: Rplus for each income level

	# get implied next period consumption from cplus
	# we add point (0,0) here to make sure that this is part of the grid.
	citp = extrapolate(
				interpolate(([0.0, grid...],), [0.0,cfunplus...], Gridded(Linear())), Interpolations.Flat())
	cplus = citp.(Rplus)
	RHS = β * [1-π  π] * u_prime(cplus) # expecte marginal utility of tomorrows consumption
	
	# euler residual: expression EEresid from above
	r = ct .- u_primeinv(RHS)
	r[1]  # array of size 1
end    

# ╔═╡ ae56f4e2-87f9-11eb-10fc-e3eda66e8a1f
md"
#
"

# ╔═╡ 17ea7f6a-87e7-11eb-2d3e-b9a1e771b08e
function policy_iter(grid,c0,u_prime,β::Float64, π, yvec)
    
    c1  = zeros(length(grid))     # next guess
    # loop over current states
    # of current resources
    for (i,r) in enumerate(grid)
		# get euler residual if we consume all resources
		res = EEresid(r,r,c0, grid, yvec,β , π)
		if res < 0
			# we consume too little today, c_t is too small for the Euler Equation.
			# could only make it bigger by borrowing next period.
			# but we cant! so really we are consuming all we have:
			c1[i] = r
		else
			# no problem here: Euler Equation holds
			# just need to find that ct that makes it zero:
			c1[i] = fzero( x-> EEresid(x,r,c0, grid, yvec, β, π) , 1e-6, r)
		end
    end
    return c1
end

# ╔═╡ bcca454c-87f9-11eb-0670-0d85db6b6e37
md"
#
"

# ╔═╡ c1e65340-87f9-11eb-3cd7-05f14edf71d2
md"
#
"

# ╔═╡ 6fbaac56-87e0-11eb-3d09-39a4fd293e88
PlutoUI.Resource("https://s3.getstickerpack.com/storage/uploads/sticker-pack/meme-pack-1/sticker_19.png?363e7ee56d4d8ad53813dae0907ef4c0&d=200x200")

# ╔═╡ a510ff40-9302-11eb-091e-6929367f6783
md"

# Infinite Time Problems

* Let's now introduce the infinite time version of this.
* For that purpose, let's rely on the well known Optimal Growth model:

$$\begin{align}
   V(k) &= \max_{0<k'<f(k)} u(f(k) - k') + \beta V(k')\\
  f(k)  & = k^\alpha\\
  k_0   & \text{ given} 
\end{align}$$

* The solution in finite time was simple by going backwards.
* Here, we need to rely on results from Functional Analysis: Remember the Contraction Mapping Theorem?
* The Contraction Mapping Theorem or the [Banache Fixed Point Theoreom](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem) says there is a unique fixed point in an appropriately chosen function space. We arrive at it from an *arbitrary* starting point by just iterating on the operator $T(V)$ until $V = T(V)$
"


# ╔═╡ 06ff3932-9304-11eb-07cc-d798d56c0931
md"

# Implementing Value Function Iteration

### Checklist

1. Set parameter values
1. define a grid for state variable $k \in [0,2]$
1. initialize value function $V$
1. start iteration, repeatedly computing a new version of $V$.
1. stop if $d(V^{r},V^{r-1}) < \text{tol}$.
1. plot value and policy function 
1. report the maximum error of both wrt to analytic solution

#
"

# ╔═╡ 2e121c60-9304-11eb-29c4-afd0a289343f
begin
	alpha     = 0.65
	beta      = 0.95
	grid_max  = 2  # upper bound of capital grid
	n         = 150  # number of grid points
	N_iter    = 3000  # number of iterations
	kgrid     = 1e-2:(grid_max-1e-2)/(n-1):grid_max  # equispaced grid
	f(x) = x^alpha  # defines the production function f(k)
	tol = 1e-9
end

# ╔═╡ 3c8e560a-9304-11eb-2c45-7df67768f75b
md"

## Analytic Solution

* If we choose $u(x)=\ln(x)$, the problem has a closed form solution.
* We can use this to check accuracy of our solution.
"

# ╔═╡ 4849479a-9304-11eb-1c06-4b53a4163f85
begin 
	ab        = alpha * beta
	c1        = (log(1 - ab) + log(ab) * ab / (1 - ab)) / (1 - beta)
	c2        = alpha / (1 - ab)
	# optimal analytical values
	v_star(k) = c1 .+ c2 .* log.(k)  
	k_star(k) = ab * k.^alpha   
	c_star(k) = (1-ab) * k.^alpha  
	ufun(x) = log.(x)
end

# ╔═╡ 5984cd7a-9304-11eb-0bd4-ebac2da4f300
md"
#
"

# ╔═╡ 5d642bca-9304-11eb-2b23-dfa47b04bb22
# Bellman Operator
# inputs
# `grid`: grid of values of state variable
# `v0`: current guess of value function

# output
# `v1`: next guess of value function
# `pol`: corresponding policy function 

#takes a grid of state variables and computes the next iterate of the value function.
function bellman_operator(grid,v0)
    
    v1  = zeros(n)     # next guess
    pol = zeros(Int,n)     # policy function
    w   = zeros(n)   # temporary vector 

    # loop over current states
    # current capital
    for (i,k) in enumerate(grid)

        # loop over all possible kprime choices
        for (iprime,kprime) in enumerate(grid)
            if f(k) - kprime < 0   #check for negative consumption
                w[iprime] = -Inf
            else
                w[iprime] = ufun(f(k) - kprime) + beta * v0[iprime]
            end
        end
        # find maximal choice
        v1[i], pol[i] = findmax(w)     # stores Value und policy (index of optimal choice)
    end
    return (v1,grid[pol])   # return both value and policy function
end

# ╔═╡ 67d367c4-9304-11eb-0bfc-21b9bfec2fb9
md"
#
"

# ╔═╡ 6b33c6ca-9304-11eb-3766-fdcc42b64f2d
# VFI iterator
#
## input
# `n`: number of grid points
# output
# `v_next`: tuple with value and policy functions after `n` iterations.
function VFI(op::Function)
    v_init = zeros(n)     # initial guess
    for iter in 1:N_iter
        v_next = op(kgrid,v_init)  # returns a tuple: (v1,pol)
        # check convergence
        if maximum(abs,v_init.-v_next[1]) < tol
            verrors = maximum(abs,v_next[1].-v_star(kgrid))
            perrors = maximum(abs,v_next[2].-k_star(kgrid))
            return (v = v_next[1], p =v_next[2], errv = verrors, errp = perrors, iter = iter)
        elseif iter==N_iter
            @warn "No solution found after $iter iterations"
            return (v = v_next[1], p =v_next[2], errv = verrors, errp = perrors, iter = iter)
        end
        v_init = v_next[1]  # update guess 
    end
end



# ╔═╡ 72cc3cdc-9304-11eb-3548-95962c3513ec
md"
#
"

# ╔═╡ 776c4712-9304-11eb-1824-a14cde26b895
function plotVFI(v::NamedTuple)
    
    p = Any[]
	    # errors of both
	if eltype(v.p) == Int
		policy = kgrid[v.p]
	else
		policy = v.p
	end
    
    # value and policy functions
	if !isnan(v.v[1])
		push!(p,plot(kgrid,v.v,
            leg = false,title = "Vfun iterations: $(v.iter)",
            ylim=(-50,-30)))
		push!(p,plot(kgrid,v.v .- v_star(kgrid),
        title = "max Vfun error: $(round(v.errv,digits=3))"))
		perrors = policy .- k_star(kgrid)
	else
		push!(p,plot(title = "Vfun iterations: $(v.iter)"), plot())
		perrors = policy .- c_star(kgrid)
		
	end
    push!(p,plot(kgrid,policy,
            title = "policy function"))
    push!(p,plot(kgrid,perrors,
        title = "max policy error: $(round(v.errp,digits=3))"))

    plot(p...,layout=grid(2,2) , leg = false)
    
end

# ╔═╡ 7999a272-9304-11eb-254c-af591bae0620
plotVFI(VFI(bellman_operator))

# ╔═╡ a4ba5a96-ad59-4319-bd94-636f525853c6
function VFI_converge(op::Function,steps)
    v_init = zeros(n)     # initial guess
	pl = plot(kgrid, v_star(kgrid), color = :red, label = "true")
    for iter in 1:steps
		plot!(pl, kgrid, v_init, label = "", color = :grey)
        v_next = op(kgrid,v_init)  # returns a tuple: (v1,pol)
        # check convergence
        if maximum(abs,v_init.-v_next[1]) < tol
            verrors = maximum(abs,v_next[1].-v_star(kgrid))
            perrors = maximum(abs,v_next[2].-k_star(kgrid))
            return (v = v_next[1], p =v_next[2], errv = verrors, errp = perrors, iter = iter)
        elseif iter==N_iter
            @warn "No solution found after $iter iterations"
            return (v = v_next[1], p =v_next[2], errv = verrors, errp = perrors, iter = iter)
        end
        v_init = v_next[1]  # update guess 
    end
	pl
end

# ╔═╡ d52d3d31-e447-4120-92ae-3d4da7a02219
md"""
steps = $(@bind stps Slider(1:100, show_value = true, default = 1))
"""

# ╔═╡ c881819e-ca81-40b0-9e0f-93abcb43a2b4
VFI_converge(bellman_operator, stps)

# ╔═╡ 84e23d2a-9388-11eb-2483-3342d1683129
md"

## continous choice

* like before, now let's treat the choice dimension as a continuous variable
* This is almost identical to before with finite time.

#
"

# ╔═╡ 7e0a9d82-9304-11eb-1d3d-bb47fc55d033
function bellman_operator2(grid,v0)
    
    v1  = zeros(n)     # next guess
    pol = zeros(n)     # consumption policy function

    Interp = interpolate((collect(grid),), v0, Gridded(Linear()) ) 
    Interp = extrapolate(Interp,Interpolations.Flat())

    # loop over current states
    # of current capital
    for (i,k) in enumerate(grid)

        objective(c) = - (log.(c) + beta * Interp(f(k) - c))
        # find max of ojbective between [0,k^alpha]
        res = optimize(objective, 1e-6, f(k))  # Optim.jl
        pol[i] = f(k) - res.minimizer   # k'
        v1[i] = -res.minimum
    end
    return (v1,pol)   # return both value and policy function
end

# ╔═╡ aa891724-9388-11eb-32bb-8f8d15dc3761
md"
#
"

# ╔═╡ 597080ba-9307-11eb-2780-87474bfc5cff
plotVFI(VFI(bellman_operator2))

# ╔═╡ 6f9b2b8a-9389-11eb-2731-398e1342538a
md"
#

* Finally, policy function iteration!
* Again, remember the euler Equation here.

#
"

# ╔═╡ 710a0212-9307-11eb-2cf3-e75b9c5eddab
begin
	uprime(x) = 1.0 ./ x
	fprime(x) = alpha * x.^(alpha-1)
end

# ╔═╡ 67c4bbb8-9307-11eb-3ecb-13d7af9d899a
function policy_iter(grid,c0,u_prime,f_prime)

	c1  = zeros(length(grid))     # next guess
	pol_fun = extrapolate(interpolate((collect(grid),), c0, Gridded(Linear()) ) , Interpolations.Flat())

	# loop over current states
	# of current capital
	for (i,k) in enumerate(grid)
		objective(c) = u_prime(c) - beta * u_prime(pol_fun(f(k)-c)) * f_prime(f(k)-c)
		c1[i] = fzero(objective, 1e-10, f(k)-1e-10) 
	end
	return c1
end



# ╔═╡ d18511ea-87e7-11eb-08ce-a9176a32fbd1
function backwards_pol(grid, nperiods,β, π, yvec)
	u_prime(c) = 0.5 * c^(-0.5)
	points = length(grid)
	V = zeros(nperiods,points)
	c = zeros(nperiods,points)
	c[end,:] = collect(grid)
	V[end,:] = sqrt.(c[end,:] )  

	for it in (nperiods-1):-1:1
		c[it,:] = policy_iter(grid, c[it+1,:], u_prime, β,  π, yvec)
		# now calulate implied value function
		v_fun = extrapolate(interpolate((collect(grid),), V[it+1,:], Gridded(Linear()) ) , Interpolations.Flat())
		V[it,:] = sqrt.(c[it,:]) + β * ((1-π) * v_fun(grid .- c[it,:] .+ yvec[1]) + (π) * v_fun(grid .- c[it,:] .+ yvec[2]))
	end
	return (V,c)
end

# ╔═╡ c339ad8a-87e7-11eb-2802-0991af7b2a78
let
	V,a = backwards_pol(Rspace,nperiods,β,π,yvec)
	cg = cgrad(:viridis)
    cols = cg[range(0.0,stop=1.0,length = nperiods)]
	pa = plot(Rspace, a[1,:], xlab = "R", ylab = "Action",label = L"a_1",leg = :topleft, color = cols[1])
	for it in 2:nperiods
		plot!(pa, Rspace, a[it,:], label = L"a_{%$(it)}", color = cols[it])
	end
	pv = plot(Rspace, V[1,:], xlab = "R", ylab = "Value",label = L"V_1",leg = :bottomright, color = cols[1])
	for it in 2:nperiods
		plot!(pv, Rspace, V[it,:], label = L"V_{%$(it)}", color = cols[it])
	end
	plot(pv,pa, layout = (1,2))
end

# ╔═╡ 96a55b70-9389-11eb-13a1-3146098e9cb9
md"

#

"

# ╔═╡ 7b137330-9307-11eb-2644-390cd9c1e569
function PFI()
    c_init = kgrid
    for iter in 1:N_iter
        c_next = policy_iter(kgrid,c_init,uprime,fprime)  
        # check convergence
        if maximum(abs,c_init.-c_next) < tol
            perrors =  maximum(abs,c_next.-c_star(kgrid))
            return (v = fill(NaN,n), p =c_next, errv = 0.0, errp = perrors, iter = iter)

        elseif iter==N_iter
            warn("No solution found after $iter iterations")
            return (v = fill(NaN,n), p =c_next, errv = 0.0, errp = perrors, iter = iter)
        end
        c_init = c_next  # update guess 
    end
end

# ╔═╡ 9d4211e6-9389-11eb-03a4-559460e28c44
md"
#
"

# ╔═╡ 3b50f766-9386-11eb-2101-97a38644fda8
plotVFI(PFI())

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Roots = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"

[compat]
Interpolations = "~0.13.4"
LaTeXStrings = "~1.3.0"
Optim = "~1.5.0"
Plots = "~1.23.6"
PlutoUI = "~0.7.19"
Roots = "~1.3.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0bc60e3006ad95b4bb7497698dd7c6d649b9bc06"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "e527b258413e0c6d4f66ade574744c94edef81f8"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.40"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f885e7e7c124f8c92650d61b9477b9ac2ee607dd"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.1"

[[ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "9a1d594397670492219635b35a3d830b04730d62"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.1"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "3287dacf67c3652d3fed09f4c12c187ae4dbb89a"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.4.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "8b3c09b56acaf3c0e581c66638b85c8650ee9dca"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.8.1"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "6406b5112809c08b1baa5703ad274e1dded0652f"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.23"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fd75fa3a2080109a2c0ec9864a6e14c60cca3866"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.62.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "35d435b512fbab1d1a29138b5229279925eba369"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.5.0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "0d185e8c33401084cab546a756b387b15f76720c"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.23.6"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "e071adf21e165ea0d904b595544a8e514c8bb42c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.19"

[[PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Roots]]
deps = ["CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "4c40dc61b51054bdb93536400420d73fdca6865e"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "1.3.7"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "def0718ddbabeb5476e51e5a43609bee889f285d"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.0"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "e7bc80dc93f50857a5d1e3c8121495852f407e6a"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.4.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "eb35dcc66558b2dda84079b9a1be17557d32091a"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.12"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─5fab5e80-87ce-11eb-111a-d5288227b97c
# ╟─705e96f2-87ce-11eb-3f5a-eb6cdb8c49d4
# ╟─34403942-87d2-11eb-1172-7bd285bf7d75
# ╟─b9ed265e-87d2-11eb-1028-05b4c4ccbc74
# ╟─07992abe-87d4-11eb-0fd7-1bef7fc2ada8
# ╟─93148b06-87d2-11eb-2356-fb460acf8e66
# ╟─3d97787a-87d2-11eb-3713-ab3dad6e5e8b
# ╟─a41aff3a-87d8-11eb-00fe-e16c14870ba2
# ╟─c6ed3824-87d9-11eb-3e0a-b5a473fdb952
# ╟─51983542-87f9-11eb-226a-81fe383f0763
# ╠═6ddfc91c-87da-11eb-3b5c-dfe006576f29
# ╟─aeb2b692-87db-11eb-3c9a-114acdc3d159
# ╠═b3f20522-87db-11eb-325d-7ba201c66452
# ╠═36610294-87dc-11eb-21e8-5b1dd06c6020
# ╟─6e8aec5e-87f9-11eb-2e53-bf5ba90fa750
# ╠═f00890ee-87db-11eb-153d-bb9244b46e89
# ╠═4da91e08-87dc-11eb-3238-433fbdda7df7
# ╟─7a97ba6a-87f9-11eb-1c88-a999bcff4647
# ╟─ca3e6b12-87dc-11eb-31c6-8d3be744def0
# ╟─027875e0-87e0-11eb-0ca5-317b58b0f8fc
# ╟─48cc5fa2-87dd-11eb-309b-9708941fd8d5
# ╠═04b6c4b8-87df-11eb-21dd-15a443042bd1
# ╟─99b19ed4-92c8-11eb-2daa-2df3db8736bd
# ╠═43539482-87e0-11eb-22c8-b9380ae1ebdb
# ╟─a361963a-92c8-11eb-0543-7decc61720a8
# ╟─5f764692-87df-11eb-3b9d-e5823eb4d1b3
# ╟─8eb731f6-87e0-11eb-35dd-61ba070afc8b
# ╟─02d26410-87e2-11eb-0278-99ee0bbd2923
# ╠═3b1f5120-87e2-11eb-1b10-85900b6fbeb6
# ╟─c6990bb0-92c8-11eb-2523-9f8a7ec1cd4a
# ╠═0b83e920-87e3-11eb-0792-479eb843b429
# ╟─d391139e-92c8-11eb-0888-b325a75fb10f
# ╟─7003b4e8-87e3-11eb-28a9-f7e3668beac3
# ╟─4504d988-87e4-11eb-05d5-c9f9f215f785
# ╟─6828b538-87e4-11eb-3cdd-71c31dad5a6e
# ╟─ba19759c-87eb-11eb-1bec-51de9ac10e31
# ╟─a4018f28-87ec-11eb-07f0-af86523dd26e
# ╠═027fda74-87f1-11eb-1441-55d6e410bf4c
# ╠═540d8814-87f1-11eb-0b8c-23357c46f93c
# ╟─d2788ffe-92c4-11eb-19e7-4b41d9f9ebdd
# ╠═c75eed74-87ee-11eb-3e9a-3b893294baec
# ╟─ae56f4e2-87f9-11eb-10fc-e3eda66e8a1f
# ╠═17ea7f6a-87e7-11eb-2d3e-b9a1e771b08e
# ╟─bcca454c-87f9-11eb-0670-0d85db6b6e37
# ╠═d18511ea-87e7-11eb-08ce-a9176a32fbd1
# ╟─c1e65340-87f9-11eb-3cd7-05f14edf71d2
# ╟─c339ad8a-87e7-11eb-2802-0991af7b2a78
# ╟─6fbaac56-87e0-11eb-3d09-39a4fd293e88
# ╟─a510ff40-9302-11eb-091e-6929367f6783
# ╟─06ff3932-9304-11eb-07cc-d798d56c0931
# ╠═2e121c60-9304-11eb-29c4-afd0a289343f
# ╟─3c8e560a-9304-11eb-2c45-7df67768f75b
# ╠═4849479a-9304-11eb-1c06-4b53a4163f85
# ╟─5984cd7a-9304-11eb-0bd4-ebac2da4f300
# ╠═5d642bca-9304-11eb-2b23-dfa47b04bb22
# ╟─67d367c4-9304-11eb-0bfc-21b9bfec2fb9
# ╠═6b33c6ca-9304-11eb-3766-fdcc42b64f2d
# ╟─72cc3cdc-9304-11eb-3548-95962c3513ec
# ╟─776c4712-9304-11eb-1824-a14cde26b895
# ╠═7999a272-9304-11eb-254c-af591bae0620
# ╟─a4ba5a96-ad59-4319-bd94-636f525853c6
# ╟─d52d3d31-e447-4120-92ae-3d4da7a02219
# ╠═c881819e-ca81-40b0-9e0f-93abcb43a2b4
# ╟─84e23d2a-9388-11eb-2483-3342d1683129
# ╠═7e0a9d82-9304-11eb-1d3d-bb47fc55d033
# ╟─aa891724-9388-11eb-32bb-8f8d15dc3761
# ╠═597080ba-9307-11eb-2780-87474bfc5cff
# ╟─6f9b2b8a-9389-11eb-2731-398e1342538a
# ╠═710a0212-9307-11eb-2cf3-e75b9c5eddab
# ╠═67c4bbb8-9307-11eb-3ecb-13d7af9d899a
# ╟─96a55b70-9389-11eb-13a1-3146098e9cb9
# ╠═7b137330-9307-11eb-2644-390cd9c1e569
# ╟─9d4211e6-9389-11eb-03a4-559460e28c44
# ╠═3b50f766-9386-11eb-2101-97a38644fda8
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
