### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 36610294-87dc-11eb-21e8-5b1dd06c6020
using PlutoUI

# ╔═╡ d1096078-87dc-11eb-2567-356e2376c9d7
using Plots, LaTeXStrings

# ╔═╡ de974196-87e2-11eb-2bd0-2be91745ee25
using Optim

# ╔═╡ 1b72f9ee-87e7-11eb-202d-47c87136deaf
using Roots	

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
md"
#

![](https://i.ytimg.com/vi/wAbnNZDhYrA/maxresdefault.jpg)"

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

# ╔═╡ f81bfd95-c526-4adc-9687-26519029f450
let
	c = rand(10)
	g = 1:10
	itp = extrapolate(interpolate((g,), c, Gridded(Linear())), Interpolations.Linear())
	itp(0.5)
end

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
md"
#

![](https://s3.getstickerpack.com/storage/uploads/sticker-pack/meme-pack-1/sticker_19.png?363e7ee56d4d8ad53813dae0907ef4c0&d=200x200)"

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

# ╔═╡ c881819e-ca81-40b0-9e0f-93abcb43a2b4
VFI_converge(bellman_operator, 50)

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

# ╔═╡ 10e1dde1-fd81-4943-92d9-5327749f9630
using Interpolations

# ╔═╡ 3f226986-87df-11eb-0bfc-953a37d5c3ff
using Interpolations

# ╔═╡ Cell order:
# ╠═5fab5e80-87ce-11eb-111a-d5288227b97c
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
# ╠═d1096078-87dc-11eb-2567-356e2376c9d7
# ╟─7a97ba6a-87f9-11eb-1c88-a999bcff4647
# ╟─ca3e6b12-87dc-11eb-31c6-8d3be744def0
# ╟─027875e0-87e0-11eb-0ca5-317b58b0f8fc
# ╟─48cc5fa2-87dd-11eb-309b-9708941fd8d5
# ╠═3f226986-87df-11eb-0bfc-953a37d5c3ff
# ╠═04b6c4b8-87df-11eb-21dd-15a443042bd1
# ╟─99b19ed4-92c8-11eb-2daa-2df3db8736bd
# ╠═43539482-87e0-11eb-22c8-b9380ae1ebdb
# ╟─a361963a-92c8-11eb-0543-7decc61720a8
# ╟─5f764692-87df-11eb-3b9d-e5823eb4d1b3
# ╟─8eb731f6-87e0-11eb-35dd-61ba070afc8b
# ╟─02d26410-87e2-11eb-0278-99ee0bbd2923
# ╠═de974196-87e2-11eb-2bd0-2be91745ee25
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
# ╠═10e1dde1-fd81-4943-92d9-5327749f9630
# ╠═f81bfd95-c526-4adc-9687-26519029f450
# ╠═c75eed74-87ee-11eb-3e9a-3b893294baec
# ╟─ae56f4e2-87f9-11eb-10fc-e3eda66e8a1f
# ╠═1b72f9ee-87e7-11eb-202d-47c87136deaf
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
# ╠═a4ba5a96-ad59-4319-bd94-636f525853c6
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
