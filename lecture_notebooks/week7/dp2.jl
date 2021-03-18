### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 36610294-87dc-11eb-21e8-5b1dd06c6020
using PlutoUI

# ╔═╡ d1096078-87dc-11eb-2567-356e2376c9d7
using Plots, LaTeXStrings

# ╔═╡ 3f226986-87df-11eb-0bfc-953a37d5c3ff
using Interpolations

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
        # find max of ojbective between [0,k^alpha]
        res = optimize(objective, 1e-6, r)  # search in [1e-6,r]
        pol[i] = res.minimizer
        v1[i] = -res.minimum
    end
    return (v1,pol)   # return both value and policy function
end


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
* In particular, $c_{t+1} \geq 0$.
* look back above at the last plot of the consumption function. It has a kink!
* The Euler Equation applies only to the *right* of that kink!
* For small values of $R_t$ (left of kink), we cannot consume along the straight line of the prescribed consumption function, since then $R_{t+1} = R_t - c_t < 0$!
"

# ╔═╡ a4018f28-87ec-11eb-07f0-af86523dd26e
md"
#

* So if we want to use the Euler Equation, we need to manually adjust for that.
* 😩
* Alright, let's do it. It's not too bad in the end.

#

"

# ╔═╡ 027fda74-87f1-11eb-1441-55d6e410bf4c
u_prime(c) = 0.5 .* c.^(-0.5)

# ╔═╡ 540d8814-87f1-11eb-0b8c-23357c46f93c
u_primeinv(u) = (2 .* u).^(-2)

# ╔═╡ c75eed74-87ee-11eb-3e9a-3b893294baec
function EEresidual(c,R,C1,grid,yvec,β)
	# next period resources, given c0
	R1 = R - c .+ yvec  # a (2,1) vector: R1 for each income level

	# get implied next period consumption from C1
	citp = extrapolate(interpolate(([0.0, grid...],), [0.0,C1...], Gridded(Linear())), Interpolations.Flat())
	c1 = citp.(R1)
	RHS = β * [1-π  π] * u_prime(c1) # expecte marginal utility of tomorrows consumption
	
	# euler residual
	# taking u_prime inverse gets predicted current consumption
	# that should be equal to our choice c.
	r = c .- u_primeinv(RHS)
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
		res = EEresidual(r,r,c0, grid, yvec,β)
		if res < 0
			# we consume too little today, c_t is too small. 
			# could only make it bigger by borrowing next period.
			# we cant. so really we are consuming all we have:
			c1[i] = r
		else
			c1[i] = fzero( x-> EEresidual(x,r,c0, grid, yvec, β) , 1e-6, r)
		end
    end
    return c1
end

# ╔═╡ bcca454c-87f9-11eb-0670-0d85db6b6e37
md"
#
"

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

# ╔═╡ c1e65340-87f9-11eb-3cd7-05f14edf71d2
md"
#
"

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

# ╔═╡ 6fbaac56-87e0-11eb-3d09-39a4fd293e88
md"
#

![](https://s3.getstickerpack.com/storage/uploads/sticker-pack/meme-pack-1/sticker_19.png?363e7ee56d4d8ad53813dae0907ef4c0&d=200x200)"

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
# ╠═d1096078-87dc-11eb-2567-356e2376c9d7
# ╟─7a97ba6a-87f9-11eb-1c88-a999bcff4647
# ╟─ca3e6b12-87dc-11eb-31c6-8d3be744def0
# ╟─027875e0-87e0-11eb-0ca5-317b58b0f8fc
# ╟─48cc5fa2-87dd-11eb-309b-9708941fd8d5
# ╠═3f226986-87df-11eb-0bfc-953a37d5c3ff
# ╠═04b6c4b8-87df-11eb-21dd-15a443042bd1
# ╠═43539482-87e0-11eb-22c8-b9380ae1ebdb
# ╠═5f764692-87df-11eb-3b9d-e5823eb4d1b3
# ╟─8eb731f6-87e0-11eb-35dd-61ba070afc8b
# ╟─02d26410-87e2-11eb-0278-99ee0bbd2923
# ╠═de974196-87e2-11eb-2bd0-2be91745ee25
# ╠═3b1f5120-87e2-11eb-1b10-85900b6fbeb6
# ╠═0b83e920-87e3-11eb-0792-479eb843b429
# ╠═7003b4e8-87e3-11eb-28a9-f7e3668beac3
# ╟─4504d988-87e4-11eb-05d5-c9f9f215f785
# ╟─6828b538-87e4-11eb-3cdd-71c31dad5a6e
# ╟─ba19759c-87eb-11eb-1bec-51de9ac10e31
# ╟─a4018f28-87ec-11eb-07f0-af86523dd26e
# ╠═027fda74-87f1-11eb-1441-55d6e410bf4c
# ╠═540d8814-87f1-11eb-0b8c-23357c46f93c
# ╠═c75eed74-87ee-11eb-3e9a-3b893294baec
# ╟─ae56f4e2-87f9-11eb-10fc-e3eda66e8a1f
# ╠═1b72f9ee-87e7-11eb-202d-47c87136deaf
# ╠═17ea7f6a-87e7-11eb-2d3e-b9a1e771b08e
# ╟─bcca454c-87f9-11eb-0670-0d85db6b6e37
# ╠═d18511ea-87e7-11eb-08ce-a9176a32fbd1
# ╟─c1e65340-87f9-11eb-3cd7-05f14edf71d2
# ╟─c339ad8a-87e7-11eb-2802-0991af7b2a78
# ╟─6fbaac56-87e0-11eb-3d09-39a4fd293e88
