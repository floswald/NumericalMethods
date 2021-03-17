### A Pluto.jl notebook ###
# v0.12.21

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

# ╔═╡ 8a796500-8722-11eb-27c9-dd7f6d6d18fb
begin
	using Plots
	using LaTeXStrings
	using PlutoUI
end

# ╔═╡ 199fe28a-870e-11eb-3e76-a555f3cca82d
html"<button onclick='present()'>present</button>"

# ╔═╡ 03008ac2-870e-11eb-0407-758be6a15570
md"
# Dynamic Programming

Florian Oswald, SciencesPo 2021
"

# ╔═╡ 413b8300-870e-11eb-3348-f5db37cc4908
md"
## Introduction

* Dynamic Programming is a powerful technique for dynamic problems, within and outside of economics.
* Some Examples:
	1. How to allocate spending over time, given a certain budget?
	2. How to position truck drivers over the USA, if demand is random?
	3. How to plan mid-air refuelling of military aircraft, if future needs are uncertain?
	4. What's the best route for a travelling salesman to visit a number of cities, given a certain road network?"

# ╔═╡ bc7c2128-870e-11eb-0807-bd44c9a21425
md"
#

* Here we will focus on deterministic problems, i.e. no uncertainty.
* Let's start with a simple example that probably you have heard of already: Cake Eating. 🍰
* Then we will introduce some notation and terminology.
* Finally, we will look at some algorithms you might use to solve this kind of problem."

# ╔═╡ 584188c8-870f-11eb-1130-a327b00705f2
md"
# Deterministic Dynamic Programming: Piece of 🍰

* Assume you dispose of a given budget $R$ of resources to allocate to spending at $t$ dates. You can think of $R$ Euros, for example.
* Let $a_t$ be the *action* you take at date $t$. Here: How much to spend at date $t$.
* Let $C_t(a_t)$ the *reward*, the *consumption*, or the *utility* you obtain from that action.
* What you see below is typically described as _action = consumption_, hence a _consumption-savings problem_, but let's keep it general and use _action_ instead.
"

# ╔═╡ 6cb6e2f2-8710-11eb-0452-470650c6beb0
md"
## A *Budgeting Problem*

* Our aim is to *maximize* our total utility:
$$\max_a \sum_{t\in\mathcal{T}} C_t(a_t)$$

* But we cannot spend more than what we have in terms of resources, so this is subject to constraint

$$\sum_{t\in\mathcal{T}} = R$$

* and we don't allow borrowing: $a_t \geq 0$
"

# ╔═╡ ce122584-8710-11eb-04e7-fb89bd1fdf65
md"
#

* We are looking for the *best* way to decide on actions $a_t$.
* So, we want a list of values $a_t$ of what to do in each period $t$ - given a certain $R$!
* We want to know this function:
$$V_t(R_t) = \text{value of having }R_t\text{ resources left to allocate to dates }t,t+1,...$$
"

# ╔═╡ 4d554b78-8711-11eb-10b7-11c968ce6a04
md"
## States and Transitions

* The value of resources evolves as 
$$R_{t+1} = R_t - a_t$$ which is called the *transition function*

* We call $R_t$ the *state variable*. It encodes all the relevant information to model the system *forward, starting in $t$*.

* A more general way of writing the above would be 
$$R_{t+1} = g(R_t,a_t) = R_t - a_t$$
"

# ╔═╡ 10c41fee-8712-11eb-0048-d5caa9c92210
md"
## Optimality Condition and Bellman Equation

* The relationship between $V_t(R_t)$ and $V_{t+1}(R_{t+1})$ is then
$$V_t(R_t) = \max_{0 \leq a_t \leq R_t} \left[ C_t(a_t) + V_{t+1}(g(R_t,a_t)) \right]$$

* This says: 
> The value of having $R_t$ resources left in period $t$ is the value of optimizing current spending (the $\max C_t(a_t)$ part) plus the value of then having $g(R_t,a_t)$ units left going forward

* This equation is commonly known as the _Bellman Equation_.
* Great. Well given $R_t$ we could just try out different $a_t$ and see which gives the highest value. But...what on earth is $V_{t+1}(\cdot)$? 🤔
"

# ╔═╡ 1d4e0706-8713-11eb-33d8-b1e6568595af
md"
## Finite Time Backward Induction

* One simple way is to say that time is finite, stops at $T$, and the value of leaving resources is zero: $V_{T+1}(R) = 0$.
* That makes for the last period
$$V_T(R_T) = \max_{0 \leq a_t \leq R_t} C_T(a_T)$$ 
for *any* value $R_T$. That's a *function* right there.
* But what is the value of that function? What is 
$$\max_{0 \leq a_t \leq R_t} C_T(a_T)?$$ 
 
#

* Ok, well then the preceding period is easy. For whatever $R_{T-1}$ we enter the period, and for whatever is left after action $a_{T-1}$, we know *the future*:
$$V_{T-1}(R_{T-1}) = \max_{0 \leq a_{T-1} \leq R_{T-1}} \left[ C_{T-1}(a_{T-1}) + V_{T}(g(R_{T-1},a_{T-1})) \right]$$

* Of course this will work all the way to period 1. If $a_t$ is discrete, we do not need *any* assumptions on $C$ other than finitness - what we are proposing here will be optimal.
"

# ╔═╡ 0372f860-8712-11eb-20d3-43c71bf6f024
md"
## Discrete Action Space, Finite Time

* Let's try that out:
* we have $C_t(a_t) = \sqrt{a_t}$
* and $V_{T+1}(R) = 0$
* so its optimal to consume whatever is left. There is not even any maximization problem to solve.
"

# ╔═╡ cefec74c-8722-11eb-22d9-7551eeb4cc96
md"
#
"

# ╔═╡ d4854ff4-8722-11eb-1ecf-89fd39b87db9
md"
#

* ok, now we have $V_T$. 
* Let's go back one period now.
"

# ╔═╡ 636d55f6-8725-11eb-2f78-691fc074cb77
md"
#
"

# ╔═╡ 8822b3a8-8725-11eb-149e-25a52208834d
md"
#

* and here are the corresponding optimal actions for both periods:
"

# ╔═╡ 6b42dfd0-8725-11eb-1f90-eb215d9a4daf
md"
#

* great. now let's go all the way to period 1!
"

# ╔═╡ 768af538-8725-11eb-2202-4f8bcf65587d
# period T-1 till 1
# each period is the same now!
# that calls for a function!
"""
	Vperiod(grid::Vector,vplus::Vector)

Given a grid and a next period value function `vplus`,
calculate current period optimal value and actions.
"""
function Vperiod(grid::Vector,vplus::Vector)
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
				r1 = r - achoice  # tomorrow's R
				jr = argmin(abs.(grid .- r1))  # index of that value in Rspace
				w[ia] = sqrt(achoice) + vplus[jr]   # value of that achoice
			end
		end
		# find best action
		Vt[ir], ix = findmax(w) # stores Value und policy (index of optimal choice)
		at[ir] = grid[ix]  # record optimal action level
	end
	return (Vt, at)
end

# ╔═╡ fc829384-8726-11eb-358e-5dffd3f96f6f
md"
#

* Let's just call `Vperiod` for all periods $T-2,\dots,1$, going backwards.
* By the way, the function `Vperiod` is also called the *Bellman Opterator*: It performs the operation on the RHS of the Bellman Equation!"

# ╔═╡ 24bc7018-8727-11eb-315d-ad2f90883d5a
function backwards(grid, nperiods)
	points = length(grid)
	V = zeros(nperiods,points)
	a = zeros(nperiods,points)
	V[end,:] = sqrt.(grid)  # from before: final period
	a[end,:] = collect(grid)

	for it in (nperiods-1):-1:1
		x = Vperiod(grid, V[it+1,:])	
		V[it,:] = x[1]
		a[it,:] = x[2]
	end
	return (V,a)
end

# ╔═╡ 18a59a1a-8728-11eb-3d1e-b3a00283280c
md"
#
"

# ╔═╡ dd0257b2-8724-11eb-3e68-8791b59f85f4
@bind highR Slider(2:200)

# ╔═╡ ac31d660-8721-11eb-226b-49124f9093de
begin
	# final period T
	points = 500
	lowR = 0.0001
	# highR = 10.0 # slider below
	# more points towards zero to make nicer plot
	Rspace = exp.(range(log(lowR), stop = log(highR), length = points))
	aT = Rspace # consume whatever is left
	VT = sqrt.(aT)  # utility of that consumption
end

# ╔═╡ 9467efca-8722-11eb-12ff-7f5446bcf618
function plotVT()
	plot(Rspace, VT, xlab = "R", ylab = "Value",label = L"V_T", m = (:circle),
	     leg = :bottomright)
end

# ╔═╡ f495704e-8729-11eb-3865-cfe53d7ecb4e
plotVT()

# ╔═╡ 1c5a5cc6-8760-11eb-1c63-030f77621098
begin
	# period T-1
	# now for each value in Rspace, we need to decide how much to consume
	w = zeros(points) # temporary vector for each choice or R'
	VT_1 = zeros(points) # optimal value in T-1 at each state of R
	ix = 0 # optimal index of action in T-1 at each state of R
	aT_1 = zeros(points) # optimal action in T-1 at each state of R

	for (ir,r) in enumerate(Rspace) # for all possible R-values
        # loop over all possible action choices
        for (ia,achoice) in enumerate(Rspace)
            if r <= achoice   # check whether that choice is feasible
                w[ia] = -Inf
            else
				r1 = r - achoice  # tomorrow's R
				jr = argmin(abs.(Rspace .- r1))  # index of that value in Rspace
                w[ia] = sqrt(achoice) + VT[jr]   # value of that achoice
            end
        end
        # find best action
        VT_1[ir], ix = findmax(w) # stores Value und policy (index of optimal choice)
		aT_1[ir] = Rspace[ix]  # record optimal action level
		
    end
end

# ╔═╡ df006604-8723-11eb-0d67-17b8090add25
let
	pv = plotVT()
	plot!(pv, Rspace, VT_1, label = L"V_{T-1}", m = (:star))
end

# ╔═╡ 96f139f4-8725-11eb-3e96-596719146475
plotaT() = plot(Rspace, aT ,label = L"a_T",leg = :topleft,ylab = "action",xlab = "R")

# ╔═╡ 36c341e8-8726-11eb-2bd5-cf97efd66410
let
	pa = plotaT()
	plot!(pa, Rspace, aT_1, label = L"a_{T-1}")
end

# ╔═╡ cc9e9b2c-8762-11eb-10a8-45e5052c0ccd
@bind nperiods Slider(2:20)

# ╔═╡ 04d66cda-8728-11eb-13bb-6b97b660bb42
V,a = backwards(Rspace, nperiods);

# ╔═╡ 301ee980-8728-11eb-200b-2dfe31f59818
let
	cg = cgrad(:viridis)
    cols = cg[range(0.0,stop=1.0,length = nperiods)]
	pv = plot(Rspace, V[1,:], xlab = "R", ylab = "Value",label = L"V_1",leg = :bottomright, color = cols[1])
	for it in 2:nperiods
		plot!(pv, Rspace, V[it,:], label = L"V_{%$(it)}", color = cols[it])
	end
	pv
end

# ╔═╡ 521cc474-872a-11eb-16c6-cb896e828195
md"
#

* same for optimal actions:
"

# ╔═╡ 5b9f63d0-872a-11eb-292c-5d0f3e2756da
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

# ╔═╡ 60cca6fe-8763-11eb-2845-db08468ce26a
md"
# Interpretation

* Does that solution make *sense*?
* What does the optimal policy prescribe to do, at a certain level of $R$, in each period?
"

# ╔═╡ 98270f9a-8763-11eb-0850-578a84382a35
bar(1:nperiods,a[:, end], leg = false, title = "Given R_t = $(Rspace[end]), take action...",xlab = "period")

# ╔═╡ 89680362-8761-11eb-3dd6-4d217063c82a
md"
# An Equivalent Formulation

* Let's slightly change the problem: 
$$V_{T-1}(R_{T-1}) = \max_{R' <R_{T-1}} \left[ C_{T-1}(R_{T-1} - R') + V_{T}(R') \right]$$
* This works because $a_t = R_{T-1} - R'$, so we choose tomorrow's state.
* This yields an almost identical solution. Not exactly because we have to force our solution onto the `Rspace` grid.
* In fact, *this* formulation works slightly better here.
"

# ╔═╡ 08a16622-8768-11eb-2121-9daf769e6db8
md"
#
"

# ╔═╡ a12b1d9e-8767-11eb-1609-111066473f61
"""
	Vperiod2(grid::Vector,vplus::Vector)

Given a grid and a next period value function `vplus`,
calculate current period optimal value and actions.
"""
function Vperiod2(grid::Vector,vplus::Vector)
	points = length(grid)
	w = zeros(points) # temporary vector for each choice or R'
	Vt = zeros(points) # optimal value in T-1 at each state of R
	ix = 0 # optimal action index in T-1 at each state of R
	at = zeros(points) # optimal action in T-1 at each state of R

	for (ir,r) in enumerate(grid) # for all possible R-values
		# loop over all possible action choices
		for (irprime,rprime) in enumerate(grid)
			if r - rprime < 0   # check whether that choice is feasible
				w[irprime] = -Inf
			else
				w[irprime] = sqrt(r - rprime) + vplus[irprime]
			end
		end
		# find best action
		Vt[ir], ix = findmax(w) # stores Value und policy (index of optimal choice)
		at[ir] = r - grid[ix] # convert back to action
	end
	return (Vt, at)
end

# ╔═╡ 146b36f4-8768-11eb-14f4-57d5a0f265a5
function backwards2(grid, nperiods)
	points = length(grid)
	V = zeros(nperiods,points)
	a = zeros(nperiods,points)
	V[end,:] = sqrt.(grid)  # from before: final period
	a[end,:] = collect(grid)

	for it in (nperiods-1):-1:1
		x = Vperiod2(grid, V[it+1,:])	
		V[it,:] = x[1]
		a[it,:] = x[2]
	end
	return (V,a)
end

# ╔═╡ 43b4a500-8768-11eb-20a0-432be2b22377
md"
#
"

# ╔═╡ 22f93d4c-8768-11eb-26ba-8fa76ced9fdd
let
	V,a = backwards2(Rspace, nperiods);
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

# ╔═╡ 47b0ecfc-8768-11eb-0e6e-e9c28ded3bf0
md"
#

* let's see the policy function again:
"

# ╔═╡ 50b7ab74-8768-11eb-3fe9-4bab892f73f9
let
	V2,a2 = backwards2(Rspace, nperiods);
	V,a = backwards(Rspace, nperiods);
	p1 = bar(1:nperiods,a[:, end], leg = false, title = "approx V_{t+1}",xlab = "period")
	p2 = bar(1:nperiods,a2[:, end], leg = false, title = "V_{t+1} on grid",xlab = "period")
	plot(p1,p2)
end

# ╔═╡ Cell order:
# ╠═199fe28a-870e-11eb-3e76-a555f3cca82d
# ╟─03008ac2-870e-11eb-0407-758be6a15570
# ╟─413b8300-870e-11eb-3348-f5db37cc4908
# ╟─bc7c2128-870e-11eb-0807-bd44c9a21425
# ╟─584188c8-870f-11eb-1130-a327b00705f2
# ╟─6cb6e2f2-8710-11eb-0452-470650c6beb0
# ╟─ce122584-8710-11eb-04e7-fb89bd1fdf65
# ╟─4d554b78-8711-11eb-10b7-11c968ce6a04
# ╠═10c41fee-8712-11eb-0048-d5caa9c92210
# ╠═1d4e0706-8713-11eb-33d8-b1e6568595af
# ╠═0372f860-8712-11eb-20d3-43c71bf6f024
# ╠═cefec74c-8722-11eb-22d9-7551eeb4cc96
# ╠═ac31d660-8721-11eb-226b-49124f9093de
# ╠═8a796500-8722-11eb-27c9-dd7f6d6d18fb
# ╠═9467efca-8722-11eb-12ff-7f5446bcf618
# ╠═f495704e-8729-11eb-3865-cfe53d7ecb4e
# ╟─d4854ff4-8722-11eb-1ecf-89fd39b87db9
# ╠═1c5a5cc6-8760-11eb-1c63-030f77621098
# ╠═636d55f6-8725-11eb-2f78-691fc074cb77
# ╠═df006604-8723-11eb-0d67-17b8090add25
# ╟─8822b3a8-8725-11eb-149e-25a52208834d
# ╠═96f139f4-8725-11eb-3e96-596719146475
# ╠═36c341e8-8726-11eb-2bd5-cf97efd66410
# ╠═6b42dfd0-8725-11eb-1f90-eb215d9a4daf
# ╠═768af538-8725-11eb-2202-4f8bcf65587d
# ╟─fc829384-8726-11eb-358e-5dffd3f96f6f
# ╠═24bc7018-8727-11eb-315d-ad2f90883d5a
# ╠═18a59a1a-8728-11eb-3d1e-b3a00283280c
# ╠═dd0257b2-8724-11eb-3e68-8791b59f85f4
# ╠═cc9e9b2c-8762-11eb-10a8-45e5052c0ccd
# ╠═04d66cda-8728-11eb-13bb-6b97b660bb42
# ╠═301ee980-8728-11eb-200b-2dfe31f59818
# ╟─521cc474-872a-11eb-16c6-cb896e828195
# ╠═5b9f63d0-872a-11eb-292c-5d0f3e2756da
# ╠═60cca6fe-8763-11eb-2845-db08468ce26a
# ╠═98270f9a-8763-11eb-0850-578a84382a35
# ╠═89680362-8761-11eb-3dd6-4d217063c82a
# ╠═08a16622-8768-11eb-2121-9daf769e6db8
# ╠═a12b1d9e-8767-11eb-1609-111066473f61
# ╠═146b36f4-8768-11eb-14f4-57d5a0f265a5
# ╠═43b4a500-8768-11eb-20a0-432be2b22377
# ╠═22f93d4c-8768-11eb-26ba-8fa76ced9fdd
# ╟─47b0ecfc-8768-11eb-0e6e-e9c28ded3bf0
# ╠═50b7ab74-8768-11eb-3fe9-4bab892f73f9
