### A Pluto.jl notebook ###
# v0.19.38

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

# ╔═╡ d257d214-0e18-444d-aad9-5f37ec978cf5
begin
	using Plots
	using Interpolations
	using FastGaussQuadrature
end

# ╔═╡ d40c12c8-b0ac-4c2c-ad5c-86c918aad387
using PlutoUI

# ╔═╡ 2cc56380-3907-4be6-b699-288093fccb50
begin
struct TwoColumn{L, R}
    left::L
    right::R
end

function Base.show(io, mime::MIME"text/html", tc::TwoColumn)
    write(io, """<div style="display: flex;"><div style="flex: 50%;">""")
    show(io, mime, tc.left)
    write(io, """</div><div style="flex: 50%;">""")
    show(io, mime, tc.right)
    write(io, """</div></div>""")
end
end

# ╔═╡ b581ffce-9d63-4fbf-992b-8f9280d83a92
html"<button onclick='present()'>present</button>"

# ╔═╡ 5285eb0a-a266-11eb-277e-a7c01cf568df
md"
# Endogenous Grid Method after Chris Caroll (EconLetters 2006)

**With and Without Discrete Choices**

*ScPo Numerical Methods 2024, Florian Oswald*

"

# ╔═╡ 32579cd1-07c0-43f5-bde6-5a6a9f52a999
md"

#

* We have encountered 2 methods to solve the consumption savings problem in class:
  1. VFI (value function iteration)
  2. PFI (policy function iteration)

* We saw that PFI works better in our applications. 
* Still, we needed to numerically solve for the root of a nonlinear equation (the Euler Euqation), and that is expensive.
* Let's introduce EGM now by way of example. EGM avoids *all* numerical root finding operations!
* At the end of the lecture we will introduce *discrete* choices into the consumption-savings world and learn about the DC-EGM extension to Carroll's method
"

# ╔═╡ 599b4b2e-dcfc-4790-a0bd-f9a60e9fadd1
md"

## Consumption-Savings Model (after Deaton)

$$V(M)=\max_{0 \le c \le M}\big\{u(c)+\beta \mathbb{E}_{y} V\big(\underset{=M'}{\underbrace{R(M-c)+y}}\big)\big\}$$

Here, we introduced a new state variable $M$ which denotes *cash-on-hand*, or, *all consumable resources*. Typically, we would lump together income and liquid assets in $M$ (like how much is on your bank account and how big is your income together determines the size of $M$)

- discrete time, infinite horizon  
- one continuous choice of consumption $0 \le c \le M$  
- state space: consumable resources in the beginning of the period $ M $, discretized  
- Cash on hand evolves as $M' = R(M-c)+y$
- income $y$ is log-normal distribution with $\mu = 0$ and $\sigma > 0$  
"

# ╔═╡ 0171a47c-f777-410d-9814-8c86da85bf3e
md"
## Euler Equation Solver for Deaton

* The FOC for this model is 
$$u'(c^\star) - \beta R \mathbb{E}_{y} V'\big(R(M-c^\star)+y\big) = 0$$
where $c^\star$ denotes *optimal consumption*

* Next, we remember the Envelope Condition, differentiating the value function by it's state variable $M$:

## Envelope Theorem in Deaton Model

Let us revisit the *envelope theorem* for this model class. First, define an implicit function $F$ of state $M$ and choice $c$ as follows:

$$F(M,c)=u(c)+\beta \mathbb{E}_{y} V\big(\underset{=M'}{\underbrace{R(M-c)+y}}\big)$$

so that the policy function $c^\star(M)$ satisfies $V(M)=F(M,c^\star(M))$. Next, we want to compute the derivative of the 2-argument function $F$ wrt its argument $M$ in each slot. We need the *chain rule*!

$$\begin{align}
\frac{d V(M)}{dM} &= \frac{d F(M,c^\star(M))}{dM}\\
                  &= \tfrac{\partial F(M,c^\star)}{\partial M} + \underset{=0\text{ by FOC}}{\underbrace{\tfrac{\partial F(M,c^\star)}{\partial c^\star}}} \tfrac{\partial c^\star(M)}{\partial M}
= \tfrac{\partial F(M,c^\star)}{\partial M} = \beta R \mathbb{E}_{y} V'\big(R(M-c^\star)+y\big)
\end{align}$$

In short, the envelope condition states:

$$V'(M) = \beta R \mathbb{E}_{y} V'\big(R(M-c^\star)+y\big)$$

which means that we have 

$$u'(c^\star) = V'(M)$$

in each period.

## Building up the Euler Equation

1. FOC:

$$u'(c^\star) - \beta R \mathbb{E}_{y} V'\big(R(M-c^\star)+y\big) = 0$$

2. Envelope: 

$$V'(M) = \beta R \mathbb{E}_{y} V'\big(R(M-c^\star)+y\big), \forall t$$

3. Combine

$$u'\big(c^\star(M)\big) = \beta R \mathbb{E}_{y} u'\big(c^\star\big(\underset{=M'}{\underbrace{R[M-c^\star(M)]+y}}\big)\big)$$

"

# ╔═╡ bf784923-f460-4816-87d6-5890f959e015
md"

## A New Variable: End-of-period Assets $A$

* Let's introduce a new variable: the *post-decision state variable*. Here: how many assets left at end of period *after* consumption took place?
* How much do you *save* today (and before you earned gross interest $R$ on it, at the start of next period)?
* Let us denote this $A$. It is useful to think throught the timing of the model:

$$M \rightarrow c(M) \rightarrow A = M-c(M) \rightarrow M' = R(M-c(M)) + y = RA + y$$

* The constraint is that we cannot consume more than we have consumable resources, which bounds $A \in [0, M]$

$$0 \le c \le M \; \Rightarrow \; 0 \le A = M-c \le M$$
"

# ╔═╡ 2dd54158-d62f-4b12-9c18-486e0a52596a
md"
## Euler Equation in terms of $A$

* Let's replace $M - c(M) = A$ in the Euler Euqation.

$$u'\big(c(M)\big) = \beta R \mathbb{E}_{y} u'\big(c(RA+\tilde{y})\big)$$

* Any optimal policy function $c^\star$ will satisfy this equation for $A = M- c(M)$ (in which case it is *just* our Euler Equation from before).
* So, *again* we will want to find the function $c^\star$, as in PFI. How?
* Given an current candidate function $c$, the next update $c^\dagger$ will be derived by taking the inverse of the marginal utility function on both sides of the above equation:

$$\begin{cases}
c^\dagger = (u')^{-1} \Big( \beta R \mathbb{E}_{y} u'(c\big(RA+y)\big) \Big) \\
M^\dagger = A + c^\dagger
\end{cases}$$

* A key insight is that the new set of points in $M^\dagger$ will be *endogenously determined*, hence the name of the method: Given $A$, the structure of the model (embedded in the euler equation) will imply optimal consumption *and* a set of corresponding grid points for cash on hand.
"


# ╔═╡ 51b82e35-fba3-4100-b313-254ece33449d
md"
## Standard PFI vs EGM PFI

"

# ╔═╡ eb1b201c-1598-4740-b7e9-5108f017f4e3
TwoColumn(
md"
### Standard
* We search for function $c^{\dagger}(M)$ that solves
$$u'\big(c^{\dagger}(M)\big) = \beta R \mathbb{E}_{y} u'\big(c[R(M-c^{\dagger}(M))+y]\big)$$
	
1. We fix a grid over $M$, say, $\mathcal{M} = \{m_1, m_2,\dots,m_n\}$
2. For each $m_i \in \mathcal{M}$, solve the above equation to obtain $c^{\dagger}(m_i)$
	
👉 must numerically solve a nonlinear equation with a root finder $n$ times.
",
	
md"
### EGM
	
1. Fix a grid over $A$.
2. Given current guess of policy function $c(M)$, (and given current *grid* $M$!), **directly** compute the next iteration of the policy function (and the **new** grid $M^\dagger$) as follows. 
3. Iterating over all points $A_j \in A$,
	
$$\begin{cases}
c_j = (u')^{-1} \Big( \beta R \mathbb{E}_{y} u'(c\big(RA_j+y)\big) \Big) \\
M_j = A_j + c_j
\end{cases}$$

👉 The next policy function $c^\dagger$ is the interpolation of points, i.e. $c^\dagger(M^\dagger) \equiv \{(M_i, c_i)\}_{i=1}^n$
	
👉 no numerical root solving or other optimization needed!
	
	
")

# ╔═╡ 2333dde1-68cc-4155-82e7-d8727ef596f0
md"
## The EGM Step

Let's put down that algorithm again:

* Given a grid point $A_j \in A$,

  1. Compute all tomorrow's potential cash on hands, filling in integration nodes for $y_k$: $M' = RA_j + y_k$
  2. Get optimal consumption at all those potential cash on hand levels, using the current guess of $c$.
  3. Compute marginal utility of each consumption and hence complete the RHS of the Euler equation
  4. Take inverse of marginal utility function to recover current period $c_j$ corresponding to end of period $A_j$
  5. Finish EGM step by accounting identity $M_j = A_j + c_j$
"

# ╔═╡ 590b81d4-26bc-46c8-82ed-2998fcffe706
md"

## Simple EGM Implementation

1. Log utility
1. fixed income (no integration): $y = \bar{y}$

$$V(M)=\max_{0 \le c \le M}\big\{log(c)+\beta V\big(R(M-c)+y\big)\big\}$$

$$u'\big(c^\star(M)\big) = \beta R u'\big(c^\star\big(R[M-c^\star(M)]+y\big)\big)$$

EGM-step:

$$\begin{cases}
c^\dagger = (u')^{-1} \Big( \beta R u'(c\big(RA+y)\big) \Big) \\
M^\dagger = A + c^\dagger
\end{cases}$$
"

# ╔═╡ e6376b86-56ee-4ace-9e9a-5ceb87bed7db
begin
	n = 5  # num points
	M = 10 # max cash on hand
	y = 0  # income: zero
	R = 1.05 # gross interest
	β = 0.95
	T = 5  # let's do 5 iterations
	
	agrid = range(0.0,stop = M,length = n)
	
	c = Vector{Float64}[Float64[] for i in 1:T]   # consumption function on m
	m = Vector{Float64}[Float64[] for i in 1:T]   # endo grids

	# start from known last period
	# (or view as some arbitrary starting value)
    m[T]       = [0.0,M]  
    c[T]       = [0.0,M]
	
	u(c) = log(c)  # utility
	mu(c) = 1/c    # marginal utility
	imu(u) = 1/u   # inverse of marginal utility
	
end

# ╔═╡ 8292ee9b-e13a-4921-888f-4f898cebc6c4
function det_egm_single(aj,policy,y)
	mplus = max(R * aj + y, 1e-10)  # all levels of next cash on hand
	cplus = policy(mplus)
	c = imu( β * R * mu( cplus ) )  # RHS of Euler Equation
	m = aj + c
	(m,c)
end

# ╔═╡ 0eb5052b-7d30-48d4-926e-15a43c3ee7de
md"
## First Iteration
"

# ╔═╡ 48b0b0dd-9487-4fbd-8ec7-bf61272d4ba6
function makeanim()
		# current policy
	polT = extrapolate(interpolate((m[T],), c[T], Gridded(Linear())), Linear()) 
	pltT = plot(m[T], c[T],title = "First Iteration: T-1", xlim = (-1,22), ylim = (-1,20),color = :black, marker = :circle, label = "period T", xlab = "M", ylab = "c(M)", ratio = 1)
	anim = @animate for j in eachindex(agrid)
		mj,cj = det_egm_single(agrid[j],polT,y)
		scatter!(pltT,[mj],[cj], label = "A$j = $(round(agrid[j],digits=1))")
	end 
	anim
end

# ╔═╡ 3317d300-0e2d-481b-ab0c-7c1fcd319162
TwoColumn(
	md"""
For each $i = 1,\dots,5$, let's take end of period assets $A_i$ and feed it through the EGM step:

	
$$\begin{cases}
c_i^\dagger = (u')^{-1} \Big( \beta R u'(c\big(R \cdot A_i+y)\big) \Big) \\
M_i^\dagger = A_i + c_i^\dagger
\end{cases}$$

Notice that $M^\dagger$ is *not* on the $A$-grid!
	""",
	gif(makeanim(), fps = 0.7)
)

# ╔═╡ 5ddaef03-bbdf-4374-82e7-fc9eb3d8fd12
function det_EGM(m0,c0,n,y,agrid)
	policy = extrapolate(interpolate((m0,), c0, Gridded(Linear())), Linear())
	m1 = zeros(n+1)  # n+1 because add 0 in first slot for borrowing constraint
	c1 = zeros(n+1)
	for (aj,a) in enumerate(agrid)
		m1[aj+1], c1[aj+1] = det_egm_single(a,policy,y)
	end
	m1,c1
end

# ╔═╡ fa84b929-6088-4bba-bb29-d0cc4d8f9b9d
md"
#
"

# ╔═╡ 034ff9fe-b925-4f6c-b9f8-e28a86e7ef9d
m[T-1], c[T-1] = det_EGM(m[T],c[T],n,y,agrid)

# ╔═╡ c12a930f-cf42-46e8-9f88-06d00aaba002
p = plot(m[T], c[T],title = "EGM Iterations", xlim = (-1,22), ylim = (-1,20),color = :black, marker = :circle, label = "period T", xlab = "M", ylab = "c(M)")

# ╔═╡ aac8dfa3-97b1-4786-baea-8f85524a27ac
md"
#
"

# ╔═╡ 6616459c-4542-4e5d-9f1f-926ebff695d2
plot!(p, m[T-1], c[T-1], color = :green, label = "T-1", marker = :circle)

# ╔═╡ 3f2351e8-aecc-420f-ac3c-f5322021a359
md"
#
"

# ╔═╡ 4aba99ee-455d-44b8-8630-be8a391ac291
m[T-2], c[T-2] = det_EGM(m[T-1],c[T-1],n,y,agrid)

# ╔═╡ 8cdbb45a-e735-4f94-97d7-a4db0fc5ec9b
plot!(p, m[T-2], c[T-2], color = :yellow, label = "T-2", marker = :circle)

# ╔═╡ 86b13890-69de-4c8f-887a-1b016d33bb43
md"
#
"

# ╔═╡ f01fef94-8a3e-4851-8be3-59c25f404a5c
m[T-3], c[T-3] = det_EGM(m[T-2],c[T-2],n,y,agrid)

# ╔═╡ 30c57fee-74dd-4e8d-a4f7-33794588c1e1
plot!(p, m[T-3], c[T-3], color = :orange, label = "T-3", marker = :circle)

# ╔═╡ 82b08239-9522-4403-a32a-5ebefd57f18e
md"
#
"

# ╔═╡ 6a20c4d3-c392-4e04-8cfd-332f400424ac
for it in (T-1):-1:1
	m[it],c[it] = det_EGM(m[it+1],c[it+1],n,y,agrid)
end

# ╔═╡ 284fed7e-7585-48dc-90fc-c00835863735
begin
	p1 = plot(m[T], c[T],title = "EGM Iterations", xlim = (-1,22), ylim = (-1,20),color = :black, marker = :circle, label = "period 5", xlab = "M", ylab = "c(M)")
	for it in (T-1):-1:1
		plot!(p1,m[it],c[it], label = "period $it",marker = :circle)
	end
end

# ╔═╡ 69a832cc-766c-42cb-be9f-68e8b89f01cb
p1

# ╔═╡ 58656e5c-5356-4d86-ad7b-33745623e096
md"
## Corner Solutions

* So far this only covers interior solutions where the Euler Equation holds exactly.
* The choice of $c$ is contrained to lie in $[0,M]$, however, which means that $0\leq A \leq M$
* The method *only works with* points $A$ chosen to respect that constraint, so there is never any issue.
* But we need to take care of the bounds ourselves.

### Lower Bound on Consumption

* Standard Inada conditions $\lim_{c\to0} = -\infty$ prevents us from ever hitting $c=0$

"

# ╔═╡ 80668dc3-ca1d-4dfb-b68e-ad64ae7abebc
md"
#
"

# ╔═╡ 8bb5d76f-cddc-45db-a0ce-1cc9eca1bc94
md"

### Upper Bound

* If we consume all resources at the upper bound, $c = M$, and we can compute the corresponding solution directly. Note that $A=0$ at that point (we save zero).
* We can rely on the fact that in this class of models with a monotonically increasing utilty function, $A = M-c$ is non-decreasing in $M$.
* The final point $M_0$ corresponding to $A=0$ is

$$M_0 = c_0 + 0 = (u')^{-1} \Big( \beta R \mathbb{E}_{y} u'(\tilde{c}\big) \Big) + 0$$

* Given non-decreasing property, for all $M < M_0$, we must have $A=0$.
* That implies that we consume all resources, hence $c = M$ at such points.
* This is just a 45 degree line connecting point $(0,0)$ and $M_0$.

"



# ╔═╡ cd05846a-0030-44e9-a4b3-01f141f8bfc2
md"
#
"

# ╔═╡ 3dbb96da-0caa-4c16-b9cb-44e447a96b22
begin
	c2 = Vector{Float64}[Float64[] for i in 1:T]   # consumption function on m
	m2 = Vector{Float64}[Float64[] for i in 1:T]   # endo grids
	m2[T]       = [0.0,M]  
    c2[T]       = [0.0,M]
	p2 = plot(m[T], c[T],title = "EGM Iterations with Corner", xlim = (-0.2,5), ylim = (-0.2,5),color = :black, marker = :circle, label = "period 5", xlab = "M", ylab = "c(M)")
	for it in (T-1):-1:1
		m2[it],c2[it] = det_EGM(m[it+1],c[it+1],n,1.0,agrid)
		plot!(p2,m2[it],c2[it], label = "period $it")
	end
	p2
end

# ╔═╡ 845f261e-9865-4cb3-a772-a46f0251f67e
md"
## Challenge

* Taking the functions from above, build an EGM solver for an infinite time model!
* Stop your iteration once successive policy funciton approximation do not incur a greater maximal absolute error than 1e-6
* write a function with the following keyword arguments: `n = 100, M = 10, tol = 1e-6,y = 1.0`
* Make a plot of the final policy function!
"

# ╔═╡ 1fefbafe-8fa9-4be3-a219-922dcf758e0e
md"
#
"

# ╔═╡ abf95185-5948-496f-9058-369587c19a1b
function EGM_inf(;n = 100, M = 10, tol = 1e-6,y = 1.0)
	agrid = range(0.0,stop = M, length = n)

	c = zeros(n+1)
	m = zeros(n+1)
	c0 = collect(range(0.0,stop = M, length = n+1))
	m0 = collect(range(0.0,stop = M, length = n+1))

	err = 100.0
	iters = 0
	while err > tol
		m, c = det_EGM(m0, c0,n,y,agrid)
		err = max( maximum(abs.(c .- c0)), maximum(abs.(m .- m0)) )
		# update from current to new functions
		m0[:] = m
		c0[:] = c
		iters += 1
	end
	(m,c,iters)
end

# ╔═╡ 00a5a487-4bf4-44bc-938a-e794c7d7ac6d
begin
	im,ic,iters = EGM_inf()
	plot(EGM_inf(),ylims = (0,2),leg = false, title = "Infinite Time EGM converged after $iters steps")
end

# ╔═╡ 089d57dd-fdbf-454f-9263-3e2ca155346e
md"
# Minimal EGM with uncertainty

* Let's go back to the initial example with log-normal income $y$
* We need to actually integrate over the RHS of the Euler Equation now.
* Reminder: $y \sim \text{LogNormal}(\mu,\sigma) \iff y = \exp(X), X \sim N(\mu,\sigma)$
* So, a good approximation scheme for the log-normal distribution can be based on getting nodes and weights for $X \sim N(\mu,\sigma)$, and then using $y_i = \exp(x_i)$, where $x_i$ is gauss-hermite node, will work well.

#
"

# ╔═╡ e80e1f7c-aa7f-48c3-a151-57ceea02286b
function minimal_EGM(;ny=5,na=100,nT=25,M=10,σ=0.25,μ=0.0,R=1.05,β=0.95,cbar = 0.0)
    nodes,weights = gausshermite(ny)  # from FastGaussQuadrature
    yvec          = sqrt(2.0) * σ .* nodes .+ μ  # gauss-hermite nodes
    ywgt          = weights .* π^(-0.5)
    avec          = collect(range(0.0,M,length = na))
    m             = Vector{Float64}[Float64[] for i in 1:nT]   # endogenous grid
    c             = Vector{Float64}[Float64[] for i in 1:nT]   # consumption function on m
    m[nT]       = [0.0,M]    
    c[nT]       = [cbar,M]

    cg = cgrad(:viridis)
    cols = cg[range(0.0,stop=1.0,length=nT)]

    pl = plot(m[nT],c[nT],label="$(nT)",leg=:topright,title="Consumption Function",
              xlims = (0,M),ylims = (0,M), color = cols[nT],
              xlab = "Cash on Hand", ylab = "Consumption")
    # cycle back in time
    for it in nT-1:-1:1
		# w1 = yshock*R*savings:  next period wealth at all states. (ny,na)
        w1 = exp.(yvec) .+ R .* avec'   
        # get next period consumption on that wealth w1
		policy = interpolate((m[it+1],),c[it+1],Gridded(Linear()))
		policy = extrapolate(policy,Line())
        c1 = reshape(policy(w1[:]),ny,na)
        c1[c1 .< cbar] .= 0.001     # don't allow negative consumption
        Emu   = ywgt' * (1 ./ c1)   # Expected marginal utility (na,1)
        rhs   = β * R * Emu[:]   # RHS of euler equation
        c[it] = 1.0 ./ rhs   # imu(u) -> c
        m[it] = avec .+ c[it]   # c -> M

        # add credit constraint region
        c[it] = vcat(cbar, c[it])   # prepend with 0
        m[it] = vcat(0.0, m[it])   #

        plot!(pl,m[it],c[it],label= it == 1 ? "$it" : "", color = cols[it])
    end
    pl = lens!(pl, [0, 2], [0, 2], inset = (1, bbox(0.2, 0.1, 0.25, 0.25)))
    pl
end

# ╔═╡ c7dde73c-e626-404d-9177-f58fc1df3972
@bind σ Slider(0.01:0.1:2.0,show_value = true)

# ╔═╡ b03ff849-715d-47ee-9299-782c14636c01
@bind μ Slider(0.0:0.1:2.0,show_value = true)

# ╔═╡ 2203856e-87fe-4a29-90c7-e0b4a065c4a1
@bind r Slider(1.0:0.01:1.1,show_value = true)

# ╔═╡ 8cd2061e-dd49-4521-94bf-de6e732af151
@bind cbar Slider(0.0:0.01:1.1,show_value = true)

# ╔═╡ b9c90c2c-4c1c-40d4-974b-e148191f67ee
minimal_EGM(σ=σ, μ = μ, R=r, cbar = cbar)

# ╔═╡ 6c7e3163-06e4-4ad6-8d55-53f8b596bde8
md"
# Conclusions about EGM

1. Very fast (not demonstrated) and accurate method
2. Works for finite and inifinite time horizon models with 1 continuous choice variable and 1 continuous state variable
3. Need analytic utility function that is invertible
4. Need to be able to compute a post-decision state variable
5. Can handle occasionally binding borrowing constraints

This is a small class of models, but very important in macro. If your application has the above features, you should almost certainly use EGM to compute the solution.
"

# ╔═╡ b62d9921-157b-4143-8246-ddd8163e6e4a
md"

# Further Resources

1. Chris Caroll's [original article](http://www.econ2.jhu.edu/people/ccarroll/EndogenousGridpoints.pdf)
1. Barillas & Fernandez-Villaverde, JEDC 2007 “A Generalization of the Endogenous Grid Method”
1. Ludwig & Schön, Computational Economics, 2018 “Endogenous Grids in Higher Dimensions: Delaunay Interpolation and Hybrid Methods”
1. Matthew White, JEDC 2015 “The Method of Endogenous Gridpoints in Theory and Practice”
1. Iskhakov, Econ Letters 2015 “Multidimensional endogenous gridpoint method: solving triangular dynamic stochastic optimization problems without root-finding operations” + Corrigendum

## Generalizations

* Adding a discrete choice makes a non-convex problem out of this: Euler Equation is no longer sufficient. 
  1. Iskhakov, Jørgensen, Rust, Schjerning, QE 2017 “The Endogenous Grid Method for Discrete-Continuous Dynamic Choice Models with (or without) Taste Shocks”
  2. Giulio Fella, RED 2014 “A Generalized Endogenous Grid Method for Non-Smooth and Non-Concave Problems”
  3. Jeppe Druedahl, Thomas Jørgensen, JEDC 2017 “A General Endogenous Grid Method for Multi-Dimensional Models with Non-Convexities and Constraints”

## DC-EGM

* Let's look at [Fedor's slides](https://github.com/dseconf/DSE2019/blob/master/11_DCEGM_Iskhakov/slides/dcegm_DSE2019.pdf) to learn more about this algorithm
* Then we will look at [my implementation of it on github](https://github.com/floswald/DCEGM.jl)
"

# ╔═╡ e5af21c8-18d2-485a-ad06-bba5d96ba69a


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
FastGaussQuadrature = "442a2c76-b920-505d-bb47-c5924d526838"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
FastGaussQuadrature = "~0.4.8"
Interpolations = "~0.13.4"
Plots = "~1.38.9"
PlutoUI = "~0.7.20"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.1"
manifest_format = "2.0"
project_hash = "c313acbf704b999bb43e8522b020cfa3a50b9517"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0bc60e3006ad95b4bb7497698dd7c6d649b9bc06"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.1"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f885e7e7c124f8c92650d61b9477b9ac2ee607dd"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.1"

[[deps.ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "9a1d594397670492219635b35a3d830b04730d62"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "5ccd8547615457402a499af9603d55876423eea8"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "0.4.8"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "0635807d28a496bb60bc15f465da0107fb29649c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "99e248f643b052a77d2766fe1a16fb32b661afd4"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.0+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "37e4657cd56b11abe3d10cd4a1ec5fbdb4180263"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.7.4"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "2422f47b34d4b127720a18f86fa7b1aa2e141f29"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.18"

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

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "6503b77492fd7fcb9379bf73cd31035670e3c509"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "c95373e73290cf50a8a22c3375e4625ded5c5280"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "186d38ea29d5c4f238b2d9fe6e1653264101944b"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.9"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "1e0cb51e0ccef0afc01aab41dc51a3e7f781e8cb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.20"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "261dddd3b862bd2c940cf6ca4d1c8fe593e457c8"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.3"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "e974477be88cb5e3040009f3767611bc6357846f"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.11"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e08890d19787ec25029113e88c34ec20cac1c91e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.0.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

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

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "94f38103c984f89cf77c402f2a68dbd870f8165f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.11"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╠═d257d214-0e18-444d-aad9-5f37ec978cf5
# ╟─2cc56380-3907-4be6-b699-288093fccb50
# ╟─b581ffce-9d63-4fbf-992b-8f9280d83a92
# ╟─5285eb0a-a266-11eb-277e-a7c01cf568df
# ╟─32579cd1-07c0-43f5-bde6-5a6a9f52a999
# ╟─599b4b2e-dcfc-4790-a0bd-f9a60e9fadd1
# ╟─0171a47c-f777-410d-9814-8c86da85bf3e
# ╟─bf784923-f460-4816-87d6-5890f959e015
# ╟─2dd54158-d62f-4b12-9c18-486e0a52596a
# ╟─51b82e35-fba3-4100-b313-254ece33449d
# ╟─eb1b201c-1598-4740-b7e9-5108f017f4e3
# ╟─2333dde1-68cc-4155-82e7-d8727ef596f0
# ╟─590b81d4-26bc-46c8-82ed-2998fcffe706
# ╠═e6376b86-56ee-4ace-9e9a-5ceb87bed7db
# ╠═8292ee9b-e13a-4921-888f-4f898cebc6c4
# ╟─0eb5052b-7d30-48d4-926e-15a43c3ee7de
# ╟─48b0b0dd-9487-4fbd-8ec7-bf61272d4ba6
# ╟─3317d300-0e2d-481b-ab0c-7c1fcd319162
# ╠═5ddaef03-bbdf-4374-82e7-fc9eb3d8fd12
# ╟─fa84b929-6088-4bba-bb29-d0cc4d8f9b9d
# ╠═034ff9fe-b925-4f6c-b9f8-e28a86e7ef9d
# ╟─c12a930f-cf42-46e8-9f88-06d00aaba002
# ╟─aac8dfa3-97b1-4786-baea-8f85524a27ac
# ╠═6616459c-4542-4e5d-9f1f-926ebff695d2
# ╟─3f2351e8-aecc-420f-ac3c-f5322021a359
# ╟─4aba99ee-455d-44b8-8630-be8a391ac291
# ╟─8cdbb45a-e735-4f94-97d7-a4db0fc5ec9b
# ╟─86b13890-69de-4c8f-887a-1b016d33bb43
# ╟─f01fef94-8a3e-4851-8be3-59c25f404a5c
# ╟─30c57fee-74dd-4e8d-a4f7-33794588c1e1
# ╟─82b08239-9522-4403-a32a-5ebefd57f18e
# ╠═6a20c4d3-c392-4e04-8cfd-332f400424ac
# ╠═284fed7e-7585-48dc-90fc-c00835863735
# ╠═69a832cc-766c-42cb-be9f-68e8b89f01cb
# ╟─58656e5c-5356-4d86-ad7b-33745623e096
# ╟─80668dc3-ca1d-4dfb-b68e-ad64ae7abebc
# ╟─8bb5d76f-cddc-45db-a0ce-1cc9eca1bc94
# ╟─cd05846a-0030-44e9-a4b3-01f141f8bfc2
# ╟─3dbb96da-0caa-4c16-b9cb-44e447a96b22
# ╟─845f261e-9865-4cb3-a772-a46f0251f67e
# ╟─1fefbafe-8fa9-4be3-a219-922dcf758e0e
# ╠═abf95185-5948-496f-9058-369587c19a1b
# ╟─00a5a487-4bf4-44bc-938a-e794c7d7ac6d
# ╟─089d57dd-fdbf-454f-9263-3e2ca155346e
# ╠═e80e1f7c-aa7f-48c3-a151-57ceea02286b
# ╠═d40c12c8-b0ac-4c2c-ad5c-86c918aad387
# ╠═c7dde73c-e626-404d-9177-f58fc1df3972
# ╠═b03ff849-715d-47ee-9299-782c14636c01
# ╠═2203856e-87fe-4a29-90c7-e0b4a065c4a1
# ╠═8cd2061e-dd49-4521-94bf-de6e732af151
# ╠═b9c90c2c-4c1c-40d4-974b-e148191f67ee
# ╟─6c7e3163-06e4-4ad6-8d55-53f8b596bde8
# ╟─b62d9921-157b-4143-8246-ddd8163e6e4a
# ╠═e5af21c8-18d2-485a-ad06-bba5d96ba69a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
