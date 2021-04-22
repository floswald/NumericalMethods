### A Pluto.jl notebook ###
# v0.14.2

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

# ╔═╡ d257d214-0e18-444d-aad9-5f37ec978cf5
using Plots

# ╔═╡ cd9da258-d1b0-4a26-8d88-69c573707969
using Interpolations

# ╔═╡ ec324cd5-b073-47f3-af2f-aec6ec0fb1cf
using FastGaussQuadrature

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

*ScPo Numerical Methods 2021, Florian Oswald*

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
	T = 5  # let's to 5 iterations
	
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

# ╔═╡ 3317d300-0e2d-481b-ab0c-7c1fcd319162
begin
	# current policy
	polT = extrapolate(interpolate((m[T],), c[T], Gridded(Linear())), Linear()) 
	pltT = plot(m[T], c[T],title = "First Iteration: T-1", xlim = (-1,22), ylim = (-1,20),color = :black, marker = :circle, label = "period T", xlab = "M", ylab = "c(M)", ratio = 1)
	anim = @animate for j in eachindex(agrid)
		mj,cj = det_egm_single(agrid[j],polT,y)
		scatter!(pltT,[mj],[cj], label = "A$j = $(round(agrid[j],digits=1))")
	end 
	gif(anim, fps = 0.7)
end

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
		err = maximum(abs.(c .- c0))
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
function minimal_EGM(;ny=5,na=100,nT=25,M=10,σ=0.25,μ=0.0,R=1.05,β=0.95)
    nodes,weights = gausshermite(ny)  # from FastGaussQuadrature
    yvec          = sqrt(2.0) * σ .* nodes .+ μ  # gauss-hermite nodes
    ywgt          = weights .* π^(-0.5)
    avec          = collect(range(0.0,M,length = na))
    m             = Vector{Float64}[Float64[] for i in 1:nT]   # endogenous grid
    c             = Vector{Float64}[Float64[] for i in 1:nT]   # consumption function on m
    m[nT]       = [0.0,M]    
    c[nT]       = [0.0,M]

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
        c1[c1 .< 0] .= 0.001     # don't allow negative consumption
        Emu   = ywgt' * (1 ./ c1)   # Expected marginal utility (na,1)
        rhs   = β * R * Emu[:]   # RHS of euler equation
        c[it] = 1.0 ./ rhs   # imu(u) -> c
        m[it] = avec .+ c[it]   # c -> M

        # add credit constraint region
        c[it] = vcat(0.0, c[it])   # prepend with 0
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

# ╔═╡ b9c90c2c-4c1c-40d4-974b-e148191f67ee
minimal_EGM(σ=σ, μ = μ, R=r)

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


# ╔═╡ Cell order:
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
# ╠═d257d214-0e18-444d-aad9-5f37ec978cf5
# ╠═cd9da258-d1b0-4a26-8d88-69c573707969
# ╟─0eb5052b-7d30-48d4-926e-15a43c3ee7de
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
# ╟─abf95185-5948-496f-9058-369587c19a1b
# ╟─00a5a487-4bf4-44bc-938a-e794c7d7ac6d
# ╟─089d57dd-fdbf-454f-9263-3e2ca155346e
# ╠═ec324cd5-b073-47f3-af2f-aec6ec0fb1cf
# ╠═e80e1f7c-aa7f-48c3-a151-57ceea02286b
# ╠═d40c12c8-b0ac-4c2c-ad5c-86c918aad387
# ╠═c7dde73c-e626-404d-9177-f58fc1df3972
# ╠═b03ff849-715d-47ee-9299-782c14636c01
# ╠═2203856e-87fe-4a29-90c7-e0b4a065c4a1
# ╠═b9c90c2c-4c1c-40d4-974b-e148191f67ee
# ╟─6c7e3163-06e4-4ad6-8d55-53f8b596bde8
# ╟─b62d9921-157b-4143-8246-ddd8163e6e4a
# ╠═e5af21c8-18d2-485a-ad06-bba5d96ba69a
