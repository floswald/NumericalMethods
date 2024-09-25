### A Pluto.jl notebook ###
# v0.19.40

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

# ╔═╡ 9c61c9d5-0d3d-47b7-8e7d-f929cfec3311
begin
	# 
	using LaTeXStrings
	using JuMP
	using Plots
	using Ipopt
	using GLPK
	using Distributions
	using Statistics
	using OrderedCollections
	using LinearAlgebra
	using Test
	using NLopt
	using PlutoUI
end

# ╔═╡ aa1f0719-5e1d-4294-815e-349d1cebc004
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
		font-size: x-large
	}
</style>
"""

# ╔═╡ aae2e545-198c-42bb-9e97-6d6635bee02a
html"<button onclick='present()'>present</button>"

# ╔═╡ 4468041e-4472-11ec-16b7-553f4def2b18
md"""
# Constraints

Recall our core optimization problem:

$$\min_{x\in\mathbb{R}^n} f(x)  \text{ s.t. } x \in \mathcal{X}$$

* Up to now, the feasible set was $\mathcal{X} \in \mathbb{R}^n$. 
* In **constrained problems** $\mathcal{X}$ is a subset thereof.
* We already encountered *box constraints*, e.g. $x \in [a,b]$.
* Sometimes the contrained solution coincides with the unconstrained one, sometimes it does not.
* There are *equality constraints* and *inequality constraints*.

"""

# ╔═╡ e66005ad-1f51-462a-9329-d77f7695cbfc
md"""
## Lagrange Multipliers

* Used to optimize a function subject to equality constraints.

$$\begin{aligned}
\min_x & f(x) \\
\text{subject to } & h(x) = 0
\end{aligned}$$

where both $f$ and $h$ have continuous partial derivatives.

* We look for contour lines of $f$ that are aligned to contours of $h(x) = 0$.

In other words, we want to find the best $x$ s.t. $h(x) = 0$ and we have

$$\nabla f(x) = \lambda \nabla h(x)$$

for some *Lagrange Mutliplier* $\lambda$
* Notice that we need the scalar $\lambda$ because the magnitudes of the gradients may be different.
* We therefore form the the **Lagrangian**:

$$\mathcal{L}(x,\lambda) = f(x) - \lambda h(x)$$
"""

# ╔═╡ 5bc7c8ce-3a4a-4505-a2fa-2b25f041437a
md"""
## Example

Suppose we have

$$\begin{align}
\min_x & -\exp\left( -\left( x_1 x_2 - \frac{3}{2} \right)^2 - \left(x_2 - \frac{3}{2}\right)^2 \right) \\
\text{subject to } & x_1 - x_2^2 = 0
\end{align}$$

We form the Lagrangiagn:

$$\mathcal{L}(x_1,x_2,\lambda) = -\exp\left( -\left( x_1 x_2 - \frac{3}{2} \right)^2 - \left(x_2 - \frac{3}{2}\right)^2 \right) - \lambda(x_1 - x_2^2)$$

Then we compute the gradient wrt to $x_1,x_2,\lambda$, set to zero and solve.
"""

# ╔═╡ 9a92426e-bc9a-4ead-a93c-ec1c3f23ca57
begin
	f(x1,x2) = -exp.(-(x1.*x2 - 3/2).^2 - (x2-3/2).^2)
	c(z) = sqrt(z)
	x=0:0.01:3.5	
end

# ╔═╡ d7acef71-fe0d-4062-9342-9061afde1293
let
	p1 = surface(x,x,(x,y)->f(x,y),xlab = L"x_1", ylab = L"x_2")
	scatter3d!(p1,[1.358],[1.165],[f(1.358,1.165)],markercolor=:red,leg=false)
	p2 = contour(x,x,(x,y)->f(x,y),lw=1.5,levels=[collect(0:-0.1:-0.85)...,-0.887,-0.95,-1],xlab = L"x_1", ylab = L"x_2")
	plot!(p2,c,0.01,3.5,label="",lw=2,color=:black)
	scatter!(p2,[1.358],[1.165],markersize=5,markercolor=:red,label="Constr. Optimum")
	plot(p1,p2,size=(900,300))
end

# ╔═╡ de879e63-fdc9-486e-9aaa-1a55c32d93d5
md"""
* If we had multiple constraints ($l$), we'd just add them up to get

$$
\mathcal{L}(\mathbf{x},\mathbf{\lambda}) = f(\mathbf{x}) - \sum_{i=1}^l \lambda_i h_i(\mathbf{x})$$
"""

# ╔═╡ 54a17a40-fd67-4040-99b9-d0d460385155
md"""
## Inequality Constraints

Suppose now we had

$$\begin{align}
\min_\mathbf{x} & f(\mathbf{x}) \\
\text{subject to } & g(\mathbf{x}) \leq 0
\end{align}$$

which, if the solution lies *on* the constraint *boundary*, means that

$$\nabla f - \mu \nabla g = 0$$

for some scalar $\mu$ - as before. 

* In this case, we say the **constraint is active**.
* In the opposite case, i.e. the solution lies **inside** the contrained region, we way the contraint is **inactive**. 
* In that case, we are back to an *unconstrained* problem, look for $\nabla f = 0$, and set $\mu=0$.


"""

# ╔═╡ 9f1f0926-b1b3-4ef3-aebe-277531abfd33
let
	# the blue area shows the FEASIBLE SET
	contour(x,x,(x,y)->f(x,y),lw=1.5,levels=[collect(0:-0.1:-0.85)...,-0.887,-0.95,-1], title = "Constraint is Active or Binding")
	plot!(c,0.01,3.5,label="",lw=2,color=:black,fill=(0,0.5,:blue))
	scatter!([1.358],[1.165],markersize=5,markercolor=:red,label="Constr. Optimum")
end

# ╔═╡ 2f8d4513-0871-48f3-acfb-f7a0254f7b8e
let
	c2(x1) = 1+sqrt(x1)
	contour(x,x,(x,y)->f(x,y),lw=1.5,levels=[collect(0:-0.1:-0.85)...,-0.887,-0.95,-1], title = "Constraint is Inactive or Slack")
	plot!(c2,0.01,3.5,label="",lw=2,color=:black,fill=(0,0.5,:blue))
	scatter!([1],[1.5],markersize=5,markercolor=:red,label="Unconstr. Optimum")
end

# ╔═╡ 8901f1e5-f243-4fe7-afda-6cc951b234a5
md"""
## Infinity Step

* We could do an **infinite step** to avoid *infeasible points*:

$$\begin{align}
f_{\infty\text{-step}} &= \begin{cases}
f(\mathbf{x}) & \text{if } g(\mathbf{x}) \leq 0 \\
\infty & \text{else. } 
\end{cases}\\
 &= f(\mathbf{x}) + \infty (g(\mathbf{x} > 0)
\end{align}$$

* Unfortunately, this is discontinous and non-differentiable, i.e. hard to handle for algorithms.
* Instead, we use a *linear penalty* $\mu g(\mathbf{x})$ on the objective if the constraint is violated.

"""

# ╔═╡ fe6b7d74-1c6f-4253-9254-57d536bb24c5
md"""
* The penalty provides a lower bound to $\infty$:

$$\mathcal{L}(\mathbf{x},\mu) = f(\mathbf{x}) + \mu g(\mathbf{x})$$

* We can get back the infinite step by maximizing the penalty:

$$f_{\infty\text{-step}} = \max_{\mu\geq 0} \mathcal{L}(\mathbf{x},\mu)$$

* Every infeasible $\mathbf{x}$ returns $\infty$, all others return $f(\mathbf{x})$
"""

# ╔═╡ 19d12886-3980-4116-8772-2c1d17b14270
md"""
## Kuhn-Karush-Tucker (KKT)

* Our problem thus becomes

$$\min_\mathbf{x} \max_{\mu\geq 0} \mathcal{L}(\mathbf{x},\mu)$$

* This is called the **primal problem**. Optimizing this requires:


1. Point is feasible.: $g(\mathbf{x}^*) \leq 0$. 
2. Penalty goes into the right direction: $\mu \geq 0$.  *Dual feasibility*.
3. Feasible point on the boundary: $\mu g(\mathbf{x}^*) = 0$  has $g(\mathbf{x}) = 0$, otherwise $g(\mathbf{x}) < 0$ and $\mu =0$.
4. Active constraint: $\nabla f(\mathbf{x}^*) - \mu \nabla g(\mathbf{x}^*) = 0$. With an active constraint, we want parallel contours of objective and constraint. When inactive, our optimum just has $\nabla f(\mathbf{x}^*) = 0$, which means $\mu = 0$.

The preceding four conditions are called the **Kuhn-Karush-Tucker (KKT)** conditions. In the above order, and in general terms, they are:

1. Feasibility
2. Dual Feasibility
3. Complementary Slackness
4. Stationarity.

The KKT conditions are the FONCs for problems with smooth constraints.

"""

# ╔═╡ 583a5640-adde-4f4f-a344-4e77e04e431d
md"""
## Duality

We can combine equality and inequality constraints:

$$\mathcal{L}(\mathbf{x},\mathbf{\lambda},\mathbf{\mu}) = f(\mathbf{x}) + \sum_{i} \lambda_i h_i(\mathbf{x}) + \sum_j \mu_j g_j(\mathbf{x})$$

where, notice, we reverted the sign of $\lambda$ since this is unrestricted.

* The Primal problem is identical to the original problem and just as difficult to solve:

$$\min_\mathbf{x} \max_{\mathbf{\mu}\geq 0,\mathbf{\lambda}} \mathcal{L}(\mathbf{x},\mathbf{\mu},\mathbf{\lambda})$$

* The Dual problem reverses min and max:

$$\max_{\mathbf{\mu}\geq 0,\mathbf{\lambda}} \min_\mathbf{x}  \mathcal{L}(\mathbf{x},\mathbf{\mu},\mathbf{\lambda})$$


### Dual Values

* The *max-min-inequality* states that for any function $f(a,b)$

$$\max_\mathbf{a} \min_\mathbf{b} f(\mathbf{a},\mathbf{b}) \leq \min_\mathbf{b} \max_\mathbf{a} f(\mathbf{a},\mathbf{b})$$

* Hence, the solution to the dual is a lower bound to the solution of the primal problem.
* The solution to the *dual function*, $\min_\mathbf{x}  \mathcal{L}(\mathbf{x},\mathbf{\mu},\mathbf{\lambda})$ is the min of a collection of linear functions, and thus always concave.
* It is easy to optimize this.
* In general, solving the dual is easy whenever minimizing $\mathcal{L}$ wrt $x$ is easy.



"""

# ╔═╡ 9be3cbee-7949-42da-9496-6afc94785f98
md"""
## Penalty Methods

* We can convert the constrained problem back to unconstrained by adding penalty terms for constraint violoations.
* A simple method could just count the number of violations:

$$p_\text{count}(\mathbf{x}) = \sum_{i} (h_i(\mathbf{x}) \neq 0 ) + \sum_j  (g_j(\mathbf{x} > 0)$$

* and add this to the objective in an *unconstrained* problem with penalty $\rho > 0$

$$\min_\mathbf{x} f(\mathbf{x}) + \rho p_\text{count}(\mathbf{x})$$

* One can choose the penalty function: for example, a quadratic penaly will produce a smooth objective function
* Notice that $\rho$ needs to become very large sometimes here.
"""

# ╔═╡ 91da671d-7602-490c-a94c-bc9ff4eca59a
md"""
## Interior Point Method

* Also called *barrier method*.
* These methods make sure that the search point remains always feasible.
* As one approaches the constraint boundary, the barrier function goes to infinity. Properties:

1. continuous: $p_\text{barrier}(\mathbf{x})$ 
2. non negative: $p_\text{barrier}(\mathbf{x})$ 
3. goes to infinitey : $p_\text{barrier}(\mathbf{x})$ as one approaches the constraint boundary

### Barriers

* Inverse Barrier

$$p_\text{barrier}(\mathbf{x}) = -\sum_i \frac{1}{g_i(\mathbf{x})}$$

* Log Barrier

$$p_\text{barrier}(\mathbf{x}) = -\sum_i \begin{cases}\log(-g_i(\mathbf{x})) & \text{if } g_i(\mathbf{x}) \geq -1 \\
0& \text{else.} 
\end{cases}$$

* The approach is as before, one transforms the problem to an unconstrained one and increases $\rho$ until convergence:

$$\min_\mathbf{x} f(\mathbf{x}) + \frac{1}{\rho} p_\text{barrier}(\mathbf{x})$$

### Examples

$$\min_{x \in \mathbb{R}^2} \sqrt{x_2} \text{ subject to }\begin{array}{c} \\
 x_2 \geq 0 \\
 x_2 \geq (a_1 x_1 + b_1)^3 \\
x_2 \geq (a_2 x_1 + b_2)^3 
\end{array}$$

## Constrained Optimisation with [`NLopt.jl`](https://github.com/JuliaOpt/NLopt.jl)

* We need to specify one function for each objective and constraint.
* Both of those functions need to compute the function value (i.e. objective or constraint) *and* it's respective gradient. 
* `NLopt` expects contraints **always** to be formulated in the format 
	
$$g(x) \leq 0$$

where $g$ is your constraint function
* The constraint function is formulated for each constraint at $x$. it returns a number (the value of the constraint at $x$), and it fills out the gradient vector, which is the partial derivative of the current constraint wrt $x$.
* There is also the option to have vector valued constraints, see the documentation.
* We set this up as follows:

"""

# ╔═╡ 7e5cedbf-05da-4573-994f-42099fe4b33a
begin
	function objective(x::Vector, grad::Vector)
	    if length(grad) > 0
	        grad[1] = 0
	        grad[2] = 0.5/sqrt(x[2])
	    end
    	sqrt(x[2])
	end
	function constraint(x::Vector, grad::Vector, a, b)
    	if length(grad) > 0
        grad[1] = 3a * (a*x[1] + b)^2
        grad[2] = -1
    	end
    (a*x[1] + b)^3 - x[2]
	end
end

# ╔═╡ b1720238-495e-47f7-ae75-866723f5fca4
x -> x^2

# ╔═╡ 1e31dc48-e849-4045-8dcc-d05889f8609b
let
	opt = Opt(:LD_MMA, 2)
	lower_bounds!(opt, [-Inf, 0.])
	xtol_rel!(opt,1e-4)

	min_objective!(opt, objective)
	inequality_constraint!(opt, (x,g) -> constraint(x,g,2,0), 1e-8)
	inequality_constraint!(opt, (x,g) -> constraint(x,g,-1,1), 1e-8)

	(minfunc,minx,ret) = NLopt.optimize(opt, [1.234, 5.678])
end

# ╔═╡ 70893b5f-51e8-4536-b5aa-4e353768bc52
md"""
## NLopt: Rosenbrock

* Let's tackle the rosenbrock example again.
* To make it more interesting, let's add an inequality constraint.
	
$$\min_{x\in \mathbb{R}^2} (1-x_1)^2  + 100(x_2-x_1^2)^2  \text{  subject to  } 0.8 - x_1^2 -x_2^2 \geq 0$$
* in `NLopt` format, the constraint is $x_1^2 + x_2^2 - 0.8 \leq 0$
"""

# ╔═╡ 2a74858c-6d13-4961-86fa-125e4c01e232
function rosenbrockf(x::Vector,grad::Vector)
    if length(grad) > 0
        grad[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
        grad[2] = 200.0 * (x[2] - x[1]^2)
    end
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

# ╔═╡ ec0c21da-c517-4d2e-9832-18eca242621b
function r_constraint(x::Vector, grad::Vector; radius = 0.8)
    if length(grad) > 0
	grad[1] = 2*x[1]
	grad[2] = 2*x[2]
	end
	return x[1]^2 + x[2]^2 - radius
end

# ╔═╡ ade668a5-87d6-40aa-b670-d0cc41148757
md"""
radius = $(@bind rad Slider(0.1:0.1:2.0,show_value = true,default = 0.2))
"""

# ╔═╡ 52a1ca3b-8a10-4c0a-b4d4-82f995b4b3df
let
	grad = zeros(2)
	xrange = collect(-2.5:0.01:2)
	cc = contour(xrange,xrange, (x,y)->sqrt(rosenbrockf([x, y],grad)), fill=false, color=:viridis, ylab = L"x_2",xlab = L"x_1", leg = :topleft,levels = exp.(range(log(0.1), stop = log(70.0), length = 50)), cbar = false)
	contour!(cc,xrange,xrange,(x,y)->r_constraint([x, y],grad,radius = rad), levels = [0],lw = 3)
	scatter!(cc,[1],[1],color = :red, lab = "unconstrained optimizer")
	
	# let's compute the constrained optimizer now!
	opt = Opt(:LD_MMA, 2)
	lower_bounds!(opt, [-5, -5.0])
	min_objective!(opt,(x,g) -> rosenbrockf(x,g))
	inequality_constraint!(opt, (x,g) -> r_constraint(x,g,radius = rad))
	ftol_rel!(opt,1e-9)
	(minfunc,minx,ret) = NLopt.optimize(opt, [-1.0,0.0])
	
	scatter!(cc, [minx[1]], [minx[2]], label = "constrained optimizer", leg = :topleft)
end

# ╔═╡ 9cc92292-4531-4f4c-b9b1-2a7a698529ac
md"""
## JuMP.jl

* Introduce [`JuMP.jl`](https://jump.dev/)
* JuMP is a mathematical programming interface for Julia. It is like AMPL, but for free and with a decent programming language.
* The main highlights are:
  * It uses automatic differentiation to compute derivatives from your expression.
  * It supplies this information, as well as the sparsity structure of the Hessian to your preferred solver.
  * It decouples your problem completely from the type of solver you are using. This is great, since you don't have to worry about different solvers having different interfaces.
  * In order to achieve this, `JuMP` uses [`MathProgBase.jl`](https://github.com/JuliaOpt/MathProgBase.jl), which converts your problem formulation into a standard representation of an optimization problem.
* Let's look at the readme
* The technical citation is Lubin et al <cite data-cite=JuMP></cite>


"""

# ╔═╡ fc75bc7b-0fe0-4f90-a425-6ed6e63acb4a
let
	model = Model(GLPK.Optimizer)
	@variable(model, 0 <= x <= 2)
	@variable(model, 0 <= y <= 30)
	# next, we set an objective function
	@objective(model, Max, 5x + 3 * y)

	# maybe add a constraint called "con"?
	@constraint(model, con, 1x + 5y <= 3)

	#At this stage JuMP has a mathematical representation of our model internalized
	#The MathProgBase machinery knows now exactly how to translate that to different solver interfaces
	#For us the only thing left: hit the button!
	JuMP.optimize!(model)
	(termination_status(model),value(x),value(y))
end

# ╔═╡ 2d7bdf41-85fc-4d0c-a387-60f9c9b36fcb
# JuMP: nonlinear Rosenbrock Example
# Instead of hand-coding first and second derivatives, you only have to give `JuMP` expressions for objective and constraints.
# Here is an example.
let
    m = Model(Ipopt.Optimizer)

    @variable(m, x)
    @variable(m, y)

    @NLobjective(m, Min, (1-x)^2 + 100(y-x^2)^2)
	@constraint(m, x + y <= 1.5)

    JuMP.optimize!(m)
    (termination_status(m),value(x),value(y))
end

# ╔═╡ 3ba47370-c4df-4d22-8bda-0b7264927ae6
md"""
## JuMP: Maximium Likelihood

* Let's redo the maximum likelihood example in JuMP.
* Let $\mu,\sigma^2$ be the unknown mean and variance of a random sample generated from the normal distribution.
* Find the maximum likelihood estimator for those parameters!
* density:

$$f(x_i|\mu,\sigma^2) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

* Likelihood Function

$$\begin{aligned} 
L(\mu,\sigma^2) = \Pi_{i=1}^N f(x_i|\mu,\sigma^2) =& \frac{1}{(\sigma \sqrt{2\pi})^n} \exp\left(-\frac{1}{2\sigma^2} \sum_{i=1}^N (x_i-\mu)^2 \right) \\
	 =& \left(\sigma^2 2\pi\right)^{-\frac{n}{2}} \exp\left(-\frac{1}{2\sigma^2} \sum_{i=1}^N (x_i-\mu)^2 \right) 
\end{aligned}$$

* Constraints: $\mu\in \mathbb{R},\sigma>0$
* log-likelihood: 

$$\log L = l = -\frac{n}{2} \log \left( 2\pi \sigma^2 \right) - \frac{1}{2\sigma^2} \sum_{i=1}^N (x_i-\mu)^2$$
"""

# ╔═╡ 84b81d87-d015-4cc2-8649-c94ba7770bd9
let
    distrib = Normal(4.5,3.5)
    n = 10000
    
    data = rand(distrib,n);
    
    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, MOI.Silent(), true)

    @variable(m, μ, start = 0.0)
    @variable(m, σ >= 0.0, start = 1.0)  # notice bound constraint
    
    @NLobjective(m, Max, 
		-(n/2) * log(2π * σ^2) - sum( (data[i] - μ)^2 for i = 1:n)/(2 * σ^2)
		)
    
    JuMP.optimize!(m)
	md"""
	parameter | data (truth) | estimate
	--------  | ------------ | --------
	μ         | $(value(μ)) | $(mean(data))
	σ         | $(value(σ))  | $(std(data))
	"""
end

# ╔═╡ 061de138-b6fe-4437-aa26-0afaec2e66f0
md"""
# Linear Constrained Problems (LPs)

* Very similar to before, just that both objective and constraints are *linear*.

$$\begin{align}
\min_\mathbf{x} & \mathbf{c}^T \mathbf{x}\\
\text{subject to } & \mathbf{w}_{LE}^{(i)T} \mathbf{x} \leq b_i \text{ for  }i\in{1,2,3,\dots}\\
& \mathbf{w}_{GE}^{(j)T} \mathbf{x} \geq b_j \text{ for  }j\in{1,2,3,\dots}\\
 & \mathbf{w}_{EQ}^{(k)T} \mathbf{x} = b_k \text{ for  }k\in{1,2,3,\dots}\\
\end{align}$$

* Our initial JuMP example was of that sort.

## A Cannery Problem

* A can factory (a cannery) has plants in Seattle and San Diego
* They need to decide how to serve markets New-York, Chicago, Topeka
* Firm wants to 
    1. minimize shipping costs
    2. shipments cannot exceed capacity
    3. shipments must satisfy demand
* Formalize that!
* Plant capacity $cap_i$, demands $d_j$ and transport costs from plant $i$ to market  $j$, $dist_{i,j} c$ are all given.
* Let $\mathbf{x}$ be a matrix with element $x_{i,j}$ for number of cans shipped from $i$ to $j$.

## From Maths ...

$$\begin{align}
\min_\mathbf{x}    & \sum_{i=1}^2 \sum_{j=1}^3 dist_{i,j}c \times x_{i,j}\\
\text{subject to } & \sum_{j=1}^3 x(i,j) \leq cap_i , \forall i \\
                   & \sum_{i=1}^2 x(i,j) \geq d_j , \forall j 
\end{align}$$

## ... to `JuMP`


"""

# ╔═╡ febc5df1-4ddd-48a7-94f8-251d316c72d0
# ... to JuMP
# https://github.com/JuliaOpt/JuMP.jl/blob/release-0.19/examples/cannery.jl
#  Copyright 2017, Iain Dunning, Joey Huchette, Miles Lubin, and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################
# JuMP
# An algebraic modeling language for Julia
# See http://github.com/JuliaOpt/JuMP.jl
#############################################################################
"""
    example_cannery(; verbose = true)
JuMP implementation of the cannery problem from Dantzig, Linear Programming and
Extensions, Princeton University Press, Princeton, NJ, 1963.
Author: Louis Luangkesorn
Date: January 30, 2015
"""
function example_cannery(; verbose = true)
    plants = ["Seattle", "San-Diego"]
    markets = ["New-York", "Chicago", "Topeka"]

    # Capacity and demand in cases.
    capacity = [350, 600]
    demand = [300, 300, 300]

    # Distance in thousand miles.
    distance = [2.5 1.7 1.8; 2.5 1.8 1.4]

    # Cost per case per thousand miles.
    freight = 90

    num_plants = length(plants)
    num_markets = length(markets)

    cannery = Model(GLPK.Optimizer)

    @variable(cannery, ship[1:num_plants, 1:num_markets] >= 0)

    # Ship no more than plant capacity
    @constraint(cannery, capacity_con[i in 1:num_plants],
        sum(ship[i,j] for j in 1:num_markets) <= capacity[i]
    )

    # Ship at least market demand
    @constraint(cannery, demand_con[j in 1:num_markets],
        sum(ship[i,j] for i in 1:num_plants) >= demand[j]
    )

    # Minimize transporatation cost
    @objective(cannery, Min, sum(distance[i, j] * freight * ship[i, j]
        for i in 1:num_plants, j in 1:num_markets)
    )

    JuMP.optimize!(cannery)

    if verbose
        println("RESULTS:")
        for i in 1:num_plants
            for j in 1:num_markets
                println("  $(plants[i]) $(markets[j]) = $(JuMP.value(ship[i, j]))")
            end
        end
    end

    @test JuMP.termination_status(cannery) == MOI.OPTIMAL
    @test JuMP.primal_status(cannery) == MOI.FEASIBLE_POINT
    @test JuMP.objective_value(cannery) == 151200.0
end


# ╔═╡ 1334fc66-23bb-4a9a-9534-6f4d7a88d235
example_cannery()

# ╔═╡ 520c3423-01d4-4794-893c-f7e5546d26f6
md"""
# Discrete Optimization / Integer Programming

* Here the choice variable is contrained to come from a discrete set $\mathcal{X}$. 
* If this set is $\mathcal{X} = \mathbb{N}$, we have an **integer program**
* If only *some* $x$ have to be discrete, this is a **mixed integer program**

## Example

$$\begin{align}
\min_\mathbf{x}    & x_1 + x_2\\
\text{subject to } & ||\mathbf{x}|| \leq 2\\
\text{and }    & \mathbf{x} \in \mathbb{N}
\end{align}$$

* continuous optimum is $(-\sqrt{2},-\sqrt{2})$ and objective is $y=-2\sqrt{2}=-2.828$
* Integer constrained problem is only delivering $y=-2$, and $\mathbf{x}^*\in \{(-2,0),(-1,-1),(0,-2)\}$
"""

# ╔═╡ ee374307-043c-473d-b561-192718ea612b
let
	x = -3:0.01:3
	dx = repeat(range(-3,stop = 3, length = 7),1,7)
	contourf(x,x,(x,y)->x+y,color=:heat)
	scatter!(dx,dx',legend=false,markercolor=:white)
	plot!(x->sqrt(4-x^2),-2,2,c=:white)
	plot!(x->-sqrt(4-x^2),-2,2,c=:white)
	scatter!([-2,-1,0],[0,-1,-2],c=:red)
	scatter!([-sqrt(2)],[-sqrt(2)],c=:red,markershape=:cross,markersize=11)
end

# ╔═╡ 636f7fe1-a518-4f09-bdbd-bcb79703dda5
md"""
## Rounding

* One solution is to just *round the continuous solution to the nearest integer*
* We compute the **relaxed** problem, i.e. the one where $x$ is continuous.
* Then we round up or down.
* Can go terribly wrong.

## Cutting Planes

* This is an exact method
* We solve the relaxed problem first.
* Then we add linear constraints that result in the solution becoming integral.

## Branch and Bound

* This enumerates all possible soultions.
* Branch and bound does this, without having to compute all of them.
"""

# ╔═╡ 2fcf8444-202a-453a-af28-0eade00e3981
md"""
## Example: The Knapsack Problem

* We are packing our knapsack for a trip but only have space for the most valuable items.
* We have $x_i=0$ if item $i$ is not in the sack, 1 else.

$$\begin{align}
\max_x &  \sum_{i=1}^n v_i x_i \\
\text{s.t. } & \sum_{i=1}^n w_i x_i \leq w_{max} \\
w_i \in \mathbb{N}_+,  & v_i \in \mathbb{R}
\end{align}$$

* If there are $n$ items, we have $2^n$ possible design vectors.
* But there is a useful recursive relationship.
* If we solved $n-1$ knapsack problems so far and deliberate about item $n$
    * If it's not worth including item $n$, then the solution **is** the knapsack problem for $n-1$ items and capacity $w_{\max}$
    * If it IS worth including it: solution will have value of knapsack with $n-1$ items and reduced capacity, plus the value of the new item
* This is dynamic progamming.
"""

# ╔═╡ 3f231741-12f6-4426-8d46-b6cfe623a248
(1:10).^2

# ╔═╡ 8eef8fc6-2272-46ff-a5ab-b2a4e63d9b7a
let
	n , w , v , W = 5, [ 2, 8, 4, 2, 5 ] , [ 5, 3, 2, 7, 4 ] , 10
	m = zeros(n,W+1)  # m[i,j] is value of including i if capacity is j
	for i in 1:n
		for (jx,jw) in enumerate(0:W)  # if current capacity is
			if w[i] > jw
				m[i,jx] = i == 1 ? 0.0 : m[i-1,jx]  # if weight exceeds current capacity, value of problem is identical to value of problem without the i-th item at that capacity.
			else
				# weight is feasible to add: should we add it?
				m[i,jx] = i == 1 ? v[i] : max(m[i-1,jx], m[i-1, jx - jw] + v[i])
			end
		end
	end
	m
end

# ╔═╡ b1f21cfe-ebc2-4321-9636-49739568a7de
md"""
...or, a bit more elegant:
"""

# ╔═╡ b175638b-db07-41c5-a409-f7814d14dc42
let
    # Maximization problem
    m = Model(GLPK.Optimizer)
    set_optimizer_attribute(m, MOI.Silent(), true)

    @variable(m, x[1:5], Bin)
    
    profit = [ 5, 3, 2, 7, 4 ]
    weight = [ 2, 8, 4, 2, 5 ]
    capacity = 10
    
    # Objective: maximize profit
    @objective(m, Max, dot(profit, x))
    
    # Constraint: can carry all
    @constraint(m, dot(weight, x) <= capacity)
    
    # Solve problem using MIP solver
    JuMP.optimize!(m)

	OrderedDict( i => Dict(:included => convert(Bool,JuMP.value(x[i])), :profit_over_weight => profit[i]/weight[i]) for i in 1:5 )
end

# ╔═╡ 9644fbd5-298c-4041-822b-1b44917b2c41
begin
danger(head,text) = Markdown.MD(Markdown.Admonition("danger", head, [text]));
danger(text) = Markdown.MD(Markdown.Admonition("danger", "Attention", [text]));
info(text) = Markdown.MD(Markdown.Admonition("info", "Info", [text]));
tip(text) = Markdown.MD(Markdown.Admonition("tip", "Tip", [text]));
midbreak = html"<br><br><br>";
end

# ╔═╡ 9cb9f9cc-fe89-44c3-8115-6a5c48b58ca0
midbreak

# ╔═╡ 7f033067-2216-4532-baa3-6d099bbfc034
sb = md"""
#
"""

# ╔═╡ c5f6d941-7161-44bb-8ec8-8644636234c0
sb

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GLPK = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
Ipopt = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
NLopt = "76087f3c-5699-56af-9a33-bf431cd00edd"
OrderedCollections = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
Distributions = "~0.25.86"
GLPK = "~1.1.1"
Ipopt = "~1.2.0"
JuMP = "~1.9.0"
LaTeXStrings = "~1.3.0"
NLopt = "~0.6.5"
OrderedCollections = "~1.4.1"
Plots = "~1.38.8"
PlutoUI = "~0.7.58"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "51b33323fe321a83c4c3fb15d9d6074d18c236f4"

[[deps.ASL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6252039f98492252f9e47c312c8ffda0e3b9e78d"
uuid = "ae81ac8f-d209-56e5-92de-9978fef736f9"
version = "0.1.3+0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "c278dfab760520b8bb7e9511b968bf4ba38b7acc"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

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

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

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

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

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

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "da9e1a9058f8d3eec3a8c9fe4faacfb89180066b"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.86"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

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

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "3b245d1e50466ca0c9529e2033a3c92387c59c2f"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.9"

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

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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

[[deps.GLPK]]
deps = ["GLPK_jll", "MathOptInterface"]
git-tree-sha1 = "c1ec0b87c6891a45892809e0ed0faa8d39198bab"
uuid = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
version = "1.1.1"

[[deps.GLPK_jll]]
deps = ["Artifacts", "GMP_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "fe68622f32828aa92275895fdb324a85894a5b1b"
uuid = "e8aa6df9-e6ca-548a-97ff-1f85fc5b8b98"
version = "5.0.1+0"

[[deps.GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"
version = "6.2.1+6"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "4423d87dc2d3201f3f1768a29e807ddc8cc867ef"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.71.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "3657eb348d44575cc5560c80d7e55b812ff6ffe1"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.71.8+0"

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

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Ipopt]]
deps = ["Ipopt_jll", "LinearAlgebra", "MathOptInterface", "OpenBLAS32_jll", "SnoopPrecompile"]
git-tree-sha1 = "7690de6bc4eb8d8e3119dc707b5717326c4c0536"
uuid = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
version = "1.2.0"

[[deps.Ipopt_jll]]
deps = ["ASL_jll", "Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "MUMPS_seq_jll", "OpenBLAS32_jll", "Pkg", "libblastrampoline_jll"]
git-tree-sha1 = "563b23f40f1c83f328daa308ce0cdf32b3a72dc4"
uuid = "9cc047cb-c261-5740-88fc-0cf96f7bdcc7"
version = "300.1400.403+1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "Printf", "SnoopPrecompile", "SparseArrays"]
git-tree-sha1 = "611b9f12f02c587d860c813743e6cec6264e94d8"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.9.0"

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

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.METIS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "1fd0a97409e418b78c53fac671cf4622efdf0f21"
uuid = "d00139f3-1899-568f-a2f0-47f597d42d70"
version = "5.1.2+0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MUMPS_seq_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "METIS_jll", "OpenBLAS32_jll", "Pkg", "libblastrampoline_jll"]
git-tree-sha1 = "f429d6bbe9ad015a2477077c9e89b978b8c26558"
uuid = "d7ed1dd3-d0ae-5e8e-bfb4-87a502085b8d"
version = "500.500.101+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "SnoopPrecompile", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "f219b62e601c2f2e8adb7b6c48db8a9caf381c82"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.13.1"

[[deps.MathProgBase]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9abbe463a1e9fc507f12a69e7f29346c2cdc472c"
uuid = "fdba3010-5040-5b88-9595-932c9decdf73"
version = "0.7.8"

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

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "3295d296288ab1a0a2528feb424b854418acff57"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.2.3"

[[deps.NLopt]]
deps = ["MathOptInterface", "MathProgBase", "NLopt_jll"]
git-tree-sha1 = "5a7e32c569200a8a03c3d55d286254b0321cd262"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "0.6.5"

[[deps.NLopt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9b1f15a08f9d00cdb2761dcfa6f453f5d0d6f973"
uuid = "079eb43e-fd8e-5478-9966-2cf3e3edb778"
version = "2.7.1+0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "2fb9ee2dc14d555a6df2a714b86b7125178344c2"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.21+0"

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
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "478ac6c952fddd4399e71d4779797c538d0ff2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.8"

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
git-tree-sha1 = "f49a45a239e13333b8b936120fe6d793fe58a972"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.8"

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
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "71a22244e352aa8c5f0f2adde4150f62368a3f2e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.58"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

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
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

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
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

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

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

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

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

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
git-tree-sha1 = "c6edfe154ad7b313c01aceca188c05c835c67360"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.4+0"

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
# ╟─aa1f0719-5e1d-4294-815e-349d1cebc004
# ╟─aae2e545-198c-42bb-9e97-6d6635bee02a
# ╠═9c61c9d5-0d3d-47b7-8e7d-f929cfec3311
# ╟─4468041e-4472-11ec-16b7-553f4def2b18
# ╟─e66005ad-1f51-462a-9329-d77f7695cbfc
# ╟─5bc7c8ce-3a4a-4505-a2fa-2b25f041437a
# ╟─c5f6d941-7161-44bb-8ec8-8644636234c0
# ╠═9a92426e-bc9a-4ead-a93c-ec1c3f23ca57
# ╠═d7acef71-fe0d-4062-9342-9061afde1293
# ╟─de879e63-fdc9-486e-9aaa-1a55c32d93d5
# ╟─54a17a40-fd67-4040-99b9-d0d460385155
# ╟─9f1f0926-b1b3-4ef3-aebe-277531abfd33
# ╟─2f8d4513-0871-48f3-acfb-f7a0254f7b8e
# ╟─8901f1e5-f243-4fe7-afda-6cc951b234a5
# ╟─fe6b7d74-1c6f-4253-9254-57d536bb24c5
# ╟─19d12886-3980-4116-8772-2c1d17b14270
# ╟─583a5640-adde-4f4f-a344-4e77e04e431d
# ╟─9be3cbee-7949-42da-9496-6afc94785f98
# ╟─91da671d-7602-490c-a94c-bc9ff4eca59a
# ╠═7e5cedbf-05da-4573-994f-42099fe4b33a
# ╠═b1720238-495e-47f7-ae75-866723f5fca4
# ╠═1e31dc48-e849-4045-8dcc-d05889f8609b
# ╟─70893b5f-51e8-4536-b5aa-4e353768bc52
# ╠═2a74858c-6d13-4961-86fa-125e4c01e232
# ╠═ec0c21da-c517-4d2e-9832-18eca242621b
# ╟─ade668a5-87d6-40aa-b670-d0cc41148757
# ╠═52a1ca3b-8a10-4c0a-b4d4-82f995b4b3df
# ╟─9cc92292-4531-4f4c-b9b1-2a7a698529ac
# ╠═fc75bc7b-0fe0-4f90-a425-6ed6e63acb4a
# ╠═2d7bdf41-85fc-4d0c-a387-60f9c9b36fcb
# ╟─3ba47370-c4df-4d22-8bda-0b7264927ae6
# ╠═84b81d87-d015-4cc2-8649-c94ba7770bd9
# ╟─061de138-b6fe-4437-aa26-0afaec2e66f0
# ╠═febc5df1-4ddd-48a7-94f8-251d316c72d0
# ╠═1334fc66-23bb-4a9a-9534-6f4d7a88d235
# ╟─520c3423-01d4-4794-893c-f7e5546d26f6
# ╟─ee374307-043c-473d-b561-192718ea612b
# ╟─636f7fe1-a518-4f09-bdbd-bcb79703dda5
# ╟─2fcf8444-202a-453a-af28-0eade00e3981
# ╠═3f231741-12f6-4426-8d46-b6cfe623a248
# ╟─8eef8fc6-2272-46ff-a5ab-b2a4e63d9b7a
# ╟─b1f21cfe-ebc2-4321-9636-49739568a7de
# ╠═b175638b-db07-41c5-a409-f7814d14dc42
# ╠═9cb9f9cc-fe89-44c3-8115-6a5c48b58ca0
# ╠═9644fbd5-298c-4041-822b-1b44917b2c41
# ╟─7f033067-2216-4532-baa3-6d099bbfc034
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
