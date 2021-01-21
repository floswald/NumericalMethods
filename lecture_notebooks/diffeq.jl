### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 88e9b04a-5b31-11eb-3a16-6d89211bf227
begin
	using DifferentialEquations
	using Plots
	using PlutoUI
end

# ╔═╡ f467c124-5b31-11eb-0302-450f032789c1
md"
# Intro to DifferentialEquations.jl

This notebook follows closely the material [here](https://tutorials.sciml.ai/html/introduction/01-ode_introduction.html)

We have seen in the SIR notebook that an ordinary differential equation (ODE) looks in general somewhat like

$$u(t) = f(u,p,t)$$
"


# ╔═╡ 3b926010-5b33-11eb-07e4-bb9ee7167f7a
md"
with

* the variable of interest: $u$ 
* a nonlinear model of how we think it changes over time: $f$ 
* a set of parameters: $p$ 
* time $t$ 

Given an initial value $u(0) = u_0$, we can use the above model to predict $u(t)$ at each point in time.
"

# ╔═╡ c25bdb3e-5b32-11eb-1db9-09d03e2e62ca
md"
## 

* We have also seen that only the simplest ODEs can actually be solved by hand.
* Our simple SIR model (system of 3 ODEs) does not admit an analytic solution.
* Let's see how to use the `DifferentialEquations.jl` package to solve ODEs.
"

# ╔═╡ 5f5415b6-5b33-11eb-3772-61e92906e87f
md"
## Exponential Growth

* Remember again from the SIR notebook, we got this ODE:
$$\frac{dI(t)}{dt} = -\lambda \, I(t)$$

* or
$$I' = -\lambda \, I$$

Let's say $\lambda = 0.05$ i.e. the infected decrease by a rate of 5% in each time step. In `DifferentialEquations`:
"

# ╔═╡ 1e7d3210-5b34-11eb-3400-ede033573e9b
f(I,p,t) = -0.05I

# ╔═╡ 377448f8-5b34-11eb-05f4-cf92609b1764
md"
Let's say with start with $I(0) = 100$ infected people. We'd set
"

# ╔═╡ 49698884-5b34-11eb-3989-b19fbbd1a945
begin
	I0 = 100.0
	timespan = (0.0,100.0)
	problem = ODEProblem(f,I0,timespan)
end

# ╔═╡ 6cf56d2c-5b34-11eb-0dd7-07308273248f
solution = solve(problem);

# ╔═╡ 75d3e45a-5b34-11eb-2908-dfc019e3105f
md"
## Looking at Solutions

* There is a [dedicated manual page](https://diffeq.sciml.ai/dev/basics/solution/)
* Let's plot the time path of $I(t)$!
"

# ╔═╡ b7923bfa-5b34-11eb-23ed-6db7fad7de6b
plot(solution, label = "rate of decay 5%") 

# ╔═╡ c3d5d1e2-5b34-11eb-0e7f-d1bd22bd5626
md"
## Systems of ODEs: The Lorenz Attractor

* The [Lorenz equation](https://en.wikipedia.org/wiki/Lorenz_system) is a famous example from chaos theory.
* It's defined as a system of 3 ODEs:


$$\begin{align}
\frac{dx}{dt} &= σ(y-x) \\
\frac{dy}{dt} &= x(ρ-z) - y \\
\frac{dz}{dt} &= xy - βz \\
\end{align}$$

In `DifferentialEquations.jl` we just have to define our `f` accordingly to take a vector as input:
"

# ╔═╡ f0dcc160-5b36-11eb-363e-1f6688533764
function lorenz!(du,u,p,t)  # modify du in-place
	σ,ρ,β = p
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
end

# ╔═╡ d81ee810-5b36-11eb-0fbf-93ed0cf4466d
md"
let's start at initial condition $u_0 = (1,0,0)$ and set parameters $(\sigma,\rho,\beta) = (10,28,8/3)$
"

# ╔═╡ 3a29cebc-5b37-11eb-0a0c-cfe7ec83caa7
begin
	u0 = [1.0,0.0,0.0]
	p = (σ = 10, ρ = 28, β = 8/3)
	tspan = (0.0,100.0)
	prob = ODEProblem(lorenz!, u0, tspan, p)
end

# ╔═╡ 65a89438-5b37-11eb-3d9a-91f15f1cc87e
sol = solve(prob);

# ╔═╡ 9a0226cc-5b37-11eb-0e3e-9de93b5479ae
md"now `sol` has the solution paths for all 3 equations, stored at each time step. We can conveniently plot those paths:
"

# ╔═╡ adeb9308-5b37-11eb-1acc-6957376d8dbd
plot(sol)

# ╔═╡ b02cf2c2-5b37-11eb-29b8-a5f7105d5154
md"
and we can also plot their values against each other ,instead of over time, by setting a keyword argument on the `plot` method."

# ╔═╡ d49abf42-5b37-11eb-1581-3b7eaf72552e
plot(sol, vars = (1,2,3) )

# ╔═╡ e42e9a6e-5b37-11eb-2d3a-69681981e8c1
md"
The docs of the Plots.jl package have a very cool animation of this object, which we can easily reproduce here.
"

# ╔═╡ 36ecad18-5b38-11eb-0fc9-3593cb2f5084
begin
	Base.@kwdef mutable struct Lorenz
		dt::Float64 = 0.02
		σ::Float64 = 10
		ρ::Float64 = 28
		β::Float64 = 8/3
		x::Float64 = 1
		y::Float64 = 1
		z::Float64 = 1
	end

	function step!(l::Lorenz)
		dx = l.σ * (l.y - l.x);         l.x += l.dt * dx
		dy = l.x * (l.ρ - l.z) - l.y;   l.y += l.dt * dy
		dz = l.x * l.y - l.β * l.z;     l.z += l.dt * dz
	end

	attractor = Lorenz()


	# initialize a 3D plot with 1 empty series
	plt = plot3d(
		1,
		xlim = (-30, 30),
		ylim = (-30, 30),
		zlim = (0, 60),
		title = "Lorenz Attractor",
		marker = 2,
	)

	# build an animated gif by pushing new points to the plot, saving every 10th frame
	@gif for i=1:1500
		step!(attractor)
		push!(plt, attractor.x, attractor.y, attractor.z)
	end every 10
end

# ╔═╡ 480d27da-5b38-11eb-2e7b-7ff4c9faf028


# ╔═╡ Cell order:
# ╟─f467c124-5b31-11eb-0302-450f032789c1
# ╟─3b926010-5b33-11eb-07e4-bb9ee7167f7a
# ╠═c25bdb3e-5b32-11eb-1db9-09d03e2e62ca
# ╟─5f5415b6-5b33-11eb-3772-61e92906e87f
# ╠═88e9b04a-5b31-11eb-3a16-6d89211bf227
# ╠═1e7d3210-5b34-11eb-3400-ede033573e9b
# ╟─377448f8-5b34-11eb-05f4-cf92609b1764
# ╠═49698884-5b34-11eb-3989-b19fbbd1a945
# ╠═6cf56d2c-5b34-11eb-0dd7-07308273248f
# ╟─75d3e45a-5b34-11eb-2908-dfc019e3105f
# ╠═b7923bfa-5b34-11eb-23ed-6db7fad7de6b
# ╟─c3d5d1e2-5b34-11eb-0e7f-d1bd22bd5626
# ╠═f0dcc160-5b36-11eb-363e-1f6688533764
# ╠═d81ee810-5b36-11eb-0fbf-93ed0cf4466d
# ╠═3a29cebc-5b37-11eb-0a0c-cfe7ec83caa7
# ╠═65a89438-5b37-11eb-3d9a-91f15f1cc87e
# ╟─9a0226cc-5b37-11eb-0e3e-9de93b5479ae
# ╟─adeb9308-5b37-11eb-1acc-6957376d8dbd
# ╟─b02cf2c2-5b37-11eb-29b8-a5f7105d5154
# ╟─d49abf42-5b37-11eb-1581-3b7eaf72552e
# ╟─e42e9a6e-5b37-11eb-2d3a-69681981e8c1
# ╠═36ecad18-5b38-11eb-0fc9-3593cb2f5084
# ╠═480d27da-5b38-11eb-2e7b-7ff4c9faf028
