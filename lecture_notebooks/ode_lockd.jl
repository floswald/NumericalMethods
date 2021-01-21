### A Pluto.jl notebook ###
# v0.12.18

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

# ╔═╡ cddc1c42-5b68-11eb-3fc9-1fa3fd94301b
begin
	using DifferentialEquations
	using Plots
end

# ╔═╡ 0022974a-5b72-11eb-2bb5-bf4ba59f0d96
using PlutoUI

# ╔═╡ 703041b2-5b67-11eb-1bd7-73a51e463182
md"
# Lockdown in the Continuous SIR Model

One widely used policy in the COVID19 pandemic was/is some flavour of *lockdown*. By this we mean to restrict mobility of people, which in turn reduces the ability of the illness to spread. In our simple SIR model this is easy to implement. Remember:


$$\begin{align}
\textstyle \frac{ds(t)}{dt} &= -\beta c\, s(t) \, i(t) \\
\textstyle \frac{di(t)}{dt} &= +\beta c\, s(t) \, i(t) &- \gamma \, i(t)\\
\textstyle \frac{dr(t)}{dt} &= &+ \gamma \, i(t)
\end{align}$$

Parameter $\beta$ is the *rate of instantaneous* infection of a susceptible person (the probability that in any instant $dt$ a $S$ turns into an $I$) - upon meeting an infectuous person. We denote $c$ the *contact rate*, i.e. how often an $I$ and a $S$ person meet. 
"

# ╔═╡ 4ca2f772-5b68-11eb-2ec2-057460859da5
md"
#

Well, lockdown could just be a time varying $c$: given our restrictions on mobility at time $t_1$, we are able to reduce the contact rate $c_0$ to $c_1$, say. So, we'd write

$$\begin{align}
\textstyle \frac{ds(t)}{dt} &= -\beta c(t) \, s(t) \, i(t) \\
\textstyle \frac{di(t)}{dt} &= +\beta c(t) \, s(t) \, i(t) &- \gamma \, i(t)\\
\textstyle \frac{dr(t)}{dt} &= &+ \gamma \, i(t)
\end{align}$$
"

# ╔═╡ 9f2cffd8-5b68-11eb-09b9-c745fdd041d1
md"#

and our setup from the plain-vanilla SIR will have to be able to change the values of c$ at certain times.

👉 let's do it!
"

# ╔═╡ d9570924-5b68-11eb-2d5f-453733ec6b48
function sir_ode!(du,u,p,t)
    (S,I,R) = u
    (β,c,γ) = p
    N = S+I+R
    @inbounds begin
        du[1] = -β*c*I/N*S
        du[2] = β*c*I/N*S - γ*I
        du[3] = γ*I
    end
    nothing
end

# ╔═╡ e51d7662-5b68-11eb-3890-d7b7df2373d7
begin 
	δt = 0.1
	tmax = 80.0
	tspan = (0.0,tmax)
	t = 0.0:δt:tmax;
end

# ╔═╡ 63637f08-5b69-11eb-0b56-db3c401c0239


# ╔═╡ f1e718d0-5b68-11eb-3bad-393d4a8b5c17
u0 = [990.0,10.0,0.0]; # S,I,R

# ╔═╡ f880399c-5b68-11eb-117b-353b47e58a3a
p = [0.05,10.0,0.25]; # β,c,γ

# ╔═╡ 498ebebe-5b69-11eb-3bb1-654bb20de8fc
md"
So far this is like last time. Let's run that model again:
"

# ╔═╡ 5367c546-5b69-11eb-0220-0757efff7cbf
sir_ode = ODEProblem(sir_ode!,u0,tspan,p)

# ╔═╡ 6a163372-5b69-11eb-03d5-45803d803cb4
sir_sol = solve(sir_ode);

# ╔═╡ 7e59d0f8-5b69-11eb-3095-5b372f4a362e
plot(sir_sol, label = ["S" "I" "R"], title = "ODE SIR model")

# ╔═╡ 87ab3a8e-5b69-11eb-302b-f1e903772c59
md"

## Lockdown

Time to introduce lockdown! Here's how. We will supply a [*callback* function](https://diffeq.sciml.ai/dev/features/callback_functions/) to the ODE solver. We need to define how we would like to change the state of the solution at certain times:
"

# ╔═╡ 26efc848-5b72-11eb-0e39-4d0054bc73de
@bind 🔒 Slider(1.0:tmax,default = 10, show_value = true)

# ╔═╡ 2a87eb0e-5b72-11eb-30f7-cbea8e78eaa1
@bind 🔓 Slider(1.0:tmax,default = 20, show_value = true)

# ╔═╡ bf9a16aa-5b6a-11eb-02ef-f1da8057d138
lockdown_times = [🔒, 🔓];

# ╔═╡ a8da7072-5b6a-11eb-0030-0d8fd8512e2a
function affect!(integrator) 
    if integrator.t < lockdown_times[2] 
        integrator.p[2] = 5.0
    else
        integrator.p[2] = 10.0
    end
end

# ╔═╡ 7b01e2d4-5b6a-11eb-3e12-39bb7d906984
md"
next, we need to define the times when we want to *affect* the system with a `condition` function - whenever that returns `true`, the `affect!` function gets called:
"

# ╔═╡ f50dab9e-5b6a-11eb-0ed6-a39c070faf5f
condition(u,t,integrator) = t ∈ lockdown_times   # make change at lockdown times

# ╔═╡ 00de7f52-5b6b-11eb-3280-3506b1a20d4e
cb = PresetTimeCallback(lockdown_times, affect!)  # finally set up the callback

# ╔═╡ 09d1a706-5b6b-11eb-01cf-2fd492fb0ccc
lockdown_sol = solve(sir_ode, callback = cb);

# ╔═╡ 27213f1c-5b6b-11eb-0013-fbf18c3c5479
begin
	plot(lockdown_sol, label = ["S" "I" "R"], 
		 title = "Lockdown from day $(🔒) to $(🔓) in SIR model",
	     w = 2)
	vline!(lockdown_times, c = :black, ls = :dash, w = 2, label = "")
end

# ╔═╡ Cell order:
# ╠═703041b2-5b67-11eb-1bd7-73a51e463182
# ╠═4ca2f772-5b68-11eb-2ec2-057460859da5
# ╠═9f2cffd8-5b68-11eb-09b9-c745fdd041d1
# ╠═cddc1c42-5b68-11eb-3fc9-1fa3fd94301b
# ╠═d9570924-5b68-11eb-2d5f-453733ec6b48
# ╠═e51d7662-5b68-11eb-3890-d7b7df2373d7
# ╠═63637f08-5b69-11eb-0b56-db3c401c0239
# ╠═f1e718d0-5b68-11eb-3bad-393d4a8b5c17
# ╠═f880399c-5b68-11eb-117b-353b47e58a3a
# ╟─498ebebe-5b69-11eb-3bb1-654bb20de8fc
# ╠═5367c546-5b69-11eb-0220-0757efff7cbf
# ╠═6a163372-5b69-11eb-03d5-45803d803cb4
# ╠═7e59d0f8-5b69-11eb-3095-5b372f4a362e
# ╟─87ab3a8e-5b69-11eb-302b-f1e903772c59
# ╠═0022974a-5b72-11eb-2bb5-bf4ba59f0d96
# ╟─26efc848-5b72-11eb-0e39-4d0054bc73de
# ╟─2a87eb0e-5b72-11eb-30f7-cbea8e78eaa1
# ╟─bf9a16aa-5b6a-11eb-02ef-f1da8057d138
# ╠═27213f1c-5b6b-11eb-0013-fbf18c3c5479
# ╟─a8da7072-5b6a-11eb-0030-0d8fd8512e2a
# ╟─7b01e2d4-5b6a-11eb-3e12-39bb7d906984
# ╟─f50dab9e-5b6a-11eb-0ed6-a39c070faf5f
# ╟─00de7f52-5b6b-11eb-3280-3506b1a20d4e
# ╠═09d1a706-5b6b-11eb-01cf-2fd492fb0ccc
