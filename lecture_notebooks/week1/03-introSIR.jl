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

# ╔═╡ 9a0cec14-08db-11eb-3cfa-4d1c327c63f1
begin
	using Plots
	using PlutoUI
	using Statistics
end

# ╔═╡ 76b23770-66fe-11eb-0d2c-732ee0b333cc
md"
# ScPo Numerical Methods Week 1

Florian Oswald, 2021
"

# ╔═╡ d6260e60-66fa-11eb-367b-d3b1d324cdf6
html"<button onclick='present()'>present</button>"

# ╔═╡ a3b2accc-0845-11eb-229a-e97bc3943016
md"""
## Deriving a SIR epidemiological model

In this notebook we will go from a discrete epidemiological model to its continuous  counterpart. We will go from a (discrete) system of difference equiations to a system of (continuous) differential equations. But first: *SIR*.

### Susceptible, Infected, Recovered (SIR)

In this workhorse of epidemiology we model agents as belonging to a certain class, or *compartment*. Either they are 

* *susceptible* to get infected, 
* *infected*, or
* *removed/recovered*

The models you may have heard about during the COVID19 pandemic are more sophisticated versions of what we will see here, but they share some important common features. At the end of this notebook, we will want to generate the following plot. It shows the number of agents belonging to each class at a certain point in time after the outbreak of the disease:

#
"""

# ╔═╡ f9a75ac4-08d9-11eb-3167-011eb698a32c
md"""
#

We will make this very simple. Instead of trying to model in great detail which type of agent (old, young, where, what time of the day, doing which activity etc) is likely to get infected, we will try to condense the overall probability of infection into a single number.
Let's start to model the recovery from an infection $I \to R$:

> we have $N$ people infected at time $0$. If each has probability $p$ to recover each day, how many are still infected at day number $t$?

For any given person in $I$ state, the probability to recover in a certain time step (i.e. each day) is a

#
"""

# ╔═╡ ba7ffe78-0845-11eb-2847-851a407dd2ec
bernoulli(p) = rand() < p 

# ╔═╡ e9a55500-6794-11eb-2d94-d9d26bbec45a
bernoulli2(p) = p > 0.7 ? rand() : 0.9  # ternary expression

# ╔═╡ d088ed2e-0845-11eb-0697-310f374effbc
N = 200

# ╔═╡ e2d764d0-0845-11eb-0031-e74d2f5acaf9
function step!(infectious, p)
	for i in 1:length(infectious)
		
		if infectious[i] && bernoulli(p)
			infectious[i] = false
		end
	end
	
	return infectious
end

# ╔═╡ 428fe6b6-66fb-11eb-370c-c1e0dfd1a702
md"
#
"

# ╔═╡ 9282eca0-08db-11eb-2e36-d761594b427c
T = 100

# ╔═╡ 58d8542c-08db-11eb-193a-398ce01b8635
# try this out: all are infected on day one.
# let's record how many are in each state I or R after i days
begin
	infected = trues(N)  # all infected
		
	results = [copy(step!(infected, 0.05)) for i in 1:T]
	# we `copy` here because we keep modifying the same array throughout
	pushfirst!(results, trues(N))
end

# ╔═╡ 8d6c0c06-08db-11eb-3790-c98fdc545352
@bind i Slider(1:T, show_value=true)

# ╔═╡ 7e1b61ac-08db-11eb-209e-1d6c328f5113
begin
	scatter(results[i], 
		alpha=0.5, size=(300, 200), leg=false, c=Int.(results[i]))
	
	annotate!(N, 0.9, text("$(count(results[i]))", 10))
	annotate!(N, 0.1, text("$(N - count(results[i]))", 10))
	
	ylims!(-0.1, 1.1)
	
	xlabel!("i")
	ylabel!("X_i(t)")

end

# ╔═╡ d57e1b8c-5b25-11eb-059d-279b5de0894e
md"
#

ok great, now we know how many are $I$ or $R$ after $t$ days with a certain probability $p$. Let's just condense this plot above and `count` how many infected there are each day! Here's how:"

# ╔═╡ 33f9fc36-0846-11eb-18c2-77f92fca3176
function simulate_recovery(p, T)
	infectious = trues(N)
	num_infectious = [N]
	
	for t in 1:T
		step!(infectious, p)
		push!(num_infectious, count(infectious))
	end
	
	return num_infectious
end

# ╔═╡ 704b68dc-66fb-11eb-3112-2ff0bdca286f
md"#"

# ╔═╡ cb278624-08dd-11eb-3375-276bfe8d7b3a
begin
	pp = 0.05
	
	plot(simulate_recovery(pp, T), label="run 1", alpha=0.5, lw=2, m=:o)
	plot!(simulate_recovery(pp, T), label="run 2", alpha=0.5, lw=2, m=:o)
	
	xlabel!("time t")
	ylabel!("number infectious")
end

# ╔═╡ 7c040b20-66fb-11eb-37c8-470a35574d7e
md"#"

# ╔═╡ f3c85814-0846-11eb-1266-63f31f351a51
all_data = [simulate_recovery(pp, T) for i in 1:30];  # do 30 runs

# ╔═╡ 01dbe272-0847-11eb-1331-4360a575ff14
begin
	plot(all_data, alpha=0.1, leg=false, m=:o, ms=1,
		size=(500, 400), label="")
	xlabel!("time t")
	ylabel!("number infectious")
end

# ╔═╡ 82cefdac-66fb-11eb-0429-7d186f270dc3
md"#"

# ╔═╡ 85057772-66fb-11eb-1e55-cf145fa97178
md"#"

# ╔═╡ caa3faa2-08e5-11eb-33fe-cbbc00cfd459
md"""
## Deterministic dynamics for the mean: Intuitive derivation

The mean seems to behave in a rather predictable way over time. Can we derive this?

Let $I_t$ be the number of infectious people at time $t$. This decreases because some people recover. Since people recover with probability $p$, the number of people that recover at time $t$ is, on average, $p I_t$. [Note that one time unit corresponds to one *sweep* of the simulation.]

So

$$I_{t+1} = I_t - p \, I_t$$

or

$$I_{t+1} = I_t (1 - p).$$

"""

# ╔═╡ 7e89f992-0847-11eb-3155-c5313575f767
md"""
#

At time $t$ there are $I_t$ infectious.
How many decay? Each decays with probability $p$, so on average $p I_t$ recover, so are removed from the number of infectious, giving the change

$$\Delta I_t = I_{t+1} - I_t = -p \, I_t$$
"""

# ╔═╡ f5756dd6-0847-11eb-0870-fd06ad10b6c7
md"""
#
We can rearrange and solve the recurrence relation:

$$I_{t+1} = (1 - p) I_t. $$

so

$$I_{t+1} = (1 - p) (1 - p) I_{t-1} = (1 - p)^2 I_{t-1}$$


and hence solve the recurrence relation:

$$I_t = (1-p)^t \, I_0.$$

Let's compare the exact and numerical results:

#
"""

# ╔═╡ 6a545268-0846-11eb-3861-c3d5f52c061b
exact = [N * (1-pp)^t for t in 0:T]

# ╔═╡ 4c8827b8-0847-11eb-0fd1-cfbdbdcf392e
begin
	plot(mean(all_data), m=:o, label="numerical mean")
	plot!(exact, lw=3, label="analytical mean")
end
	

# ╔═╡ 2f980870-0848-11eb-3edb-0d4cd1ed5b3d
md"""
## Continuous time

If we look at the graph of the mean as a function of time, it seems to follow a smooth curve. Indeed it makes sense to ask not only how many people have recovered each *day*, but to aim for finer granularity.

Suppose we instead increment time in steps of $\delta t$; the above analysis was for $\delta t = 1$.

Then we will need to adjust the probability of recovery in each time step. 
It turns out that to make sense in the limit $\delta t \to 0$, we need to choose the probability $p(\delta t)$ to recover in time $t$ to be proportional to $\delta t$:

$$p(\delta t) \simeq \lambda \, \delta t,$$

where $\lambda$ is the recovery **rate**. Note that a rate is a probability *per unit time*.

##

We get

$$I(t + \delta t) - I(t) \simeq -\lambda \,\delta t \, I(t)$$
"""

# ╔═╡ 6af30142-08b4-11eb-3759-4d2505faf5a0
md"""
$$I(t + \delta t) - I(t) \simeq -\lambda \,\delta t \, I(t)$$

Dividing by $\delta t$ gives

$$\frac{I(t + \delta t) - I(t)}{\delta t} \simeq -\lambda \, I(t)$$

We recognise the left-hand side as the definition of the **derivative** when $\delta t \to 0$. Taking that limit finally gives

$$\frac{dI(t)}{dt} = -\lambda \, I(t)$$

##

$$\frac{dI(t)}{dt} = -\lambda \, I(t)$$

We recognise the left-hand side as the definition of the **derivative** when $\delta t \to 0$. Taking that limit finally gives

$$\frac{dI(t)}{dt} = -\lambda \, I(t)$$

That is, we obtain an **ordinary differential equation** that gives the solution implicitly. We know this has a general solution of the form

$$I(t) = C \exp(-\lambda t),$$

and together with an initial value of $I(0) = I_0$ we get (just set $t=0$ and use the given value)

#

$$I(t) = I_0 \exp(-\lambda \, t).$$

Starting at time zero ($t=0$) and at a number of $I_0$ infected, this is *exponential decay at rate $\lambda$*. Of course with a positive number (i.e. no minus sign, as here), we get *exponential growth*.

"""

# ╔═╡ d74bace6-08f4-11eb-2a6b-891e52952f57
md"""
## SIR model

Now let's extend the procedure to the full SIR model, $S \to I \to R$. Since we already know how to deal with recovery, consider just the SI model, where susceptible agents are infected via contact, with a certain probability.
"""

# ╔═╡ 238f0716-0903-11eb-1595-df71600f5de7
md"""
## Setup

Let's denote by $S_t$ and $I_t$ be the number of susceptible and infectious people at time $t$, respectively, and by $N$ the total number of people.

On average, in each sweep each infectious individual has the chance to interact with one other individual. That individual is chosen uniformly at random from the total population of size $N$. But a new infection occurs *only if that chosen individual is susceptible*, which happens with probability $S_t / N$. Then, upon meeting a susceptible person, the infection is transmitted only with a certain probability, $b$ say.

Hence the change in the number of infectious people after that step is:


$$\Delta I_t = I_{t+1} - I_t = b \, I_t \, \left(\frac{S_t}{N} \right)$$

It is useful to normalize by $N$, so we define

$$s_t := \frac{S_t}{N}; \quad i_t := \frac{I_t}{N}; \quad r_t := \frac{R_t}{N}$$

#

Including recovery with probability $g$ we obtain the **discrete-time SIR model**:

$$\begin{align}
s_{t+1} &= s_t - b \, s_t \, i_t \\
i_{t+1} &= i_t + b \, s_t \, i_t - g \, i_t\\
r_{t+1} &= r_t + c \, i_t
\end{align}$$
"""

# ╔═╡ 4e3c7e62-090d-11eb-3d16-e921405a6b16
md"""
#

And again we can allow the processes to occur in steps of length $\delta t$ and take the limit $\delta t \to 0$. With rates $\beta$ and $\gamma$ we obtain the standard (continuous-time) **SIR model**:

$$\begin{align}
\textstyle \frac{ds(t)}{dt} &= -\beta \, s(t) \, i(t) \\
\textstyle \frac{di(t)}{dt} &= +\beta \, s(t) \, i(t) &- \gamma \, i(t)\\
\textstyle \frac{dr(t)}{dt} &= &+ \gamma \, i(t)
\end{align}$$

Note that no analytical solutions of these (simple) nonlinear ODEs are known as a function of time!
"""

# ╔═╡ f8a28ba0-0915-11eb-12d1-336f291e1d84
md"""
##

Below is a simulation of the discrete-time model. Note that the simplest numerical method to solve (approximately) the system of ODEs, the **Euler method**, basically reduces to solving the discrete-time model!  A whole suite of more advanced ODE solvers is provided in the [Julia `DiffEq` ecosystem](https://diffeq.sciml.ai/dev/). We will in another notebook introduce that package, together with a *continuous* version of the SIR model.
"""

# ╔═╡ d994e972-090d-11eb-1b77-6d5ddb5daeab
begin
	NN = 100
	
	SS = NN - 1
	II = 1
	RR = 0
end

# ╔═╡ 050bffbc-0915-11eb-2925-ad11b3f67030
ss, ii, rr = SS/NN, II/NN, RR/NN

# ╔═╡ 1d0baf98-0915-11eb-2f1e-8176d14c06ad
p_infection, p_recovery = 0.1, 0.01

# ╔═╡ 349eb1b6-0915-11eb-36e3-1b9459c38a95
function discrete_SIR(s0, i0, r0, T=1000)

	s, i, r = s0, i0, r0
	
	results = [(s=s, i=i, r=r)]
	
	for t in 1:T

		Δi = p_infection * s * i
		Δr = p_recovery * i
		
		s_new = s - Δi
		i_new = i + Δi - Δr
		r_new = r      + Δr

		push!(results, (s=s_new, i=i_new, r=r_new))

		s, i, r = s_new, i_new, r_new
	end
	
	return results
end

# ╔═╡ 39c24ef0-0915-11eb-1a0e-c56f7dd01235
SIR = discrete_SIR(ss, ii, rr)

# ╔═╡ 442035a6-0915-11eb-21de-e11cf950f230
begin
	ts = 1:length(SIR)
	discrete_time_SIR_plot = plot(ts, [x.s for x in SIR], 
		m=:o, label="S", alpha=0.2, linecolor=:blue, leg=:right, size=(400, 300))
	plot!(ts, [x.i for x in SIR], m=:o, label="I", alpha=0.2)
	plot!(ts, [x.r for x in SIR], m=:o, label="R", alpha=0.2)
	
	xlims!(0, 500)
end

# ╔═╡ 5f4516fe-098c-11eb-3abe-418aac994cc3
discrete_time_SIR_plot

# ╔═╡ b54787d8-66fe-11eb-1efe-cdf92fe2148f
md"function library"

# ╔═╡ a480df46-66fc-11eb-1e49-4bd078962d93
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

# ╔═╡ be8e4ac2-08dd-11eb-2f72-a9da5a750d32
TwoColumn(
md"""
Here we add to the previous plot with `plot!`:

```julia
plot!(
	mean(all_data), 
	leg=true, 
	label="mean",
	lw=3, 
	c=:red, 
	m=:o, 
	alpha=0.5)
```
	
""",
	plot!(mean(all_data), leg=true, label="mean",
		lw=3, c=:red, m=:o, alpha=0.5, 
		size=(500, 400)))

# ╔═╡ 8bc52d58-0848-11eb-3487-ef0d06061042
TwoColumn(
md"""
```julia
begin
plot(replace.(all_data, 0.0 => NaN), 
	yscale=:log10, alpha=0.3, leg=false, m=:o, ms=1,
	size=(500, 400))

plot!(mean(all_data), yscale=:log10, lw=3, c=:red, m=:o, label="mean", alpha=0.5)

xlabel!("time t")
ylabel!("number infectious")
end)
```
""",
	begin
	plot(replace.(all_data, 0.0 => NaN), 
		yscale=:log10, alpha=0.3, leg=false, m=:o, ms=1,
		size=(500, 400))
	
	plot!(mean(all_data), yscale=:log10, lw=3, c=:red, m=:o, label="mean", alpha=0.5)
	
	xlabel!("time t")
	ylabel!("number infectious")
	end)

# ╔═╡ Cell order:
# ╟─76b23770-66fe-11eb-0d2c-732ee0b333cc
# ╠═d6260e60-66fa-11eb-367b-d3b1d324cdf6
# ╟─9a0cec14-08db-11eb-3cfa-4d1c327c63f1
# ╟─a3b2accc-0845-11eb-229a-e97bc3943016
# ╠═5f4516fe-098c-11eb-3abe-418aac994cc3
# ╟─f9a75ac4-08d9-11eb-3167-011eb698a32c
# ╠═ba7ffe78-0845-11eb-2847-851a407dd2ec
# ╠═e9a55500-6794-11eb-2d94-d9d26bbec45a
# ╠═d088ed2e-0845-11eb-0697-310f374effbc
# ╠═e2d764d0-0845-11eb-0031-e74d2f5acaf9
# ╟─428fe6b6-66fb-11eb-370c-c1e0dfd1a702
# ╠═9282eca0-08db-11eb-2e36-d761594b427c
# ╠═58d8542c-08db-11eb-193a-398ce01b8635
# ╠═8d6c0c06-08db-11eb-3790-c98fdc545352
# ╠═7e1b61ac-08db-11eb-209e-1d6c328f5113
# ╟─d57e1b8c-5b25-11eb-059d-279b5de0894e
# ╠═33f9fc36-0846-11eb-18c2-77f92fca3176
# ╟─704b68dc-66fb-11eb-3112-2ff0bdca286f
# ╠═cb278624-08dd-11eb-3375-276bfe8d7b3a
# ╟─7c040b20-66fb-11eb-37c8-470a35574d7e
# ╠═f3c85814-0846-11eb-1266-63f31f351a51
# ╠═01dbe272-0847-11eb-1331-4360a575ff14
# ╟─82cefdac-66fb-11eb-0429-7d186f270dc3
# ╟─be8e4ac2-08dd-11eb-2f72-a9da5a750d32
# ╟─85057772-66fb-11eb-1e55-cf145fa97178
# ╟─8bc52d58-0848-11eb-3487-ef0d06061042
# ╟─caa3faa2-08e5-11eb-33fe-cbbc00cfd459
# ╟─7e89f992-0847-11eb-3155-c5313575f767
# ╟─f5756dd6-0847-11eb-0870-fd06ad10b6c7
# ╟─6a545268-0846-11eb-3861-c3d5f52c061b
# ╠═4c8827b8-0847-11eb-0fd1-cfbdbdcf392e
# ╟─2f980870-0848-11eb-3edb-0d4cd1ed5b3d
# ╟─6af30142-08b4-11eb-3759-4d2505faf5a0
# ╟─d74bace6-08f4-11eb-2a6b-891e52952f57
# ╟─238f0716-0903-11eb-1595-df71600f5de7
# ╟─4e3c7e62-090d-11eb-3d16-e921405a6b16
# ╟─f8a28ba0-0915-11eb-12d1-336f291e1d84
# ╠═442035a6-0915-11eb-21de-e11cf950f230
# ╠═d994e972-090d-11eb-1b77-6d5ddb5daeab
# ╠═050bffbc-0915-11eb-2925-ad11b3f67030
# ╠═1d0baf98-0915-11eb-2f1e-8176d14c06ad
# ╠═349eb1b6-0915-11eb-36e3-1b9459c38a95
# ╠═39c24ef0-0915-11eb-1a0e-c56f7dd01235
# ╟─b54787d8-66fe-11eb-1efe-cdf92fe2148f
# ╠═a480df46-66fc-11eb-1e49-4bd078962d93
