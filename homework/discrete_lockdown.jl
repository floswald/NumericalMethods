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

# ╔═╡ 8c9fb16e-5bc5-11eb-14ce-039b59e408ef
using PlutoUI

# ╔═╡ 362d2fba-5bce-11eb-14da-ef225485facb
md"
# COVID19 Lockdown in a SIR Model

* Simulate a discrete SIR model
* Investigate the importance of lockdown intensity, and lockdown start and end dates
"

# ╔═╡ 8fe81ca2-5bc1-11eb-31a5-b39641ed1225
begin
	NN = 1000
	SS = NN - 1
	II = 1
	RR = 0
	TT = 500
end

# ╔═╡ a64367e0-5bc1-11eb-0ea1-7ba30e6b4d73
ss, ii, rr = SS/NN, II/NN, RR/NN

# ╔═╡ b53e57c8-5bc1-11eb-1a22-6da90e069232
β, γ = 0.1, 0.01

# ╔═╡ 13e7c542-5bc3-11eb-3bd4-877eb2050c77
@bind 🔒start Slider(1:200,show_value = true, default = 50)

# ╔═╡ 68d2e158-5bc4-11eb-2a84-35548106ce43
@bind 🔒stop  Slider(1:200,show_value = true, default = 100)

# ╔═╡ d10fe5c2-5bc4-11eb-3f5f-fb8a8220cef4
@bind 🔒intensity  Slider(0.0:0.01:1.0,show_value = true, default = 0.5)

# ╔═╡ c16fdc42-5bc1-11eb-228b-a30dd530e2fb
# 🔒intensity = [1.0 - 0.25*(it > 100) - 0.25*(it > 110) + 0.5*(it > 200) for it in 1:TT]
contact = [1.0 - 🔒intensity*(it > 🔒start) + 🔒intensity*(it > 🔒stop) for it in 1:TT];

# ╔═╡ 8040fd60-5bc0-11eb-2be1-01265f68ec1b
function discrete_lockdown(β, γ, s0, i0, r0, c, T=1000)

	s, i, r = s0, i0, r0
	
	results = [(s=s, i=i, r=r)]
	
	for t in 1:T

		Δi = c[t] * β * s * i
		Δr = γ * i
		
		s_new = s - Δi
		i_new = i + Δi - Δr
		r_new = r      + Δr

		push!(results, (s=s_new, i=i_new, r=r_new))

		s, i, r = s_new, i_new, r_new
	end
	
	return results
end

# ╔═╡ 87f504ba-5bc1-11eb-1919-73376bbf34a0
SIR = discrete_lockdown(β,γ,ss,ii,rr,contact, TT)

# ╔═╡ 78a17740-5bc2-11eb-043a-2f7d348236c4
begin
	using Plots
	ts = 1:length(SIR)
	discrete_time_SIR_plot = plot(ts, [x.s for x in SIR], 
		label="S", linecolor=:blue, leg=:right, lw=2)
	plot!(ts, [x.i for x in SIR], label="I", lw=2)
	plot!(ts, [x.r for x in SIR], label="R", lw=2)
	vline!([🔒start,🔒stop], c=:black, ls=:dash,label = "")
	title!("Lockdown in [$(🔒start),$(🔒stop)] with $(round(🔒intensity*100,digits=0))% reduction in contacts")
end

# ╔═╡ d9ba975a-5bc2-11eb-13bc-af33da4a7ef0
discrete_time_SIR_plot

# ╔═╡ Cell order:
# ╟─362d2fba-5bce-11eb-14da-ef225485facb
# ╠═8fe81ca2-5bc1-11eb-31a5-b39641ed1225
# ╠═a64367e0-5bc1-11eb-0ea1-7ba30e6b4d73
# ╠═b53e57c8-5bc1-11eb-1a22-6da90e069232
# ╠═8c9fb16e-5bc5-11eb-14ce-039b59e408ef
# ╠═13e7c542-5bc3-11eb-3bd4-877eb2050c77
# ╠═68d2e158-5bc4-11eb-2a84-35548106ce43
# ╠═d10fe5c2-5bc4-11eb-3f5f-fb8a8220cef4
# ╟─c16fdc42-5bc1-11eb-228b-a30dd530e2fb
# ╠═d9ba975a-5bc2-11eb-13bc-af33da4a7ef0
# ╠═8040fd60-5bc0-11eb-2be1-01265f68ec1b
# ╠═87f504ba-5bc1-11eb-1919-73376bbf34a0
# ╠═78a17740-5bc2-11eb-043a-2f7d348236c4
