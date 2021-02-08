### A Pluto.jl notebook ###
# v0.12.20

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

# ╔═╡ d3241b7a-6a43-11eb-13b0-5d35d22508f1
begin
	using Plots
	using PlutoUI
end

# ╔═╡ cbf5f6c0-616f-11eb-2bfc-97f62c8694d2
# setting up an empty package environment
begin
	import Pkg
	Pkg.activate(mktempdir())
end

# ╔═╡ 8c9fb16e-5bc5-11eb-14ce-039b59e408ef
begin
	Pkg.add("Plots")
	Pkg.add("PlutoUI")
end

# ╔═╡ 57005404-6170-11eb-3ffc-f57feda21784
# edit the code below to set your name and email

student = (name = "Jazzy Jeff", email = "jazzy.jeff@yahoo.com")

# press the ▶ button in the bottom right of this cell to run your edits
# or use Shift+Enter

# ╔═╡ 14d5c6aa-670a-11eb-3cd3-9dd24111525b
md"Homework 1 submitted by $(student.name)"

# ╔═╡ 6239ebaa-6170-11eb-2217-c1937ab254bc
begin
	if (student.name == "Jazzy Jeff") 
		Markdown.MD(Markdown.Admonition("danger", "Oops!", [md"You are not *really* called **Jazzy Jeff**. Please fill out name!"]))
	elseif !(contains(student.email, "@"))
		Markdown.MD(Markdown.Admonition("danger", "Oops!", [md"Please enter a valid email address.  "]))
	elseif (student.email == "jazzy.jeff@yahoo.com")
		Markdown.MD(Markdown.Admonition("danger", "Oops!", [md"Please enter a valid email in the `student` tuple above.  "]))
	end
end

# ╔═╡ 362d2fba-5bce-11eb-14da-ef225485facb
md"
# COVID19 Lockdown in a SIR Model

In this homework we want to make a small modification to the discrete SIR model introduced in [this notebook](https://github.com/floswald/NumericalMethods/blob/master/lecture_notebooks/week1/03-introSIR.jl) (in lecture 1).

Remember that the discrete SIR model is defined as


$$\begin{align}
s_{t+1} &= s_t - \beta \, s_t \, i_t \\
i_{t+1} &= i_t + \beta \, s_t \, i_t - \gamma \, i_t\\
r_{t+1} &= r_t + \gamma \, i_t
\end{align}$$

where he had normalized each *compartment* of people by the total population $N$:

$$s_t := \frac{S_t}{N}; \quad i_t := \frac{I_t}{N}; \quad r_t := \frac{R_t}{N}$$
"

# ╔═╡ 1d791a08-6171-11eb-3857-8da3d47c397e
md" ## Setup

* Start by defining the parameters. Let's have 1000 individuals, 
* let's start with a single infected person
* Nobody is recovered yet
* We will runs this for 500 periods
"

# ╔═╡ 8fe81ca2-5bc1-11eb-31a5-b39641ed1225
begin
	NN = missing
	SS = NN - 1
	II = 1
	RR = 0
	TT = 500
end

# ╔═╡ 1fdf8918-6172-11eb-1719-63a5b105ed7d
if ismissing(NN)
md"""
!!! warning
    variable `NN` needs to be defined!
"""
end	

# ╔═╡ 67018b82-6172-11eb-37a5-77d965851708
md"next, we normalize by total population, as in the above definition:"

# ╔═╡ a64367e0-5bc1-11eb-0ea1-7ba30e6b4d73
ss, ii, rr = SS/NN, II/NN, RR/NN

# ╔═╡ 758456c6-6172-11eb-3ad5-cfd8416ff735
md"finally, we set the transmission rate $\beta$ and the recovery rate $\gamma$"

# ╔═╡ b53e57c8-5bc1-11eb-1a22-6da90e069232
β, γ = 0.1, 0.01

# ╔═╡ 8a59e9c6-6172-11eb-3159-ebb009cab7ff
md"## Introducing Lockdown

* A simple way to simumlate a lockdown policy is to introduce an additional *time varying* parameter called the **contact rate**
* This is supposed to capture at which *rate* people meet each other at any given time.
* Let's call it $c_t$:

$$\begin{align}
s_{t+1} &= s_t - c_t \beta \, s_t \, i_t \\
i_{t+1} &= i_t + c_t \beta \, s_t \, i_t - \gamma \, i_t\\
r_{t+1} &= r_t + \gamma \, i_t
\end{align}$$

You should copy the code from the above linked notebook into the cell below, and modify it such that this change will be taken care off.
"

# ╔═╡ 8040fd60-5bc0-11eb-2be1-01265f68ec1b
# you need to copy the discrete_SIR code from the linked notebook above 
# and modify it accordingly!
function discrete_lockdown(β, γ, s0, i0, r0, c, T=1000)

	
	results = missing
	
	
	
	return results
end

# ╔═╡ 845c1ad4-6178-11eb-149a-4d73019356d0
if ismissing(discrete_lockdown(β,γ, 0.1,0.3,0.1,1))
md"""
!!! warning
    you need to fix the body of the `discrete_lockdown` function first.
"""
end	

# ╔═╡ 472fc704-6174-11eb-30fe-8d8463ab536c
md" ## Simulating a given lockdown policy

* run your `discrete_lockdown(β,γ,ss,ii,rr,contact, TT)` function, choosing for `contact` a vector of values that will represent the *intensity* of the lockdown
* if `c[3] = 0` this means that in period 3 the contact rate is zero, hence an extremely strict and effective lockdown.
* It's probably a good idea to start with `c = ones(TT)` such that you replicate the exact same model as in the lecture (i.e. no lockdown)

"

# ╔═╡ 87f504ba-5bc1-11eb-1919-73376bbf34a0
SIR = discrete_lockdown(β,γ,ss,ii,rr,contact, TT)

# ╔═╡ 18b23e9c-6175-11eb-3e59-2775a1bc5a94
md"
* Now Make a plot similar to the one in the lecture from that first model. (copy the code!)
* If you assign the plot to a variable, then we can display it in different locations of the notebook, which is going to be useful. (this is already in the code you will copy)
* You should add two vertical lines that indicate the start and end date of your lockdown such that it's easier to see! google *plots.jl vertical line*!
"

# ╔═╡ 78a17740-5bc2-11eb-043a-2f7d348236c4
begin
	# here code that makes the plot!
	# copy from the lecture notebook!
	discrete_time_SIR_plot = plot()  # i'll make an empty plot for you ;-)
end

# ╔═╡ 4679a54a-6175-11eb-3f65-4d91d741140f
md"## Implement a lockdown!

* Great! now wouldn't it be cool to see how changing some parameters would affect this plot?
* Start with setting `contact` to a something like `0.25` in periods 50 to 100. For that, you should set the indices `50:100` of `contact` to 0.25.
* Notice that you don't have to move the cell where you defined `contact` around! just leave it where it is!
"

# ╔═╡ a43170b4-6175-11eb-1ceb-792e90936ade
md"## Make it Move!

* very nice. now lets make this operation automatic. 
* Instead of manually setting the numbers in `contact`, let's choose just three numbers: `🔒start` *start date*, `🔒stop` *end date* and `🔒intensity` *intensity* of the lockdown.
* ( you can get the lock via `\:lock: + tab` )
* then you can define your `contact` such that up to period `🔒start` it is equal to 1, then you fill in `1-🔒intensity`, and finally it's 1 again after 🔒stop"

# ╔═╡ 280c9750-617a-11eb-1599-e7af6e944315
md"""
!!! hint
    something like this maybe ? `contact = [1.0 - 0.25*(it > 100) + 0.25*(it > 200) for it in 1:TT]`. you need to replace 0.25, 100 and 200! ;-) but again, DO NOT REDEFINE contact here, instead modify where you first defined it
"""

# ╔═╡ 13e7c542-5bc3-11eb-3bd4-877eb2050c77
@bind 🔒start Slider(1:200,show_value = true, default = 50)

# ╔═╡ 68d2e158-5bc4-11eb-2a84-35548106ce43
@bind 🔒stop  Slider(1:200,show_value = true, default = 100)

# ╔═╡ d10fe5c2-5bc4-11eb-3f5f-fb8a8220cef4
@bind 🔒intensity  Slider(0.0:0.01:1.0,show_value = true, default = 0.5)

# ╔═╡ d9ba975a-5bc2-11eb-13bc-af33da4a7ef0
discrete_time_SIR_plot

# ╔═╡ dc9a2a08-6709-11eb-0e6d-836db998a5c9
begin
	if (student.name == "Jazzy Jeff") 
		Markdown.MD(Markdown.Admonition("danger", "Oops!", [md"You are not *really* called **Jazzy Jeff**. Please fill out name!"]))
	elseif !(contains(student.email, "@"))
		Markdown.MD(Markdown.Admonition("danger", "Oops!", [md"Please enter a valid email address.  "]))
	elseif (student.email == "jazzy.jeff@yahoo.com")
		Markdown.MD(Markdown.Admonition("danger", "Oops!", [md"Please enter a valid email in the `student` tuple above.  "]))
	end
end

# ╔═╡ Cell order:
# ╠═cbf5f6c0-616f-11eb-2bfc-97f62c8694d2
# ╠═8c9fb16e-5bc5-11eb-14ce-039b59e408ef
# ╠═d3241b7a-6a43-11eb-13b0-5d35d22508f1
# ╟─14d5c6aa-670a-11eb-3cd3-9dd24111525b
# ╠═57005404-6170-11eb-3ffc-f57feda21784
# ╟─6239ebaa-6170-11eb-2217-c1937ab254bc
# ╟─362d2fba-5bce-11eb-14da-ef225485facb
# ╟─1d791a08-6171-11eb-3857-8da3d47c397e
# ╠═8fe81ca2-5bc1-11eb-31a5-b39641ed1225
# ╟─1fdf8918-6172-11eb-1719-63a5b105ed7d
# ╟─67018b82-6172-11eb-37a5-77d965851708
# ╠═a64367e0-5bc1-11eb-0ea1-7ba30e6b4d73
# ╟─758456c6-6172-11eb-3ad5-cfd8416ff735
# ╠═b53e57c8-5bc1-11eb-1a22-6da90e069232
# ╟─8a59e9c6-6172-11eb-3159-ebb009cab7ff
# ╠═8040fd60-5bc0-11eb-2be1-01265f68ec1b
# ╟─845c1ad4-6178-11eb-149a-4d73019356d0
# ╟─472fc704-6174-11eb-30fe-8d8463ab536c
# ╠═87f504ba-5bc1-11eb-1919-73376bbf34a0
# ╟─18b23e9c-6175-11eb-3e59-2775a1bc5a94
# ╠═78a17740-5bc2-11eb-043a-2f7d348236c4
# ╟─4679a54a-6175-11eb-3f65-4d91d741140f
# ╟─a43170b4-6175-11eb-1ceb-792e90936ade
# ╟─280c9750-617a-11eb-1599-e7af6e944315
# ╠═13e7c542-5bc3-11eb-3bd4-877eb2050c77
# ╠═68d2e158-5bc4-11eb-2a84-35548106ce43
# ╠═d10fe5c2-5bc4-11eb-3f5f-fb8a8220cef4
# ╠═d9ba975a-5bc2-11eb-13bc-af33da4a7ef0
# ╟─dc9a2a08-6709-11eb-0e6d-836db998a5c9
