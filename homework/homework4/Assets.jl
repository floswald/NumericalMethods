### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 313a1e80-7b42-11eb-2b4c-cd7fe27b5483
# load some packages
begin
	using NLopt
	using DataFrames
	using Ipopt
	using JuMP
	using LinearAlgebra
	using Statistics
	using Plots
end

# ╔═╡ c69606ac-7b41-11eb-075f-27802f4d671e
md"
# A Portfolio Choice Problem

We want to solve the portfolio allocation problem of an investor. You can think that they want to keep some part of their endowment for consumption $c$, and invest the remainder into one of $n$ financial assets. You can think about this as an intertemporal choices (consumption _today_ vs consumption _tomorrow_), but it is not necessary. 

* Assume we have $n$ assets, and asset number 1 is the safe asset. It will pay 1 in any state.
* Assets $i=2,\dots,n$ are risky and pay out $z_i^s$ in state $s=1,\dots,S$
* State $s$ occurs with probability $\pi^s$
* The investor is characterized by
    * an initial endowment of each asset: $(e_1,e_2,\dots,e_n)$
    * a utility function $u(c) = -\exp(-ac)$
* The problem of the investor is to choose consumption $c$ and how much of each asset to hold. $\omega_i$ represents the number of units of asset $i$ the investor wants to hold. Notice that there is no constraint on $\omega_i$, so could be negative or positive.
$$\begin{align}
    \max_{c,\omega_1,\dots,\omega_n} u(c) + \sum_{s=1}^S \pi^s u \left( \sum_{i=1}^n \omega_i z_i^s \right) \quad (1)\\
    \text{subject to   } c + \sum_{i=1}^n p_i \omega_i = \sum_{i=1}^n p_i e_i  \quad (2)\\
c \geq 0,\quad w_i \in \mathbb{R}
\end{align}$$

## Setup

* Assume $n=3,p=(1,1,1),e=(2,0,0)$ and suppose that $z_2 = (0.72,0.92,1.12,1.32),z_3=(0.86,0.96,1.06,1.16)$. Each of these realizations is equally likely to occur, and all assets are independent. This means we have a total of $4\times4=16$ random states.

"

# ╔═╡ 4d3447d2-7b42-11eb-1ce3-095b89785f6c
md"
# Question 1: Data Creator

* write a function that creates the data used in this exercise.
* function `data(a = 0.5)` should return a `Dict` with keys `:a,:na,:nc,:ns,:nss,:e,:p,:zp,:z,:π`, where `:nc` is number of choice variables, `:ns` is number of states for each asset and `nss` is number of total states of the world, `:zp` is a matrix of payoffs for each asset (`4 x 3`).
* I suggest to create a `z` matrix which has one column for each asset, and one row for each state of the world. Notice that *independence* of assets implies that it's equally likely to have $(z_2^1, z_3^1)$ (both risky assets in same state of world $s=1$, as it is to have $(z_2^3, z_3^2)$. The first column of `z` is all ones (the safe asset pays 1 in each state of the world). There should be 16 rows."



# ╔═╡ 06219b52-7b43-11eb-3ca9-c334f6d53908
function data(a=0.5)
	# na =   # number of assets
	# nc =   # number of choice variables: na assets + 1 consumption
	# ns =  # number of states for each asset
	# nss =  # number of total states of the world
	# e = 
	# p = 
	# zp = 
	# z = 
	# return Dict(...)
end

# ╔═╡ f07efffc-7cd9-11eb-15aa-338b74552e6a
md"
## Question 1.2

👉 What is the expected payoff of each asset if the portfolio is $(\omega_1,\omega_2,\omega_3) = (1,1,1)$? Write a function `mean_payoff(d)` that returns a `Array` (1, 3),  computing the average payoff across all 4 states of the world for each asset, i.e. columns $i$ is

$$\sum_{s=1}^S \pi^s \omega_i z_i^s$$

have a look at the documentation of `mean` to find out about operating along `dims`!
"

# ╔═╡ 03302a44-7cdb-11eb-0e34-73d2add94f8e
mean_payoff(d) = missing

# ╔═╡ d44226c2-7cda-11eb-1e95-d5c362a2a27c
md"
## Question 1.3

👉 What is the variance of each asset's payoff if the portfolio is $(\omega_1,\omega_2,\omega_3) = (1,1,1)$? Write a function `var_payoff(d)` that returns a `Array` (1, 3),  computing the variance of payoff across all 4 states of the world for each asset.


"

# ╔═╡ 12a088f2-7cdb-11eb-01a6-21baf4b9ba02
var_payoff(d) = missing

# ╔═╡ 632f1fea-7ce5-11eb-17ab-b5d2e8fa6995
md"* How do you compare the assets, now that you know their average payoff and variance? Which ones would you choose to invest in, if any?
* How could we represent the way you think about this tradeoff?"

# ╔═╡ 39f70dac-7b45-11eb-16d7-316a34077ace
md"

## Question 2.1

> How could we represent the way you think about this tradeoff?

_Well, with a uility function of course!_

* Write down the given utility function `u` and it's first derivative `up`!
* Both take 2 arguments and return a number.
"

# ╔═╡ 5124ed8a-7b45-11eb-0984-a944bd9302e0
u(a,c) = missing

# ╔═╡ 587f244c-7b45-11eb-0238-070d76122af1
up(a,c) = missing

# ╔═╡ a6f5e970-7ce5-11eb-179c-39d4f242c202
md"
## Question 2.2

Let's generate some intuition before we attempt to solve this numerically. To do so, let's simplify the problem. So, suppose there is only one asset, endowment is given by the positive number $y$, and consumer cares about consumption today $c_1$ and tomorrow $c_2$, no discounting, no interest on savings. In short, the problem is

$$\begin{array}{c}
    \max_{c_1, c_2} u(c_1) + u(c_2) \\
    \text{subject to   } c_2 = y - c_1\\
c_i \geq 0
\end{array}$$

👉 Given the above utility function, what is the optimal level of consumption in each period?  

👉 Do you think this result will survive if we introduce uncertainty and asset choices?

**Don't look at the hint! It shows you the solution. First think about it a little while!** it is easy. 😉
"

# ╔═╡ d5555454-7da6-11eb-052e-7313220e3edb
md"

## Question 2.3

Let's look at the utility function!

👉 make a plot that shows $u(c,a)$ for different values of $a$!
"

# ╔═╡ f22090dc-7da6-11eb-29dc-51aac7458cb2
let
	# your plot here
end

# ╔═╡ 1910d264-7da8-11eb-1434-fb54dbeda64c
md"

### Question 2.4


* What is this expression called in economics, and what is it equal to here?

$$\frac{-u''(c)}{u'(c)}$$

* What's your interpretation of the number $a$ in this context? How do you expect our investor to behave as $a$ increases?

"

# ╔═╡ 13b50308-7b46-11eb-034c-c3e2b017e5e9
md"

# 3. Numerical Solution to the problem

Here is the correct solution to the problem for three different values of `a`. We will compare your solution against this dataframe. Please compute a solution for $a\in\{0.5,1,1.5\}$.
"

# ╔═╡ c1a8d1ec-7b99-11eb-02bb-6917b65f33a7
solution = DataFrame(a=[0.5;1.0;5.0],c = [1.008;1.004;1.0008],omega1=[-1.41237;-0.20618;0.758763],omega2=[0.801455;0.400728;0.0801455],omega3=[1.60291;0.801455;0.160291],fval=[-1.20821;-0.732819;-0.013422])

# ╔═╡ b1e51722-7b99-11eb-33ec-9dc089e8b7fe
md"

## 3.1. Solve this problem using `NLopt`


* Define 2 functions `obj(x::Vector,grad::Vector,data::Dict)` and `constr(x::Vector,grad::Vector,data::Dict)`, similar to the way we set this up in class.
* In particular, remember that both need to modify their gradient `in-place`. 

👉 `obj` returns the value of equation (1) and its gradient wrt all choice variables in `grad`

👉 `constr` returns the value of equation (2) and its gradient wrt all choice variables in `grad`

"

# ╔═╡ 32f9d464-7b46-11eb-274c-0713593fe800
function obj(x::Vector,grad::Vector,data::Dict)

end

# ╔═╡ 3af65458-7b46-11eb-2d0c-e5a480ffe003
function constr(x::Vector,grad::Vector,data::Dict)

end

# ╔═╡ 389f51a0-7cea-11eb-19a3-438cb29fc2e0
md"

## 3.2 Numerically Solve!

now call the NLopt solver to get the solution.

👉 write a function `max_NLopt(a=0.5)` which 
1. creates a data set with `data(a)`
2. sets up an `Opt` instance using the `:LD_SLSQP` algorithm for equality constraints.
3. sets objective, equality constraint, and bounds
4. returns a tuple `(optf,optx,ret)` with `(optimal f value, optimal choice, return code)`
5. choose as starting value $0.2$ for all choice variables.
"
	

# ╔═╡ 84893626-7b46-11eb-2e81-dbb07a7afdf6
function max_NLopt(a=0.5)

	return missing
end

# ╔═╡ 74ada204-7ceb-11eb-0f48-ddc9c1a5fec8
md"Here is your answer: check the return code! somthing like `:FTOL_REACHED` would be good!"

# ╔═╡ 97f12fa2-7b46-11eb-0183-199ee3de1de1
nlopt_res = max_NLopt()

# ╔═╡ e5233fa0-7b9a-11eb-39cc-c70c0addca97
md"check that the budget constraint is satisfied at this solution"

# ╔═╡ ef08e970-7b9a-11eb-203d-41095002a3b7
constr(nlopt_res[2],zeros(4),data()) == 0.0

# ╔═╡ aa8e4912-7ceb-11eb-0705-2bf588d303fe
md"

### 3.3 checking

What is the expected utility of the optimal investment portfolio, i.e.  what is this number?

$$\sum_{s=1}^S \pi^s u \left( \sum_{i=1}^n \omega_i^* z_i^s \right)$$

👉 write a function `exp_u(a=0.5)` which 
1. creates a data set with `data(a)`
2. gets your solution with `max_NLopt(a)`
3. Returns the above expression
"

# ╔═╡ 2de86fba-7b9b-11eb-379d-0b0f85d9c9a3
function exp_u(a=0.5)

	return missing
end

# ╔═╡ f910cfce-7b9b-11eb-3162-9be77eb705ee
md" 

### 3.4 Checking $u$

And finally: what is $u(c^*)$?

👉 write a function `opt_c(a=0.5)` which 
1. gets your solution with `max_NLopt(a)`
2. Returns $u(c^*)$


I'll then check whether `opt_c(a) ≈ exp_u(a)`!
"

# ╔═╡ 8913fa58-7cec-11eb-1bfa-cf84ef627bd2
function opt_c(a=0.5)
	nlopt_res = max_NLopt(a)
	u(nlopt_res[2][1],a)
end

# ╔═╡ 84b43576-7ced-11eb-37c3-5f6b595f0730
md"
### 3.5 Checking all your results

* In the next cell I run your model for all $a$ values!
"

# ╔═╡ f078be5a-7b4c-11eb-3f03-138505242d43
md"

# 4. Now solve with `JuMP`!

* Let's solve the identical problem with JuMP.jl now.
* write a function `max_JuMP(a)` that defines the constrained optimization problem
* it should return a dict with optimal values `:obj`, `:c` and `:omega`.
"

# ╔═╡ 0e6842e2-7cf0-11eb-0707-35a4d85c382c
function max_JuMP(a=0.5)

	return missing
end



# ╔═╡ b771d01e-7da5-11eb-357b-f7d3641e3803
max_JuMP()

# ╔═╡ 0e41d664-7da6-11eb-3598-bd1ce2944ebd
md"
### 4.2 Checking all your results

* In the next cell I run your model for all $a$ values!
"

# ╔═╡ 7cc76266-7da6-11eb-3c61-6d4b7206c46c
md"

# 5. Going Further

Here are couple of fun things to try, often with minimal effort:

1. how does the solution change if $\omega_i \geq 0,\forall i$?
2. How does the solution change with a different utility function? In the Jump part, just use a different u!


"

# ╔═╡ a1e0ec36-7cee-11eb-3e4c-69ad1c5d018f
md"Function Library"

# ╔═╡ 061b765e-7b4c-11eb-337a-61f8783875bb
function table_NLopt()
	d = DataFrame(a=[0.5;1.0;5.0],c = zeros(3),omega1=zeros(3),omega2=zeros(3),omega3=zeros(3),fval=zeros(3))
	for i in 1:nrow(d)
		xx = max_NLopt(d[i,:a])
		for j in 2:ncol(d)-1
			d[i,j] = xx[2][j-1]
		end
		d[i,end] = xx[1]
	end
	return d
end

# ╔═╡ 13d41f9e-7cf0-11eb-1e01-f1ddcbe01b8a
function table_JuMP()
	d = DataFrame(a=[0.5;1.0;5.0],c = zeros(3),omega1=zeros(3),omega2=zeros(3),omega3=zeros(3),fval=zeros(3))
	for i in 1:nrow(d)
		xx = max_JuMP(d[i,:a])
		d[i,:c] = xx["c"]
		d[i,:omega1] = xx["omegas"][1]
		d[i,:omega2] = xx["omegas"][2]
		d[i,:omega3] = xx["omegas"][3]
		d[i,:fval] = xx["obj"]
	end
	return d
end

# ╔═╡ 88317198-7b42-11eb-1364-470e8ef725d9
hint(text) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]))

# ╔═╡ f149f7b0-7ce6-11eb-207b-d3830fbf83ef
hint(md"""
$$\begin{array}{c}
    u(c_1) + u(c_2) \\
    \text{subject to   } c_2 = y - c_1\\
	\Rightarrow \\
	u(c_1) + u(y - c_1) \\
	u(c_1)' = u'(y - c_1) \\
	a \exp(-a c_1) = a \exp(-a (y - c_1)) \\
	 -a c_1 =  -a (y - c_1) \\	
 	2 c_1 =  y \\
	c_1^* = \frac{y}{2} = c_2^* 
\end{array}$$
	
So, consumption is *constant*. What's the solution to this question if $u$ were linear?
""")

# ╔═╡ babf2e98-7b42-11eb-09ef-1bfcc53d3d16
almost(text) = Markdown.MD(Markdown.Admonition("warning", "Almost there!", [text]))

# ╔═╡ bf3b2792-7b42-11eb-0e28-295cbed65a13
still_missing(text=md"Replace `missing` with your answer.") = Markdown.MD(Markdown.Admonition("warning", "Here we go!", [text]))

# ╔═╡ c47c5640-7b42-11eb-3b72-25362925c2d1
keep_working(text=md"The answer is not quite right.") = Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]))

# ╔═╡ c961ea44-7b42-11eb-3a35-bfd505fb99fb
yays = [md"Fantastic!", md"Splendid!", md"Great!", md"Yay ❤", md"Great! 🎉", md"Well done!", md"Keep it up!", md"Good job!", md"Awesome!", md"You got the right answer!", md"Let's move on to the next section."]

# ╔═╡ cd08f232-7b42-11eb-3b5b-55079d868638
correct(text=rand(yays)) = Markdown.MD(Markdown.Admonition("correct", "Got it!", [text]))

# ╔═╡ 0a5c81f2-7b4c-11eb-2a96-7163671eed67
let
	r = maximum(abs.(Array(solution .- table_NLopt())))
	if r > 1e-4
		keep_working(md"Your results are not close enough to the correct ones (above in `solution`!)")
	else
		correct()
	end
end

# ╔═╡ 19d9f3f8-7da6-11eb-169c-9765c55cfac6
let
	r = maximum(abs.(Array(solution .- table_JuMP())))
	if r > 1e-4
		keep_working(md"Your results are not close enough to the correct ones (above in `solution`!)")
	else
		correct()
	end
end

# ╔═╡ d0d5d506-7b42-11eb-0517-6318fd0ede8b
not_defined(variable_name) = Markdown.MD(Markdown.Admonition("danger", "Oopsie!", [md"Make sure that you define a variable called **$(Markdown.Code(string(variable_name)))**"]))

# ╔═╡ d4548c4a-7b42-11eb-294f-d1c27fe76543
not_definedf(variable_name) = Markdown.MD(Markdown.Admonition("danger", "Oopsie!", [md"Make sure that you define a function called **$(Markdown.Code(string(variable_name)))**"]))

# ╔═╡ 1532e6bc-7b43-11eb-3f27-59134ea6acc0
if !@isdefined(data)
	not_definedf(:data)
else
	let
		res = data()
		if res isa Missing
			still_missing()
		elseif !(res isa Dict)
			keep_working(md"Make sure you return a `Dict`")
		elseif !issetequal(collect(keys(res)),[:a,:na,:nc,:ns,:nss,:e,:p,:z,:zp,:π])
			keep_working(md"your Dict needs to have keys [:a,:na,:nc,:ns,:nss,:e,:p,:z,:π]")
		elseif res[:na] != 3
			keep_working(md"We need three assets!")
		elseif length(res[:e]) != 3
			keep_working(md"the :e vector needs to be 3 x 1!")
		elseif size(res[:z]) != (16,3)
			almost(md"your z matrix should be 16 x 3")
		elseif res[:e] != [2.0;0;0]
			almost(md"check your `:e` key! ")
		else
			correct()
		end
	end
end

# ╔═╡ 2757d17a-7cda-11eb-3694-65c409660196
if !@isdefined(mean_payoff)
	not_definedf(:mean_payoff)
else
	let
		d = data()
		res = mean_payoff(d)
		if res isa Missing
			still_missing()
		elseif !(res isa Array)
			keep_working(md"Make sure you return a 1 by 3 `Array`")
		elseif !(norm(res .- mean(d[:zp],dims=1)) < 1e-6)
			almost(md"your result is incorrect.")
		else
			correct()
		end
	end
end

# ╔═╡ 20316894-7cdb-11eb-2fec-f3f6876126ba
if !@isdefined(var_payoff)
	not_definedf(:var_payoff)
else
	let
		d = data()
		res = var_payoff(d)
		if res isa Missing
			still_missing()
		elseif !(res isa Array)
			keep_working(md"Make sure you return a 1 by 3 `Array`")
		elseif !(norm(res .- var(d[:zp],dims=1)) < 1e-6)
			almost(md"your result is incorrect.")
		else
			correct()
		end
	end
end

# ╔═╡ 6183c9a8-7b45-11eb-0bd5-1302c23d6570
if !@isdefined(u)
	not_definedf(:u)
else
	let
		a = 1.1
		c = 0.5
		res = u(a,c)
		if res isa Missing
			still_missing()
		elseif !(res == -exp(-a * c))
			keep_working(md"check your function definition! it's wrong.")
		else
			correct()
		end
	end
end

# ╔═╡ 5bef3e96-7b45-11eb-257a-09f7dbe66aaa
if !@isdefined(up)
	not_definedf(:up)
else
	let
		a = 1.1
		c = 0.5
		res = up(a,c)
		if res isa Missing
			still_missing()
		elseif !(res == a * exp(-a * c))
			keep_working(md"check your function definition! it's wrong.")
		else
			correct()
		end
	end
end

# ╔═╡ 241521be-7ceb-11eb-0553-2de70f4c57e9
if !@isdefined(max_NLopt)
	not_definedf(:max_NLopt)
else
	let
		res = max_NLopt()
		if res isa Missing
			still_missing()
		elseif !(res isa Tuple)
			keep_working(md"You should return a tuple")
		else
			correct()
		end
	end
end

# ╔═╡ 06aba85e-7cec-11eb-075f-23d500eda202
if !@isdefined(exp_u)
	not_definedf(:exp_u)
else
	let
		res = exp_u(0.5)
		if res isa Missing
			still_missing()
		elseif !(res isa Number)
			keep_working(md"You should return a number")
		else
			correct()
		end
	end
end

# ╔═╡ cac793ec-7cec-11eb-12a0-b7caf963d07f
if !@isdefined(opt_c)
	not_definedf(:opt_c)
else
	let
		a = 0.5
		res = opt_c(a)
		if res isa Missing
			still_missing()
		elseif !(res isa Number)
			keep_working(md"You should return a number")
		elseif abs(res - exp_u(a)) > 1e-4
			keep_working(md"your opt_c(0.5) and exp_u(a) should be approximately equal!")
		else
			correct()
		end
	end
end

# ╔═╡ 316f5b3e-7da6-11eb-1d30-5f7a350de422
if !@isdefined(max_JuMP)
	not_definedf(:max_JuMP)
else
	let
		res = max_JuMP()
		if res isa Missing
			still_missing()
		elseif !(res isa Dict)
			keep_working(md"You should return a Dict")
		elseif !issetequal(collect(keys(res)),["obj","c","omegas"])
			keep_working(md"""your Dict needs to have keys ["obj","c","omegas"]""")
		else
			correct()
		end
	end
end

# ╔═╡ d7915d7a-7b42-11eb-2ea7-4f0b7bd6a904


# ╔═╡ Cell order:
# ╟─c69606ac-7b41-11eb-075f-27802f4d671e
# ╠═313a1e80-7b42-11eb-2b4c-cd7fe27b5483
# ╟─4d3447d2-7b42-11eb-1ce3-095b89785f6c
# ╠═06219b52-7b43-11eb-3ca9-c334f6d53908
# ╟─1532e6bc-7b43-11eb-3f27-59134ea6acc0
# ╟─f07efffc-7cd9-11eb-15aa-338b74552e6a
# ╠═03302a44-7cdb-11eb-0e34-73d2add94f8e
# ╟─2757d17a-7cda-11eb-3694-65c409660196
# ╟─d44226c2-7cda-11eb-1e95-d5c362a2a27c
# ╠═12a088f2-7cdb-11eb-01a6-21baf4b9ba02
# ╟─20316894-7cdb-11eb-2fec-f3f6876126ba
# ╟─632f1fea-7ce5-11eb-17ab-b5d2e8fa6995
# ╟─39f70dac-7b45-11eb-16d7-316a34077ace
# ╠═5124ed8a-7b45-11eb-0984-a944bd9302e0
# ╟─6183c9a8-7b45-11eb-0bd5-1302c23d6570
# ╠═587f244c-7b45-11eb-0238-070d76122af1
# ╟─5bef3e96-7b45-11eb-257a-09f7dbe66aaa
# ╟─a6f5e970-7ce5-11eb-179c-39d4f242c202
# ╟─f149f7b0-7ce6-11eb-207b-d3830fbf83ef
# ╟─d5555454-7da6-11eb-052e-7313220e3edb
# ╠═f22090dc-7da6-11eb-29dc-51aac7458cb2
# ╟─1910d264-7da8-11eb-1434-fb54dbeda64c
# ╟─13b50308-7b46-11eb-034c-c3e2b017e5e9
# ╟─c1a8d1ec-7b99-11eb-02bb-6917b65f33a7
# ╟─b1e51722-7b99-11eb-33ec-9dc089e8b7fe
# ╠═32f9d464-7b46-11eb-274c-0713593fe800
# ╠═3af65458-7b46-11eb-2d0c-e5a480ffe003
# ╟─389f51a0-7cea-11eb-19a3-438cb29fc2e0
# ╠═84893626-7b46-11eb-2e81-dbb07a7afdf6
# ╟─241521be-7ceb-11eb-0553-2de70f4c57e9
# ╟─74ada204-7ceb-11eb-0f48-ddc9c1a5fec8
# ╠═97f12fa2-7b46-11eb-0183-199ee3de1de1
# ╟─e5233fa0-7b9a-11eb-39cc-c70c0addca97
# ╠═ef08e970-7b9a-11eb-203d-41095002a3b7
# ╟─aa8e4912-7ceb-11eb-0705-2bf588d303fe
# ╠═2de86fba-7b9b-11eb-379d-0b0f85d9c9a3
# ╟─06aba85e-7cec-11eb-075f-23d500eda202
# ╟─f910cfce-7b9b-11eb-3162-9be77eb705ee
# ╟─8913fa58-7cec-11eb-1bfa-cf84ef627bd2
# ╟─cac793ec-7cec-11eb-12a0-b7caf963d07f
# ╟─84b43576-7ced-11eb-37c3-5f6b595f0730
# ╟─0a5c81f2-7b4c-11eb-2a96-7163671eed67
# ╟─f078be5a-7b4c-11eb-3f03-138505242d43
# ╠═0e6842e2-7cf0-11eb-0707-35a4d85c382c
# ╟─316f5b3e-7da6-11eb-1d30-5f7a350de422
# ╠═b771d01e-7da5-11eb-357b-f7d3641e3803
# ╟─0e41d664-7da6-11eb-3598-bd1ce2944ebd
# ╟─19d9f3f8-7da6-11eb-169c-9765c55cfac6
# ╟─7cc76266-7da6-11eb-3c61-6d4b7206c46c
# ╟─a1e0ec36-7cee-11eb-3e4c-69ad1c5d018f
# ╟─061b765e-7b4c-11eb-337a-61f8783875bb
# ╟─13d41f9e-7cf0-11eb-1e01-f1ddcbe01b8a
# ╠═88317198-7b42-11eb-1364-470e8ef725d9
# ╠═babf2e98-7b42-11eb-09ef-1bfcc53d3d16
# ╠═bf3b2792-7b42-11eb-0e28-295cbed65a13
# ╠═c47c5640-7b42-11eb-3b72-25362925c2d1
# ╠═c961ea44-7b42-11eb-3a35-bfd505fb99fb
# ╠═cd08f232-7b42-11eb-3b5b-55079d868638
# ╠═d0d5d506-7b42-11eb-0517-6318fd0ede8b
# ╠═d4548c4a-7b42-11eb-294f-d1c27fe76543
# ╠═d7915d7a-7b42-11eb-2ea7-4f0b7bd6a904
