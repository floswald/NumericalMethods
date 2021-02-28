### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ f3634aac-7289-11eb-3a70-c5696ac59412
begin
	# load required packages
	using Distributions
	using Optim
	using Plots
	using Random
	using LinearAlgebra
	using Calculus
end

# ╔═╡ aed8965e-75f3-11eb-35af-df22bcc9c11c
begin
	using GLM
	using Statistics
	using DataFrames
end

# ╔═╡ f4ee44c4-722c-11eb-3b10-2911aa3a1c84
md"""

# Probit 🔮 Monte Carlo: Maximum Likelihood

In this homework you will implement an estimator for the well known Probit model. To remind you, a simple version of this model is defined as follows:

$$\begin{align}  
y_i  &\in \{0,1\} \\
\Pr\{y_i=1|x_i\} &= \Phi(x_i \beta) \\
L(\beta)   & = \Pi_{i=1}^N  \Phi(x_i \beta)^{y_i} (1-\Phi(x_i \beta))^{1-y_i} \\
\beta  & \in \mathbb{R}^k \\
x_i  & \sim N\left([0,0,0],\left[ \begin{array}{ccc} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1\end{array} \right] \right) \\
k & = 3 
\end{align}$$

where $\Phi$ denotes the standard normal cdf and where $L$ denotes the likelihood function. Think of $x_i$ as a _row vector_ (`1,k`) such that if you stack all $x$ for all $n$ people you get an $(n,k)$ matrix where a row is an observation. Notice that a common way to set this up is as a latent variable model like

$$\begin{align}
y_i^* &= x_i \beta + \epsilon_i \\
y_i &= \mathbf{1}[y_i^* \geq 0] \\
\epsilon_i &\sim \mathcal{N}(0,1)
\end{align}$$

"""

# ╔═╡ de94544c-722d-11eb-1240-713caa73862a
md"
## Question 1: a data-creator

* 👉 write a function that creates a simulated dataset.
* set the true β to `[ 1; 1.5; -0.5 ]`
* Return a `Dict` with at keys `:beta` (true values), `:n` (number of observations, `:X` an `n,k` matrix of data and `:y` a response vector.
* You create a dict with `Dict(:key1 => value1, :key2 => value2, and so on)`
"

# ╔═╡ 895d2a70-722e-11eb-028a-5b2c44bdd141
function makeData(n=10000)
	Random.seed!(12345)      # make results reproducible be setting a seed
	# beta = ...  # truth
	# numobs = n
	# X =     # n,k matrix
	# epsilon =   # draw epsilon for all i
	# Y =    # compute y*^
	# y =        # derive y
	# norm = 	# create a normal distribution object with mean 0 and variance 1
	return missing
end

# ╔═╡ a2482bac-722e-11eb-004c-9d02a807d23c
md"
## Question 2: Write up the log likelihood function

* 👉 This is $\mathcal{L} = \log(L)$:

$$\mathcal{L}(\beta) = \sum_{y_i=1} \log{\Phi(x_i \beta)} + \sum_{y_i=0} \log{1- \Phi(x_i \beta)}$$ 

* For numerical stability it is better to focus on the _average_ log likelihood function, hence we divide through by the sample size $n$. Additionally we multiply by negative one, since `Optim.jl` wants to _minimize_ functions. Let's call this _the log likelihood function_ in what follows:

$$l(\beta) = -\frac{1}{n} \left( \sum_{y_i=1} \log{\Phi(x_i \beta)} + \sum_{y_i=0} \log{1- \Phi(x_i \beta)} \right)$$ 

* write the function `loglik(betas::Vector,d::Dict)` which will compute the log likelihood at a given value for the β.
* The maximum likelihood estimator will maximize that expression.
"

# ╔═╡ 5275176a-728e-11eb-3ed8-3bce4485e601
function loglik(betas::Vector,d::Dict)
	# xbeta     = ...	# (n,1)
	# G_xbeta   = cdf of xbeta...	# (n,1)
	# loglike   = expression inside round brackets  # (n,1)
	# objective = -1 times mean of loglike
	return missing
end

# ╔═╡ c65cfcbc-72b2-11eb-2b9e-9f1109e6d09d
md"
## Question 2.1 Let's visualize that

Let's visually check whether our function is coded up correctly.
* I made a graph with 3 panels below, each panel for one β, here is what it does:
* for panel $i$, vary the values of $\beta_i$ in a range of +-1 of its true value, and keep the other βs fixed.
* do the same for all βs.
* evaluate the likelihood function on each of those 3 grids of values.
* plot each range $\beta_i$ against the resulting log likelihood values.
* If your function is correct, the lowest function value should correspond to the $\beta_i = \beta_i^*$, i.e. the _true_ parameter value!
* See the role of sample size and accuracy of that plot!
"

# ╔═╡ 6a49a494-72b3-11eb-13d0-1bea0a985743
function plotLike()
	d = makeData(1000)
	ngrid = 100
	pad = 1.0
	k = length(d[:β])
	beta0 = reshape(repeat(d[:β],outer = ngrid, inner = 1),k,ngrid)
	pl = Any[]

	for ib in 1:k
		betas = copy(beta0)
		betas[ib,:] = collect(range(d[:β][ib] - pad, stop = d[:β][ib] + pad, length = ngrid))
		p0 = plot(betas[ib,:], mapslices(x -> loglik(x,d), betas, dims = 1)[:], title = "β$(ib-1)",leg = false, lw = 2)
		vline!(p0,[d[:β][ib]], color = :red,lw =2)
		push!(pl,p0)
	end
	plot(pl...,layout = (1,3),size = (700,200))
end

# ╔═╡ 7a2753d4-72b3-11eb-3eb0-c326168de29f
plotLike()

# ╔═╡ 547f08be-72b0-11eb-1dfa-f5caf3d2c9f2
md"
## Question 2.1 Maximize!

Let's maximize this function now for a first time. Use the `Optim.jl` function `optimize` and deploy the `NelderMead()` method on the problem. 
* Write a function `maximize_like( ; x0 = X0)` which takes a keyword argument `x0`
* set the variable `X0 = [0.8,1.0,-0.1]` to be our default _starting value_.
* Inside the function, create a dataset with `n=10_000` observations.
* Check that your result is close to the true values in `d[:β]`!
* You can get results with `Optim.minimizer(res)` where `res` is a returned object from `optimize`
"

# ╔═╡ 54144c70-72b2-11eb-350e-6bce5b0d0fc8
# X0 = [0.8,1.0,-0.1]   # uncomment that line!

# ╔═╡ d5feab24-72b0-11eb-22ec-e5c416a83a65
function maximize_like(; x0 = X0)

	return missing
end

# ╔═╡ 5f3c8cc0-7230-11eb-3da2-c139ef715daf
md"
## Question 3: Write up the *gradient* of the log likelihood function

* 👉 take the first derivative of the log likelihood function wrt β, i.e. compute $\frac{\partial l}{\partial \beta}$
* your function should comply with the `Optim.jl` API, i.e. it should modify it's first argument: call it `grad!(g::Vector, betas::Vector, d::Dict)`
* It should return `nothing`, instead it should modify the elements of `g`
* Notice that since β is a `(3,1)` vector, `g` is going to be `(3,1)` as well.
"

# ╔═╡ eca17258-72ab-11eb-08b5-f9a84a1aa986
function grad!(g::Vector,betas::Vector,d)

	return missing
end

# ╔═╡ 30d04bea-72b4-11eb-1d1a-59ac9614fdb4
md"
## Question 3.1 visualize

Let's visually check whether our function is coded up correctly.
* make the same graph as above, but now plot the value of the gradients for each parameter.
* Given that `g` is `(3,1)`, you will have three lines in each panel now. They should all be approximately equal to zero at the true parameter value
"

# ╔═╡ 21ac2226-72b4-11eb-2010-f3c411dc32bf
function plotGrad()
	d = makeData(1000)
	ngrid = 100
	pad = 1.0
	k = length(d[:β])
	beta0 = reshape(repeat(d[:β],outer = ngrid, inner = 1),k,ngrid)
	values = zeros(ngrid,k)
	grad = ones(k)

	pl = Any[]

	for ib in 1:k
		betas = copy(beta0)
		betas[ib,:] = collect(range(d[:β][ib] - pad, stop = d[:β][ib] + pad, length = ngrid))
		for i in 1:ngrid
			grad!(grad,betas[:,i][:],d)
			values[i,:] = grad
		end
		p0 = plot(betas[ib,:], values, title = "β$(ib-1)",leg = false, lw = 2)
		vline!(p0,[d[:β][ib]], color = :red,lw =2)
		push!(pl,p0)
	end
	plot(pl...,layout = (1,3),size = (700,200))
end

# ╔═╡ 470c8f36-72b4-11eb-2db8-d982bc481397
plotGrad()

# ╔═╡ 875d1ef2-72b2-11eb-1859-2b7ce0dab4c0
md"
## Question 3.2 Maximize $l$ with the gradient now!

Let's maximize this but now with the gradient!
* Write a function `maximize_like_grad( ; x0 = X0)` which takes a keyword argument `x0`
* Inside the function, create a dataset with `n=10_000` observations.
* Choose a method for Optim.jl that takes advantage of the supplied gradient.
* Check that the solver has converged.
* Check that your result is close to the true values in `d[:β]`!
"

# ╔═╡ abbf948e-72b2-11eb-3044-b5de8fee0bf5
function maximize_like_grad(; x0 = X0)

	return missing
end

# ╔═╡ a816dc78-7232-11eb-10ad-1be6fbd99abb
md"
## Question 4: Write up the *hessian* of the log likelihood function (from here on all bonus)

* 👉 take the second derivative of the log likelihood function wrt β, i.e. compute $\frac{\partial^2 l}{\partial \beta^2}$
* your function should comply with the `Optim.jl` API, i.e. it should modify it's first argument: call it `hessian!(h::Matrix, betas::Vector, d::Dict)`
* It should return `nothing`, instead it should modify the elements of `h`
"

# ╔═╡ 5d0d9492-72af-11eb-2530-078dc03f6c10
function hessian!(H::Matrix,betas::Vector,d)

	return missing
end

# ╔═╡ b053bc9e-72b4-11eb-0b8e-ffc0479fea16
md"
## Question 4.1 Maximize $l$ with the gradient and hessian now!

* Write a function `maximize_like_grad_hess( ; x0 = X0)` which takes a keyword argument `x0`
* Inside the function, create a dataset with `n=10_000` observations.
* Choose a suitable second order method to optimize this.
* Check that the solver has converged.
* Check that your result is close to the true values in `d[:β]`!
"

# ╔═╡ c8742db8-72b4-11eb-24be-b1363f3f4bcf
function maximize_like_grad_hess(; x0 = X0)

	return missing
end

# ╔═╡ f0e913d8-75ef-11eb-34e1-1175cbacafd8
md"# Question 5: standard errors!

We can use the either the inverse hessian or the _information matrix_ to obtain standard errors for the maximimum likelihood estimator. Here is how we calculate the covariance matrix for the estimator in either case

### The Inverse _observed_ information matrix

$$cov_1 \left(\beta^{ML} \right) = \frac{1}{n} \left( \left . \frac{\partial^2 l}{\partial \beta^2} \right|_{\beta = \beta^{ML}}  \right)^{-1}$$

We divide by $n$ because we computed the _average_ log likelihood all along. To get the standard errors for each $\beta$, just grab the diagonal elements of this $k,k$ matrix with `diag`, and then take the square root of each element.


### The inverse of the _expected_ information matrix

The information matrix is defined as the negative of the expected value of the hessian. It comes down to this expression:

$$\mathbf{I}(\beta) = \sum_{i=1}^n \frac{\phi(s_i)^2}{\Phi(s_i)(1-\Phi(s_i))} x_i'x_i$$

Then do the same as above, i.e the standard errors are 

    sqrt.(diag(inv_Info(betas,d)))

at the optimal `betas`.

"


# ╔═╡ f40e1210-75f2-11eb-0391-09eaa939ac85
md"
## Question 5.1 Benchmarking vs GLM.jl

The `GLM.jl` package is our goto package for general linear models. We can compare our hand-rolled standard errors against what comes out of that package to check!

Here is how:
	
```julia
using GLM
using Statistics
using DataFrames
d = HWUnconstrained.makeData();
df = hcat(DataFrame(y=d[:y]),convert(DataFrame,d[:X]))
gg = glm(@formula(y~x1+x2+x3),df,Binomial(),ProbitLink())
```

The `Statistics` package contains the functions `stderror` and `coef` which extract the corresponding objects from all sorts of models. so you can use `stderror(gg)` to get the standard errors from this GLM probit model! the cells below use this method to check your standard errors.

"

# ╔═╡ 92f15dac-75f5-11eb-28b1-47b33da031c5
md"Define the function `inv_observedInfo(betas::Vector,d)` here!"

# ╔═╡ bf01e018-75f2-11eb-0cec-8957850d2db9
function inv_observedInfo(betas::Vector,d)
	missing
end

# ╔═╡ 64995e54-75f6-11eb-39c7-61406ddc7a6e
md"Define finally the function `inv_Info(betas::Vector,d)` here!"

# ╔═╡ 7a069524-75f4-11eb-34b7-db06fac6d646
function inv_Info(betas::Vector,d)
	missing
end

# ╔═╡ 9d49c7fc-75fb-11eb-3f8a-e355f688ea97
md"Functions library"

# ╔═╡ 8f0b6128-7289-11eb-1e54-0930af02a28e
hint(text) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]))

# ╔═╡ f8b01012-722f-11eb-21c0-d3ef14765e11
hint(md"Remember that `.` will apply your function to an array. So, to compute something like `log(Φ(x_i β)` could compute `Xβ` first (an `n,1` vector) and then apply `Φ` and `log` to it. Search the `Distributions` package for the `cdf` function!")

# ╔═╡ e11cda26-72b0-11eb-0ba9-759ecfc8a29b
hint(md"""
Notice that `loglik` takes 2 arguments, `x` _and_ `d`! However, `optimize` expects a single argument `x` for the objective function. A simple solution is to create a _closure_ with an anoymous function:
	
```julia
myfun(x,a,b) = a * x^2 + b
a = 2
b = 3
optimize( y -> myfun(y,a,b), x0 )
```
	""")

# ╔═╡ 990d630a-7232-11eb-2e53-a337a97012c8
hint(md"""
Let $s_i = x_i \beta$. Then
	
$$\frac{\partial l}{\partial \beta} = -\frac{1}{n} \left( \sum_{y_i=1} \phi(s_i)\Phi(s_i)^{-1}x_i -  \sum_{y_i=0} \phi(s_i)(1-\Phi(s_i))^{-1}x_i  \right)$$

	
where $\phi(x)$ is the pdf of the standard normal distribution.
""")

# ╔═╡ e2e6fe3c-72b4-11eb-18a6-03ea4a69759f
hint(md"Notice that here the anonymous function for the gradient needs to take 2 arguments:

	(g,x) -> grad!(g,x,d)
	
i.e. your call to `optimize` will look something like
	
	optimize((x)->loglik(x,d),(g,x)->grad!(g,x,d), ...)
")

# ╔═╡ fc476bc6-72ac-11eb-27b7-d9e4adac8a1f
hint(md"""
The hessian is defined as 
	
$$\begin{align}
	\frac{\partial^2 l}{\partial \beta^2} &= \frac{1}{n} \left( \sum_{y_i=1} \left[ s_i\phi(s_i)\Phi(s_i)^{-1} + \phi(s_i)^2\Phi(s_i)^{-2} \right] x_i'x_i \right.\\
	&+ \left. \sum_{y_i=0}\left[ \phi(s_i)^2(1-\Phi(s_i))^{-2} - s_i\phi(s_i)(1-\Phi(s_i))^{-1} \right] x_i'x_i \right)
	\end{align}$$
	
You can vectorize some of this computation. For example the $\Phi(s_i)$ is $\Phi(s)$ if you stack all the $x_i$. In fact, the _entire square bracketed expressions_ can be precomputed for all $i$ in one go. You then need to loop over this for each $i$ , form the matrix $x_i'x_i$ (which is (3,3))
	
""")

# ╔═╡ f6a46804-75f4-11eb-0aae-7de5394c4864
hint(md"""
	
```julia
function info_mat(betas::Vector,d)
	xbeta     = d[:X]*betas	# (n,1)
	G_xbeta   = cdf.(d[:dist],xbeta)	# (n,1)
	one_G_xbeta= 1.0 .- G_xbeta
	g_xbeta   = pdf.(d[:dist],xbeta)	# (n,1)

	out = zeros(length(betas),length(betas))

	for i in 1:d[:n]
		XX = d[:X][i,:] * d[:X][i,:]'   #k,k
		out[:,:] = out + XX * g_xbeta[i]^2 / (G_xbeta[i] * one_G_xbeta[i])
	end
	return out
end
```
""")

# ╔═╡ 94ef2d92-7289-11eb-2293-3f0c04f52e93
almost(text) = Markdown.MD(Markdown.Admonition("warning", "Almost there!", [text]))

# ╔═╡ 9606a958-7289-11eb-32f9-eb8b2deaf1f2
still_missing(text=md"Replace `missing` with your answer.") = Markdown.MD(Markdown.Admonition("warning", "Here we go!", [text]))

# ╔═╡ 9a4edf28-7289-11eb-3773-ab2ba97acc11
keep_working(text=md"The answer is not quite right.") = Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]))

# ╔═╡ 9ebe21d4-7289-11eb-3197-b969d3a402ce
yays = [md"Fantastic!", md"Splendid!", md"Great!", md"Yay ❤", md"Great! 🎉", md"Well done!", md"Keep it up!", md"Good job!", md"Awesome!", md"You got the right answer!", md"Let's move on to the next section."]

# ╔═╡ a32fcc86-7289-11eb-3134-8b5e7a91dbda
correct(text=rand(yays)) = Markdown.MD(Markdown.Admonition("correct", "Got it!", [text]))

# ╔═╡ a843de22-7289-11eb-21c4-cb0ecb4d5aad
not_defined(variable_name) = Markdown.MD(Markdown.Admonition("danger", "Oopsie!", [md"Make sure that you define a variable called **$(Markdown.Code(string(variable_name)))**"]))

# ╔═╡ 2ee2330e-72b2-11eb-33ad-27efe91db68b
if !@isdefined(X0)
	not_defined(:X0)
elseif length(X0) != 3
	keep_working(md"length of `X0` should be 3!")
end

# ╔═╡ b1b5ebe6-728e-11eb-1865-9f63880d9671
not_definedf(variable_name) = Markdown.MD(Markdown.Admonition("danger", "Oopsie!", [md"Make sure that you define a function called **$(Markdown.Code(string(variable_name)))**"]))

# ╔═╡ 32f59f3c-728a-11eb-35bd-7352a92bf6b0
if !@isdefined(makeData)
	not_definedf(:makeData)
else
	let
		N = 30
		res = makeData(N)
		if res isa Missing
			still_missing()
		elseif !(res isa Dict)
			keep_working(md"Make sure you return a `Dict`")
		elseif !issetequal(collect(keys(res)),[:β,:n,:X,:y,:dist])
			keep_working(md"your Dict needs to have keys [:β,:n,:X,:y,:dist]")
		elseif res[:β] != [ 1; 1.5; -0.5 ]
			keep_working(md"your :β entry needs to be equal to [ 1; 1.5; -0.5 ]")
		elseif res[:n] != N
			almost(md"check your `:n` key!")
		elseif mean(res[:y]) == 1
			almost(md"check your `:y` key! they are all equal to 1?")
		elseif !isa(res[:dist], Distributions.Normal)
			keep_working(md"make sure your `:dist` key holds a `Distributions.Normal` object")
		else
			correct()
		end
	end
end

# ╔═╡ 5293f91e-72b1-11eb-249f-db5a939d01dd
if !@isdefined(maximize_like)
	not_definedf(:maximize_like)
else
	let
		d0 = makeData(10)
		m = maximize_like()
		if m isa Missing
			still_missing()
		elseif !Optim.converged(m)
			keep_working(md"your optimizer has not converged")
		elseif norm(Optim.minimizer(m) .- d0[:β]) > 1e-1
			keep_working(md"your optimizer converged to the wrong solution")
		else
			correct()
		end
	end
end

# ╔═╡ f5e8cfaa-72ab-11eb-2945-292a1e1c639d
if !@isdefined(grad!)
	not_definedf(:grad!)
else
	let
		N = 3000
		d = makeData(N)
		g = ones(3)
		res = grad!(g,d[:β],d)
		
		if !isnothing(res)
			keep_working(md"the function should not return anything (i.e. return `nothing`)")
		elseif norm(g,	Inf) > 1e-1
			keep_working(md"your gradient is not approximately zero at the true values. something is wrong here!")
		else
			correct()
		end
	end
end

# ╔═╡ c11513b0-75ee-11eb-11c0-e18b91ed923c
if !@isdefined(maximize_like_grad)
	not_definedf(:maximize_like_grad)
else
	let
		d0 = makeData(10)
		m = maximize_like_grad()
		if m isa Missing
			still_missing()
		elseif !Optim.converged(m)
			keep_working(md"your optimizer has not converged")
		elseif norm(Optim.minimizer(m) .- d0[:β]) > 1e-1
			keep_working(md"your optimizer converged to the wrong solution")
		else
			correct()
		end
	end
end

# ╔═╡ 78038cd4-72af-11eb-2d21-d1f2158975bc
if !@isdefined(hessian!)
	not_definedf(:hessian!)
else
	let
		N = 3000
		d = makeData(N)
		h = ones(3,3)
		X0 = [0.8,1.0,-0.1]
		res = hessian!(h,X0,d)
		
		if !isnothing(res)
			keep_working(md"the function should not return anything (i.e. return `nothing`)")
		elseif maximum(abs.(Calculus.hessian(x -> loglik(x,d),X0) .-  h)) > 1e-5
			keep_working(md"your hessian is not correct! Try to check the algebra!")
		else
			correct()
		end
	end
end

# ╔═╡ 6c99cc08-75ef-11eb-1121-e93a30261298
if !@isdefined(maximize_like_grad_hess)
	not_definedf(:maximize_like_grad_hess)
else
	let
		d0 = makeData(10)
		m = maximize_like_grad_hess()
		if m isa Missing
			still_missing()
		elseif !Optim.converged(m)
			keep_working(md"your optimizer has not converged")
		elseif norm(Optim.minimizer(m) .- d0[:β]) > 1e-1
			keep_working(md"your optimizer converged to the wrong solution")
		else
			correct()
		end
	end
end

# ╔═╡ d83a3256-75f2-11eb-03d0-5bae3801946c
if !@isdefined(inv_observedInfo)
	not_definedf(:inv_observedInfo)
else
	let
		d0 = makeData()
		m = maximize_like_grad_hess()
		df = hcat(DataFrame(y=d0[:y]),convert(DataFrame,d0[:X]))
        gg = glm(@formula(y~x1+x2+x3),df,Binomial(),ProbitLink())
	    io = inv_observedInfo(m.minimizer,d0)
		
		if io isa Missing
			still_missing()
		else
			se1 = sqrt.(diag(io))		
			if maximum(abs.(se1 .- stderror(gg)[2:end])) > 1e-4
				keep_working(md"standard errors are not correct")
			else
				correct()
			end
		end
	end
end

# ╔═╡ Cell order:
# ╟─f4ee44c4-722c-11eb-3b10-2911aa3a1c84
# ╟─de94544c-722d-11eb-1240-713caa73862a
# ╠═f3634aac-7289-11eb-3a70-c5696ac59412
# ╠═895d2a70-722e-11eb-028a-5b2c44bdd141
# ╟─32f59f3c-728a-11eb-35bd-7352a92bf6b0
# ╟─a2482bac-722e-11eb-004c-9d02a807d23c
# ╟─f8b01012-722f-11eb-21c0-d3ef14765e11
# ╠═5275176a-728e-11eb-3ed8-3bce4485e601
# ╟─c65cfcbc-72b2-11eb-2b9e-9f1109e6d09d
# ╟─6a49a494-72b3-11eb-13d0-1bea0a985743
# ╠═7a2753d4-72b3-11eb-3eb0-c326168de29f
# ╟─547f08be-72b0-11eb-1dfa-f5caf3d2c9f2
# ╟─e11cda26-72b0-11eb-0ba9-759ecfc8a29b
# ╠═54144c70-72b2-11eb-350e-6bce5b0d0fc8
# ╟─2ee2330e-72b2-11eb-33ad-27efe91db68b
# ╠═d5feab24-72b0-11eb-22ec-e5c416a83a65
# ╟─5293f91e-72b1-11eb-249f-db5a939d01dd
# ╟─5f3c8cc0-7230-11eb-3da2-c139ef715daf
# ╟─990d630a-7232-11eb-2e53-a337a97012c8
# ╠═eca17258-72ab-11eb-08b5-f9a84a1aa986
# ╟─30d04bea-72b4-11eb-1d1a-59ac9614fdb4
# ╟─21ac2226-72b4-11eb-2010-f3c411dc32bf
# ╠═470c8f36-72b4-11eb-2db8-d982bc481397
# ╟─f5e8cfaa-72ab-11eb-2945-292a1e1c639d
# ╟─875d1ef2-72b2-11eb-1859-2b7ce0dab4c0
# ╟─e2e6fe3c-72b4-11eb-18a6-03ea4a69759f
# ╠═abbf948e-72b2-11eb-3044-b5de8fee0bf5
# ╟─c11513b0-75ee-11eb-11c0-e18b91ed923c
# ╟─a816dc78-7232-11eb-10ad-1be6fbd99abb
# ╠═fc476bc6-72ac-11eb-27b7-d9e4adac8a1f
# ╠═5d0d9492-72af-11eb-2530-078dc03f6c10
# ╟─78038cd4-72af-11eb-2d21-d1f2158975bc
# ╟─b053bc9e-72b4-11eb-0b8e-ffc0479fea16
# ╠═c8742db8-72b4-11eb-24be-b1363f3f4bcf
# ╟─6c99cc08-75ef-11eb-1121-e93a30261298
# ╟─f0e913d8-75ef-11eb-34e1-1175cbacafd8
# ╟─f6a46804-75f4-11eb-0aae-7de5394c4864
# ╟─f40e1210-75f2-11eb-0391-09eaa939ac85
# ╠═aed8965e-75f3-11eb-35af-df22bcc9c11c
# ╟─92f15dac-75f5-11eb-28b1-47b33da031c5
# ╠═bf01e018-75f2-11eb-0cec-8957850d2db9
# ╟─d83a3256-75f2-11eb-03d0-5bae3801946c
# ╟─64995e54-75f6-11eb-39c7-61406ddc7a6e
# ╠═7a069524-75f4-11eb-34b7-db06fac6d646
# ╟─9d49c7fc-75fb-11eb-3f8a-e355f688ea97
# ╟─8f0b6128-7289-11eb-1e54-0930af02a28e
# ╟─94ef2d92-7289-11eb-2293-3f0c04f52e93
# ╟─9606a958-7289-11eb-32f9-eb8b2deaf1f2
# ╟─9a4edf28-7289-11eb-3773-ab2ba97acc11
# ╟─9ebe21d4-7289-11eb-3197-b969d3a402ce
# ╟─a32fcc86-7289-11eb-3134-8b5e7a91dbda
# ╟─a843de22-7289-11eb-21c4-cb0ecb4d5aad
# ╟─b1b5ebe6-728e-11eb-1865-9f63880d9671
