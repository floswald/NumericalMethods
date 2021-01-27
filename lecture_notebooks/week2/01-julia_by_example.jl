### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ cda37261-310a-43b3-b043-fbf0769ca73d
begin
	using Distributions
	
	function plothistogram(distribution, n)
	    ϵ = rand(distribution, n)  # n draws from distribution
	    histogram(ϵ)
	end
	
	lp = Laplace()  # an "instance" of a Distribution type. More on this later!
	plothistogram(lp, 500)
end

# ╔═╡ f52b4210-d4ff-496e-aafb-f9b677f00b2c
md"""
# Julia By Example

In this notebook we will start using julia on some very simple examples to get you going. You will need to get familiar with the excellent [Julia documentation](https://docs.julialang.org/en/v1/) on your own and explore the language.
"""

# ╔═╡ 2f6e9583-c4af-446a-838e-919dc120c18e
md"""
## Example 1: Plotting a White Noise Process

To begin, let’s suppose that we want to simulate and plot the white noise
process $\epsilon_0, \epsilon_1, \ldots, \epsilon_T$, where each draw $\epsilon_t$ is independent standard normal.
"""

# ╔═╡ 56f37478-c59b-4bdf-8bee-aa339cba2381
md"""
### Using Functions from a Package

Some functions are built into the base Julia, such as `randn`, which returns a single draw from a normal distibution with mean 0 and variance 1 if given no parameters:
"""

# ╔═╡ ec460128-63fa-496c-9b41-58c3a406d8d4
randn()

# ╔═╡ 7f31f8ec-564e-11eb-03ec-d90906a4ed54
md"
* For other functionality we rely on external *packages*. Generally, we use the keyword `using xyz` to load the content of the `xyz` package into the current scope.
* It is a trickier issue than you might think as to **which version** of each package to use. 
* In contrast to proprietary software such as Matlab, where all toolboxes and add-ons are designed to work within a current release, in open source projects this is not enforceable.
* Developers change their packages at their own pace. Most of them try of course to be up to date with the latest official julia release.
* A complication is that many packages have interdependencies - they build upon each other.
* It may happen that a certain (new) feature of an external package is compatible only with a certain version `v.x.y.z` of another package.
* We will talk about julia's package manager, which is a great tool to tackle this issue.
* For now, we will just create clean **environment** where we install the latest version of packages into to make sure this works for everyone.
* Going forward, you could just install those packages into the global environment. We'll talk about that. Promised.

For example, we'll want to use the the [`Plots.jl`](https://github.com/JuliaPlots/Plots.jl) and the [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) packages today:
"

# ╔═╡ d89d5372-564e-11eb-2280-db78011ddc62
# first let's create a clean environment for our packages
# we will talk about the package manager in detail later on.
begin
	import Pkg
	Pkg.activate(mktempdir())
end

# ╔═╡ ea46e48a-564e-11eb-0372-af8627ab932c
# then we add packages needed for this notebook
begin
	Pkg.add(["Plots","Distributions"])
	using Plots  # and we load Plots right away
end

# ╔═╡ 46d75658-564f-11eb-1224-91767a319f2c
n = 100

# ╔═╡ 4ef2d0c4-564f-11eb-3ccc-e53cc3bb4a9d
ϵ = randn(n)

# ╔═╡ 52650650-564f-11eb-10ac-676826c30101
plot(1:n, ϵ)

# ╔═╡ 29b8d03a-5302-4b46-a63f-5957134c69ff
md"""
Let’s break this down and see how it works:

* The effect of the statement `using Plots` is to make all the names exported by the `Plots` module available
* The arguments to `plot` are the numbers `1,2, ..., n` for the x-axis, a vector `ϵ` for the y-axis, and (optional) settings
* The function `randn(n)` returns a column vector `n` random draws from a normal distribution with mean 0 and variance 1
"""

# ╔═╡ c7bef754-5b21-4e1f-892c-43781f3f54ad
md"""
### Arrays

As a language intended for mathematical and scientific computing, Julia has
strong support for using unicode characters

In the above case, the `ϵ` and many other symbols can be typed in most Julia editor by providing the LaTeX and `<TAB>`, i.e. `\epsilon<TAB>`

The return type is one of the most fundamental Julia data types: an array
"""

# ╔═╡ 0d7bbe75-7675-47f0-b779-cf2a0fca5d47
typeof(ϵ)

# ╔═╡ 2a48e20c-763e-4242-8094-d0d0540df937
ϵ[1:5]

# ╔═╡ 4dc06b06-f980-454f-b3ef-798ee19d9807
md"""
### For Loops

Although there’s no need in terms of what we wanted to achieve with our
program, for the sake of learning syntax let’s rewrite our program to use a
`for` loop for generating the data

Starting with the most direct version, and pretending we are in a world where `randn` can only return a single value
"""

# ╔═╡ 6eaa2905-4200-4665-817d-21f826c61c17
begin
	# poor style
	for i in 1:n
	    ϵ[i] = randn()
	end
	plot(1:n, ϵ)
end

# ╔═╡ b1e96444-8721-4784-870a-77ccb39535fd
md"""
* Here we first declared `ϵ` to be a vector of `n` numbers, initialized by the floating point `0.0`
* The `for` loop then populates this array by successive calls to `randn()`
* Like all code blocks in Julia, the end of the `for` loop code block (which is just one line here) is indicated by the keyword `end`
* The word `in` from the `for` loop can be replaced by either `∈` or `=`
* The index variable is looped over for all integers from `1:n` – but this does not actually create a vector of those indices
* Instead, it creates an **iterator** that is looped over – in this case the **range** of integers from `1` to `n`
* While this example successfully fills in `ϵ` with the correct values, it is very indirect as the connection between the index `i` and the `ϵ` vector is unclear
* To fix this, use `eachindex`
"""

# ╔═╡ 35b657f7-19f2-40aa-96aa-70078c6a9534
begin
	# better style
	for i in eachindex(ϵ)
	    ϵ[i] = randn()
	end
	plot(1:n, ϵ)
end

# ╔═╡ 280bed35-5ee9-405f-8595-f5649d4c7864
md"""
Here, `eachindex(ϵ)` returns an iterator of indices which can be used to access `ϵ`

While iterators are memory efficient because the elements are generated on the fly rather than stored in memory, the main benefit is (1) it can lead to code which is clearer and less prone to typos; and (2) it allows the compiler flexibility to creatively generate fast code

In Julia you can also loop directly over arrays themselves, like so
"""

# ╔═╡ efee794e-c36e-4b97-89d3-21beee6152f6
md"""
### User-Defined Functions

For the sake of the exercise, let’s go back to the `for` loop but restructure our program so that generation of random variables takes place within a user-defined function

To make things more interesting, instead of directly plotting the draws from the distribution, let’s plot the squares of these draws.
"""

# ╔═╡ 3dec8bc3-509e-40ad-bee3-3f5610871d50
begin
	# still poor style
	
	function generatedata(n)
	    ϵ = zeros(n)
	    for i in eachindex(ϵ)
	        ϵ[i] = (randn())^2 # squaring the result
	    end
	    return ϵ
	end
	
	data = generatedata(10)  # create the data
	plot(data)
end

# ╔═╡ 1feb7c9e-db01-4690-88a2-6701641267ad
md"""
Here

- `function` is a Julia keyword that indicates the start of a function definition  
- `generatedata` is an arbitrary name for the function  
- `return` is a keyword indicating the return value, as is often unnecessary  


Let us make this example slightly better by “remembering” that `randn` can return a vectors
"""

# ╔═╡ 46c2c1e0-5147-476d-9227-5595f87049f9
begin
	# still poor style
	function generatedata2(n)
	    ϵ = randn(n) # use built in function
	
	    for i in eachindex(ϵ)
	        ϵ[i] = ϵ[i]^2 # squaring the result
	    end
	
	    return ϵ
	end
	data2 = generatedata2(5)
	plot(data2)

end

# ╔═╡ fd9a3c53-3cb2-45a5-a012-572bb29b0a7c
md"""
### Broadcasting: the dot `.`

While better, the looping over the `i` index to square the results is difficult to read

Instead of looping, we can **broadcast** the `^2` square function over a vector using a `.`

To be clear, unlike Python, R, and MATLAB (to a lesser extent), the reason to drop the `for` is **not** for performance reasons, but rather because of code clarity

Loops of this sort are at least as efficient as vectorized approach in compiled languages like Julia, so use a for loop if you think it makes the code more clear
"""

# ╔═╡ f175afdb-afb0-40ae-b8b5-09efe17f5e47
begin
	# better style
	function generatedata3(n)
	    ϵ = randn(n) # use built in function
	    return ϵ.^2
	 end
	data3 = generatedata3(5)
end

# ╔═╡ bdb56cea-4017-4ce5-8337-ad78cc054bf9
md"""
We can even drop the `function` if we define it on a single line
"""

# ╔═╡ b47202da-016a-4fe8-b1f8-2cc17d4d0dd6
begin
	# We can even drop the `function` if we define it on a single line
	# good style
	generatedata4(n) = randn(n).^2
	data4 = generatedata4(5)
end

# ╔═╡ 641075db-e18a-4a02-8d0e-3b95140b7657
md"""
* Finally, we can broadcast *any* function! 
* The function `^()` is only one case.
* We can broadcast your own user-defined functions as well.
"""

# ╔═╡ 9e9961a9-dff1-43e0-a215-506e09396bd9
md"""
As a final – abstract – approach, we can make the `generatedata` function able to generically apply to a function
"""

# ╔═╡ 132e2ef1-434f-4daf-813a-ac13004d94d0
md"""
Whether this example is better or worse than the previous version depends on how it is used

High degrees of abstraction and generality, e.g. passing in a function `f` in this case, can make code either clearer or more confusing, but Julia enables you to use these techniques **with no performance overhead**

For this particular case, the clearest and most general solution is probably the simplest
"""

# ╔═╡ 20210cb7-355b-4f13-b96f-ae2ada076ae0
begin
	# direct solution with broadcasting, and small user-defined function
	x = randn(n)
	plot(f.(x), label="x^2")
	plot!(x, label="x") # layer on the same plot
end

# ╔═╡ ac18f9c1-7ae7-496c-a879-92f7bf0da5b4
md"""
While broadcasting above superficially looks like vectorizing functions in MATLAB, or Python ufuncs, it is much richer and built on core foundations of the language

The other additional function `plot!` adds a graph to the existing plot

This follows a general convention in Julia, where a function that modifies the arguments or a global state has a `!` at the end of its name
"""

# ╔═╡ dd6fd112-53fc-410f-ab3c-63e9bcd4fad3
md"""
#### A Slightly More Useful Function

Let’s make a slightly more useful function

This function will be passed in a choice of probability distribution and respond by plotting a histogram of observations.

Here’s the code:
"""

# ╔═╡ 53a72a3f-e3ea-46a7-bae3-09bc8e5aa548
md"""
Let’s have a casual discussion of how all this works while leaving technical details for later in the lectures:

1. `lp = Laplace()` creates an instance of a data type defined in the `Distributions` module that represents the Laplace distribution
2. The name `lp` is bound to this value
3. When we make the function call `plothistogram(lp, 500)` the code in the body of the function `plothistogram` is run with
    - the name `distribution` bound to the same value as `lp`  
    - the name `n` bound to the integer `500`  
"""

# ╔═╡ 066f43af-814d-40ac-86bb-f1bfec6f55c9
md"""
### Here's a Mystery

Now consider the function call `rand(distribution, n)`:

* This looks like something of a mystery, doesn't it?
* The function `rand()` is defined in the base library such that `rand(n)` returns `n` uniform random variables on $[0, 1)$:
"""

# ╔═╡ df0d1fe0-2127-485d-95c1-a6563c8f13a3
rand(3)

# ╔═╡ 9ea969cc-34d6-4d4b-b8fb-12739b6be3cf
md"""
On the other hand, we just heard that `distribution` points to a data type representing the Laplace distribution that has been defined in a third party package.

>So how can it be that `rand()` is able to take this kind of value as an argument and return the output that we want?

The answer in a nutshell is **multiple dispatch**, which Julia uses to implement **generic programming**.

* This refers to the idea that functions in Julia can have different behavior depending on the particular arguments that they’re passed.
* Hence in Julia we can take an existing function and give it a new behavior by defining how it acts on a new type of value
* The compiler knows which function definition to apply to in a given setting by looking at the types of the values the function is called on
* In Julia these alternative versions of a function are called **methods**
"""

# ╔═╡ 8f301ff9-fbeb-41c5-b265-533630649025
begin
	# good style
	square(x) = x^2 # our own simple square function
	generatedata6(n) = square.(randn(n)) # uses broadcast for some function `f`
	data6 = generatedata6(5)
end

# ╔═╡ 9e4cd30a-9dc2-4e3f-8d4a-6cf1ca543bcb
begin
	generatedatax(n, gen) = gen.(randn(n)) # uses broadcast for some function `gen`
	square(x) = x^2 # our own simple square function
	datax = generatedatax(5, square) # applies square
end

# ╔═╡ Cell order:
# ╟─f52b4210-d4ff-496e-aafb-f9b677f00b2c
# ╟─2f6e9583-c4af-446a-838e-919dc120c18e
# ╟─56f37478-c59b-4bdf-8bee-aa339cba2381
# ╠═ec460128-63fa-496c-9b41-58c3a406d8d4
# ╟─7f31f8ec-564e-11eb-03ec-d90906a4ed54
# ╠═d89d5372-564e-11eb-2280-db78011ddc62
# ╠═ea46e48a-564e-11eb-0372-af8627ab932c
# ╠═46d75658-564f-11eb-1224-91767a319f2c
# ╠═4ef2d0c4-564f-11eb-3ccc-e53cc3bb4a9d
# ╠═52650650-564f-11eb-10ac-676826c30101
# ╟─29b8d03a-5302-4b46-a63f-5957134c69ff
# ╟─c7bef754-5b21-4e1f-892c-43781f3f54ad
# ╠═0d7bbe75-7675-47f0-b779-cf2a0fca5d47
# ╠═2a48e20c-763e-4242-8094-d0d0540df937
# ╟─4dc06b06-f980-454f-b3ef-798ee19d9807
# ╠═6eaa2905-4200-4665-817d-21f826c61c17
# ╟─b1e96444-8721-4784-870a-77ccb39535fd
# ╠═35b657f7-19f2-40aa-96aa-70078c6a9534
# ╟─280bed35-5ee9-405f-8595-f5649d4c7864
# ╟─efee794e-c36e-4b97-89d3-21beee6152f6
# ╠═3dec8bc3-509e-40ad-bee3-3f5610871d50
# ╟─1feb7c9e-db01-4690-88a2-6701641267ad
# ╠═46c2c1e0-5147-476d-9227-5595f87049f9
# ╟─fd9a3c53-3cb2-45a5-a012-572bb29b0a7c
# ╠═f175afdb-afb0-40ae-b8b5-09efe17f5e47
# ╟─bdb56cea-4017-4ce5-8337-ad78cc054bf9
# ╠═b47202da-016a-4fe8-b1f8-2cc17d4d0dd6
# ╟─641075db-e18a-4a02-8d0e-3b95140b7657
# ╠═8f301ff9-fbeb-41c5-b265-533630649025
# ╟─9e9961a9-dff1-43e0-a215-506e09396bd9
# ╠═9e4cd30a-9dc2-4e3f-8d4a-6cf1ca543bcb
# ╟─132e2ef1-434f-4daf-813a-ac13004d94d0
# ╠═20210cb7-355b-4f13-b96f-ae2ada076ae0
# ╟─ac18f9c1-7ae7-496c-a879-92f7bf0da5b4
# ╟─dd6fd112-53fc-410f-ab3c-63e9bcd4fad3
# ╠═cda37261-310a-43b3-b043-fbf0769ca73d
# ╟─53a72a3f-e3ea-46a7-bae3-09bc8e5aa548
# ╠═066f43af-814d-40ac-86bb-f1bfec6f55c9
# ╠═df0d1fe0-2127-485d-95c1-a6563c8f13a3
# ╟─9ea969cc-34d6-4d4b-b8fb-12739b6be3cf
