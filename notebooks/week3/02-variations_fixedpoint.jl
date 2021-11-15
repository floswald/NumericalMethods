### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 03a56358-5658-11eb-2c13-b75fae8371cc
begin
	# best style
	using NLsolve

	p = 1.0
	β = 0.9
	f(v) = p .+ β * v # broadcast the +
	sol = fixedpoint(f, [0.8])  # NLsolve.fixedpoint
end

# ╔═╡ 75d25868-5659-11eb-1454-838e4783c86b
using StaticArrays

# ╔═╡ b0105e24-6ac4-11eb-24d1-4f1e7812b751
html"<button onclick='present()'>present</button>"

# ╔═╡ 495e8658-5654-11eb-378f-990600aefeee
md"""
## Example: Variations on Fixed Points

Take a mapping $f : X \to X$ for some set $X$

If there exists an $X^* \in X$ such that $f(x^*) = x^*$, then $x^*$ is called a “fixed point” of $f$.

For this example, we will start with a simple example of determining fixed points of a function

The goal is to start with code in a MATLAB style, and move towards a more **Julian** style with high mathematical clarity.
"""

# ╔═╡ bb261a40-5ff9-11eb-1209-4536480658d3
md"
in order to run this notebook, you need to activate the the project file in this week's [folder](https://github.com/floswald/NumericalMethods/blob/master/lecture_notebooks/week2/Project.toml)"

# ╔═╡ 8012ff7a-565a-11eb-0fa4-9f5c4bd09204
md"

#

Let's start with a basic `while` loop. 

* Set starting values and tolerance levels
* while successive guesses are *far* from each other, and we haven't exceeded the iterations limit, 
* apply the `f` map
* recompute the distance between old and new guess
* update old -> new
"

# ╔═╡ 380135f8-5655-11eb-2845-71dad8c3acdd
fpprinter(v,diff,iter) = md"Fixed point = $(round(v,digits=6)), and |f(x) - x| = $(round(diff,digits = 10)) in $iter iterations"

# ╔═╡ 5d49e75c-5654-11eb-1808-2d45b1bb8c3d
let
	using LinearAlgebra  # for norm() fnction
	# poor style
	p = 1.0 # note 1.0 rather than 1
	β = 0.9
	maxiter = 1000
	tolerance = 1.0E-7
	v_iv = 0.8 # initial condition

	# setup the algorithm
	v_old = v_iv
	normdiff = Inf
	iter = 1
	while normdiff > tolerance && iter <= maxiter
		v_new = p + β * v_old # the f(v) map
		normdiff = norm(v_new - v_old)  # "size" of a vector in some space

		# replace and continue
		v_old = v_new
		iter = iter + 1
	end
	# md"Fixed point = $v_old, and |f(x) - x| = $normdiff in $iter iterations"
	fpprinter(v_old,normdiff,iter)
end

# ╔═╡ d0002c4c-5655-11eb-30a1-5b7e6d56d14a
md"""

#

We use a `let` block to define a local scope in those examples. (The variables are local to that block).

Here, we have used the `norm` function (from the `LinearAlgebra` base library) to compare the values

The other new function is the `println` in the `fpprinter`, with the string interpolation, which splices the value of an expression or variable prefixed by `$` into a string

An alternative approach is to use a `for` loop, and check for convergence in each iteration
"""

# ╔═╡ a2a32222-5655-11eb-1a46-919ef9ae613a
let
	p = 1.0 # note 1.0 rather than 1
	β = 0.9
	maxiter = 1000
	tolerance = 1.0E-7
	v_iv = 0.8 # initial condition
	v_old = v_iv
	normdiff = Inf
	iter = 1
	for i in 1:maxiter
		v_new = p + β * v_old # the f(v) map
		normdiff = norm(v_new - v_old)
		if normdiff < tolerance # check convergence
			iter = i
			break # converged, exit loop
		end
		# replace and continue
		v_old = v_new
	end
	fpprinter(v_old,normdiff,iter)
end

# ╔═╡ fec76306-5655-11eb-2ca5-ef280da95e08
md"""
# Using a Function

The first problem with this setup is that it depends on being sequentially run – which can be easily remedied with a function. In general, you should always write functions by default.
"""

# ╔═╡ 28c4ee50-5656-11eb-148a-03a76814ee31
# our function definition here
# still poor style
function v_fp(β, ρ, v_iv, tolerance, maxiter)
    # setup the algorithm
    v_old = v_iv
    normdiff = Inf
    iter = 1
    while normdiff > tolerance && iter <= maxiter
        v_new = ρ + β * v_old # the f(v) map
        normdiff = norm(v_new - v_old)

        # replace and continue
        v_old = v_new
        iter = iter + 1
    end
    return (v_old, normdiff, iter) # returns a tuple
end

# ╔═╡ eb26528e-6ac4-11eb-141b-412543bf2596
md"
#
"

# ╔═╡ 33fcd416-5656-11eb-1e5e-57d0c9d0bdc3
let
	p = 1.0 # note 1.0 rather than 1
	β = 0.9
	maxiter = 1000
	tolerance = 1.0E-7
	v_initial = 0.8 # initial condition
	v_star, normdiff, iter = v_fp(β, p, v_initial, tolerance, maxiter)
end

# ╔═╡ 3da13308-5657-11eb-2e7d-c3d2cf988e2a
md"""
# Passing a Function

The main issue is that the algorithm (finding a fixed point) is reusable and generic, while the function we calculate `p + β * v` is specific to our problem

A key feature of languages like Julia, is the ability to efficiently handle functions passed to other functions
"""

# ╔═╡ 233dbc6c-565b-11eb-22f2-6beff3fc5ec7
function fixedpointmap(f, iv, tolerance, maxiter)
	# setup the algorithm
	x_old = iv
	normdiff = Inf
	iter = 1
	while normdiff > tolerance && iter <= maxiter
		x_new = f(x_old) # use the passed in map
		normdiff = norm(x_new - x_old)
		x_old = x_new
		iter = iter + 1
	end
	return (x_old, normdiff, iter)
end

# ╔═╡ fbf0b698-6ac4-11eb-07f4-91e0c6b201b6
md"
#"


# ╔═╡ 8c42759e-5657-11eb-3aa8-49c357238ae2
md"""
Much better (closer to math), but there are still hidden bugs if the user orders the settings or returns types wrong

# Named Arguments and Return Values

To enable this, Julia has two features:  named function parameters, and named tuples
"""

# ╔═╡ cd05b122-5657-11eb-06c7-ff76b80fb543

	# good style
	function fixedpointmap(f; iv, tolerance=1E-7, maxiter=1000)
		# setup the algorithm
		x_old = iv
		normdiff = Inf
		iter = 1
		while normdiff > tolerance && iter <= maxiter
			x_new = f(x_old) # use the passed in map
			normdiff = norm(x_new - x_old)
			x_old = x_new
			iter = iter + 1
		end
		return (value = x_old, normdiff=normdiff, iter=iter) # A named tuple
	end


# ╔═╡ 4647c698-5657-11eb-2d66-ad8c036a9323
let
	# define a map and parameters
	f(v) = p + β * v # note that p and β are used in the function!
	
	p = 1.0 # note 1.0 rather than 1
	β = 0.9
	maxiter = 1000
	tolerance = 1.0E-7
	v_initial = 0.8 # initial condition

	v_star, normdiff, iter = fixedpointmap(f, v_initial, tolerance, maxiter)
end

# ╔═╡ 0a37c2dc-6ac5-11eb-36fd-e7e675113948
md"
#
"

# ╔═╡ 9613922e-5657-11eb-37e7-5df733b7121f
let
	# define a map and parameters
	p = 1.0
	β = 0.9
	f(v) = p + β * v # note that p and β are used in the function!

	sol = fixedpointmap(f, iv=0.8, tolerance=1.0E-8) # don't need to pass
end

# ╔═╡ b98e3484-5657-11eb-3ce4-6fd123e122d3
md"""
#

In this example, all function parameters after the `;` in the list, must be called **by name**.

Furthermore, a default value may be enabled – so the named parameter `iv` is required while `tolerance` and `maxiter` have default values.

The return type of the function also has named fields, `value, normdiff,` and `iter` – all accessed intuitively using `.`

To show the flexibilty of this code, we can use it to find a fixed point of the non-linear logistic equation, $ x = f(x) $ where $ f(x) := r x (1-x) $
"""

# ╔═╡ bfc3df5c-5657-11eb-1f6f-7bf36b96aa97
let
	r = 2.0
	f(x) = r * x * (1 - x)
	sol = fixedpointmap(f, iv=0.8)
end

# ╔═╡ f41f568c-5657-11eb-2130-5b3596edecf5
md"""
# Using a Package

But best of all is to avoid writing code altogether
"""

# ╔═╡ 9d18c958-5658-11eb-22d9-99bbfc6c21fd
md"""
# Benefits of Generic Code

The above example can be extended to multivariate maps without *any* modifications to the fixed point iteration code we wrote. Suppose we had a *bivariate* map instead. We could use our own `fixedpointmap` function:

"""		

# ╔═╡ bae8e210-5658-11eb-0795-ebba7a2e2966
let
	p = [1.0, 2.0]
	β = 0.9
	iv = [0.8, 2.0]
	f(v) = p .+ β * v # note that p and β are used in the function!

	sol = fixedpointmap(f, iv = iv, tolerance = 1.0E-8)
end

# ╔═╡ e1493180-5658-11eb-2f0c-e7b3f5eb7f73
md"
#

Also with the `NLsolve` library function! take a 3 dimensional map now!
"

# ╔═╡ f01aa778-5658-11eb-25df-a9705636e545
let
	p = [1.0, 2.0, 0.1]
	β = 0.9
	iv =[0.8, 2.0, 51.0]
	f(v) = p .+ β * v

	sol = fixedpoint(v -> p .+ β * v, iv)
end

# ╔═╡ 1f666712-5659-11eb-3e47-4b45164831ec
md"""
# Composability

Finally, to demonstrate the importance of composing different libraries, use a [`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl) type, which provides an efficient implementation for small arrays and matrices. We can just **swap** the kind or array that we are using, because our code is generic enough. 

Notice that the same idea carries over to arrays *which are stored and worked upon on different devices* - like a GPU card for example. 
"""

# ╔═╡ 63403e4a-5659-11eb-3598-8be6ac60a2bc
let
	p = @MVector [1.0, 2.0, 0.1]
	β = 0.9
	iv = @MVector [0.8, 2.0, 51.0]
	f(v) = p .+ β * v

	sol = fixedpoint(v -> p .+ β * v, iv)
end

# ╔═╡ Cell order:
# ╠═b0105e24-6ac4-11eb-24d1-4f1e7812b751
# ╟─495e8658-5654-11eb-378f-990600aefeee
# ╟─bb261a40-5ff9-11eb-1209-4536480658d3
# ╟─8012ff7a-565a-11eb-0fa4-9f5c4bd09204
# ╠═5d49e75c-5654-11eb-1808-2d45b1bb8c3d
# ╟─380135f8-5655-11eb-2845-71dad8c3acdd
# ╟─d0002c4c-5655-11eb-30a1-5b7e6d56d14a
# ╠═a2a32222-5655-11eb-1a46-919ef9ae613a
# ╟─fec76306-5655-11eb-2ca5-ef280da95e08
# ╠═28c4ee50-5656-11eb-148a-03a76814ee31
# ╟─eb26528e-6ac4-11eb-141b-412543bf2596
# ╠═33fcd416-5656-11eb-1e5e-57d0c9d0bdc3
# ╟─3da13308-5657-11eb-2e7d-c3d2cf988e2a
# ╠═233dbc6c-565b-11eb-22f2-6beff3fc5ec7
# ╟─fbf0b698-6ac4-11eb-07f4-91e0c6b201b6
# ╠═4647c698-5657-11eb-2d66-ad8c036a9323
# ╟─8c42759e-5657-11eb-3aa8-49c357238ae2
# ╠═cd05b122-5657-11eb-06c7-ff76b80fb543
# ╟─0a37c2dc-6ac5-11eb-36fd-e7e675113948
# ╠═9613922e-5657-11eb-37e7-5df733b7121f
# ╟─b98e3484-5657-11eb-3ce4-6fd123e122d3
# ╠═bfc3df5c-5657-11eb-1f6f-7bf36b96aa97
# ╟─f41f568c-5657-11eb-2130-5b3596edecf5
# ╠═03a56358-5658-11eb-2c13-b75fae8371cc
# ╟─9d18c958-5658-11eb-22d9-99bbfc6c21fd
# ╠═bae8e210-5658-11eb-0795-ebba7a2e2966
# ╟─e1493180-5658-11eb-2f0c-e7b3f5eb7f73
# ╠═f01aa778-5658-11eb-25df-a9705636e545
# ╟─1f666712-5659-11eb-3e47-4b45164831ec
# ╠═75d25868-5659-11eb-1454-838e4783c86b
# ╠═63403e4a-5659-11eb-3598-8be6ac60a2bc
