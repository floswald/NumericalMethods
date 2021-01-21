### A Pluto.jl notebook ###
# v0.12.12

using Markdown
using InteractiveUtils

# ╔═╡ d0cb1cea-51fa-11eb-3097-f1b352770dbf
md"
# The Julia-is-Fast Benchmark Fun: `sum`

In this notebook we are going to compare performance of a very simple function across several different languages: The `sum`. This function computes

$\text{sum}(a) = \sum_{i=1}^n a_i$

where $n$ is the length of vector `a`. (This notebook started life in a [version](https://github.com/mitmath/18S096/blob/master/lectures/lecture1/Boxes-and-registers.ipynb) published by [Steven Johnson](https://math.mit.edu/~stevenj/)).

Let's get a vector of 10 million random numbers:
"

# ╔═╡ 794c7764-51fc-11eb-2325-91421e8833e3
a = rand(10_000_000) # 1 million random numbers, uniform on [0,1)

# ╔═╡ 1837b192-55de-11eb-319d-2f4e3205a3d7
begin
	using PyCall
	
	apy_list = PyCall.array2py(a)

	# get the Python built-in "sum" function:
	pysum = pybuiltin("sum")
end

# ╔═╡ 8a1db652-51fc-11eb-191c-af8d4141c316
md"What's the sum of `a` now?"

# ╔═╡ 9be2d85c-51fc-11eb-1770-e3ee78651c62
sum(a)  # the expected value of each draw is 0.5 - so 0.5 * 1 million = 5000000

# ╔═╡ ee1de4d8-51fc-11eb-1d5f-119571ab06a5
md"
## Benchmarking in different Languages

Let's use the BenchmarkTools.jl package for this.
"

# ╔═╡ fef98014-51fc-11eb-15ad-d17044bcab51
# let's create a package environment
begin
	import Pkg
	Pkg.activate(mktempdir())
end

# ╔═╡ 488c605c-51fd-11eb-1d04-61c8684822e7
# and add all packages we need to it
begin
	Pkg.add(["BenchmarkTools", "PyCall", "Conda", "Plots","RCall", "DataFrames"])
	using BenchmarkTools
end

# ╔═╡ 91c95162-51fd-11eb-0eac-e1df447f55dd
md"
C is often considered the gold standard: difficult on the human, nice for the machine. Getting within a factor of 2 of C is often satisfying. Nonetheless, even within C, there are many kinds of optimizations possible that a naive C writer may or may not get the advantage of.

We can write the C code in a string, and hand it from the julia session to a C compiler. Then we can call the compiled function directly. Notice that the C function will accept only a single data type, i.e. `double` , which is `Float64` in julia: 
"

# ╔═╡ a3087dd6-51fd-11eb-1a23-fd0a56e35457
C_code = """
#include <stddef.h>
double c_sum(size_t n, double *X) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        s += X[i];
    }
    return s;
}
"""

# ╔═╡ aa655158-51fd-11eb-3014-9560d3c32afc
# we call my gcc compiler to compile the C_code
begin
	const Clib = tempname()   # make a temporary file
	using Libdl
	open(`gcc  -fPIC -O3 -ffast-math -msse3 -xc -shared -o $(Clib * "." * Libdl.dlext) -`, "w") do f
    	print(f, C_code) 
	end
end
	

# ╔═╡ 0258a7e8-51fe-11eb-3705-497c4e1ec940
# define a Julia function that calls the C function:
c_sum(X::Array{Float64}) = ccall(("c_sum", Clib), Float64, (Csize_t, Ptr{Float64}), length(X), X)

# ╔═╡ 06df6446-51fe-11eb-12a1-e7ff18534536
c_sum(a)

# ╔═╡ 05a3d930-55dd-11eb-144e-2d5058bd89c1
md"wait, is that the same as the julia result?"

# ╔═╡ 10a3b738-55dd-11eb-154b-0f2b4d155753
c_sum(a) == sum(a)

# ╔═╡ 18a3f560-55dd-11eb-1fed-4b5dc0b96df6
md"No?! How come? We are summing over the same numbers! 🤔

What's going on here is the *order* of summation matters in floating point arithmetic. Notice that the computer can represent a *number* only up to some precision (i.e. up to a certain number of digits). then, how you round those numbers of, and in which order you sum them up, starts to matter for the result.

So, we'll look at whether those are *approximately* equal:
"

# ╔═╡ 0f449a7c-51fe-11eb-2c20-bfb771a8b00c
c_sum(a) ≈ sum(a) # type \approx and then <TAB> to get the ≈ symbol

# ╔═╡ 76d0693e-55dd-11eb-10c5-1b6930bfee9d
md"
Now lets run the Benchmark Trial!
"

# ╔═╡ 166fb3de-51fe-11eb-0b84-6b4126f44d0e
c_bench = @benchmark c_sum($a)

# ╔═╡ e0e3ca6e-55dd-11eb-0e94-f33e3d61a4d2
begin
	using Plots

	t = c_bench.times / 1e6 # times in milliseconds
	using Statistics
	m, σ = minimum(t), std(t)

	histogram(t, bins=500,
    xlim=(m - 0.01, m + σ),
    xlabel="milliseconds", ylabel="count", label="")
end

# ╔═╡ a66b3930-55dd-11eb-2e47-a70a33b7fd9a
md"
Let's collect the results in a `Dict`:
"

# ╔═╡ b506dd50-55dd-11eb-2e38-87fd5ce241dc
begin
	d = Dict()  # a "dictionary", i.e. an associative array
	d[:C] = minimum(c_bench.times) / 1e6  # in milliseconds
	d
end

# ╔═╡ bc7080ca-55e5-11eb-0508-bf469f549365
begin
	using RCall

	r_bench = @benchmark R"sum($a)"
	d[:R] = minimum(r_bench.times) / 1e6
end

# ╔═╡ fed8b4f0-55e5-11eb-2cc2-990221ad2789
begin
	using DataFrames
	sort!(DataFrame(language = collect(keys(d)), time = collect(values(d))), :time)
end

# ╔═╡ cafc00f4-55dd-11eb-24d5-85c5bf3cb9b0
md"
We can see above that the BenchmarkTools library takes many sample runs to account for machine noise in the benchmark. We can look at the distribution of times:
"

# ╔═╡ f6ad1ecc-55dd-11eb-00ce-692fdfc74f81
md"
## Python: built-in `sum` function

Call a low-level PyCall function to get a Python list, because by default PyCall will convert to a NumPy array instead (we benchmark NumPy below):
"

# ╔═╡ 32002adc-55de-11eb-241c-7780d36ec5d0
md"
now we have the python function `pysum` available in our julia session. is it giving the correct resuls?
"

# ╔═╡ 437ebce2-55de-11eb-3ab7-85568c871c0a
begin
	
	pysum(a)

	pysum(a) ≈ sum(a)
	
	# benchmark it!
	py_list_bench = @benchmark $pysum($apy_list)

	d[:python] = minimum(py_list_bench.times) / 1e6
	d
end

# ╔═╡ 0ceaa1ae-55e1-11eb-2b16-ed63d0f5f752
begin
	relc(x,d) = round(d[x]/d[:C],digits = 1);
	rely(x,y,d) = round(d[x]/d[y],digits = 1);
end

# ╔═╡ accb0914-55de-11eb-1d8e-13664efdd168
md"
Wowzer! That's $(relc(:python,d)) times slower than C! And this even though the python `sum` is [*written* in C](https://github.com/python/cpython/blob/3.7/Python/bltinmodule.c#L2314-L2479)! 

How is this possible? It comes from python being generic, i.e. not specialised to certain data types at all. Whatever data type you give it, it will look them up, sum the together in the same way. 

## `Numpy`

Now every python user would say *sure, we knew that - that's why we use numpy!* If you have homogenuos data, like here, using `numpy` has a lot of upsides in terms of speed. It being a C library, it will come close or even exceed our naive C code by using [`SIMD`](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions) instructions - a kind of parallel loop which works on modern CPUs. Let's see!
"

# ╔═╡ 12603fbc-55e0-11eb-2570-173a7f5fb60d
begin
	numpy_sum = pyimport("numpy")["sum"]
	apy_numpy = PyObject(a) # converts to a numpy array by default
	py_numpy_bench = @benchmark $numpy_sum($apy_numpy)
	if numpy_sum(apy_list) ≈ sum(a) nothing else error() end
	d[:numpy] = minimum(py_numpy_bench.times) / 1e6
	d
end

# ╔═╡ fa241eae-55e0-11eb-3bb4-f53733f79ac2
md"
Ok, not bad! Clearly much faster than standard python, and only $(relc(:numpy,d)) times slower than C
"

# ╔═╡ 65ed46ec-55e1-11eb-31f7-89d53b044287
md"
## Python by hand

Now let's roll our own `sum` function in python. This is something you would never do, but it fits our story. 😈
"

# ╔═╡ b067eda8-55e1-11eb-0f71-f9d8ae131d90
begin
	py"""
	def py_sum(a):
		s = 0.0
		for x in a:
			s = s + x
		return s
	"""

	sum_py = py"py_sum"

	py_hand = @benchmark $sum_py($apy_list)

	sum_py(apy_list)

	sum_py(apy_list) ≈ sum(a)

	d[:pyhand] = minimum(py_hand.times) / 1e6
	d
end

# ╔═╡ d672614a-55e1-11eb-078e-53c6eb9b4c30
md"
Oh. That's no good. $(relc(:pyhand,d)) times than C and $(rely(:pyhand,:numpy,d)) times slower than numpy.
"

# ╔═╡ 6b12b1fa-55e3-11eb-120f-a9ed49cb857c
md"
## Julia time

ok, now let's try julia. first the built-in `sum` function. What's cool about that function? It's not written in C - but in julia! so everybody can read it. look:
"

# ╔═╡ 968cb4b6-55e3-11eb-1de9-7def68cbc3f6
@edit sum(a)  # opens up my editor at the line where this is defined in julia source

# ╔═╡ 7d631336-55e3-11eb-2b77-95ef2b170eb6
begin
	j_bench = @benchmark sum($a)
	d[:julia] = minimum(j_bench.times) / 1e6
	d
end

# ╔═╡ ce645eb6-55e3-11eb-2fdc-13335c7752c7
md"
Hooray! that's fast! We are at $(relc(:julia,d)) times C, not bad at all! 

Ok, now what about a hand-written version? Like, we are probably not as good julia programmers as the julia developers are, right, so how much can we as normal users achieve here?

**Julia hand written**
"

# ╔═╡ 486493e6-55e4-11eb-0cd5-05a1265db1c4
begin
	function mysum(A)   
    	s = zero(eltype(A)) # correct zero for any data type in A
		for a in A
			s += a
		end
		s
	end

	j_bench_hand = @benchmark mysum($a)

	d[:julia_hand] = minimum(j_bench_hand.times) / 1e6
	d
end

# ╔═╡ Cell order:
# ╟─d0cb1cea-51fa-11eb-3097-f1b352770dbf
# ╟─794c7764-51fc-11eb-2325-91421e8833e3
# ╟─8a1db652-51fc-11eb-191c-af8d4141c316
# ╠═9be2d85c-51fc-11eb-1770-e3ee78651c62
# ╟─ee1de4d8-51fc-11eb-1d5f-119571ab06a5
# ╠═fef98014-51fc-11eb-15ad-d17044bcab51
# ╠═488c605c-51fd-11eb-1d04-61c8684822e7
# ╟─91c95162-51fd-11eb-0eac-e1df447f55dd
# ╠═a3087dd6-51fd-11eb-1a23-fd0a56e35457
# ╠═aa655158-51fd-11eb-3014-9560d3c32afc
# ╠═0258a7e8-51fe-11eb-3705-497c4e1ec940
# ╠═06df6446-51fe-11eb-12a1-e7ff18534536
# ╟─05a3d930-55dd-11eb-144e-2d5058bd89c1
# ╠═10a3b738-55dd-11eb-154b-0f2b4d155753
# ╟─18a3f560-55dd-11eb-1fed-4b5dc0b96df6
# ╠═0f449a7c-51fe-11eb-2c20-bfb771a8b00c
# ╟─76d0693e-55dd-11eb-10c5-1b6930bfee9d
# ╠═166fb3de-51fe-11eb-0b84-6b4126f44d0e
# ╟─a66b3930-55dd-11eb-2e47-a70a33b7fd9a
# ╠═b506dd50-55dd-11eb-2e38-87fd5ce241dc
# ╟─cafc00f4-55dd-11eb-24d5-85c5bf3cb9b0
# ╠═e0e3ca6e-55dd-11eb-0e94-f33e3d61a4d2
# ╟─f6ad1ecc-55dd-11eb-00ce-692fdfc74f81
# ╠═1837b192-55de-11eb-319d-2f4e3205a3d7
# ╟─32002adc-55de-11eb-241c-7780d36ec5d0
# ╠═437ebce2-55de-11eb-3ab7-85568c871c0a
# ╟─0ceaa1ae-55e1-11eb-2b16-ed63d0f5f752
# ╟─accb0914-55de-11eb-1d8e-13664efdd168
# ╠═12603fbc-55e0-11eb-2570-173a7f5fb60d
# ╟─fa241eae-55e0-11eb-3bb4-f53733f79ac2
# ╟─65ed46ec-55e1-11eb-31f7-89d53b044287
# ╠═b067eda8-55e1-11eb-0f71-f9d8ae131d90
# ╠═d672614a-55e1-11eb-078e-53c6eb9b4c30
# ╟─6b12b1fa-55e3-11eb-120f-a9ed49cb857c
# ╠═968cb4b6-55e3-11eb-1de9-7def68cbc3f6
# ╠═7d631336-55e3-11eb-2b77-95ef2b170eb6
# ╟─ce645eb6-55e3-11eb-2fdc-13335c7752c7
# ╠═486493e6-55e4-11eb-0cd5-05a1265db1c4
# ╠═bc7080ca-55e5-11eb-0508-bf469f549365
# ╠═fed8b4f0-55e5-11eb-2cc2-990221ad2789
