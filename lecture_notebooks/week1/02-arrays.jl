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

# ╔═╡ c348f288-55a9-11eb-1683-a7438f911f11
using Plots

# ╔═╡ 42ecaf0c-5585-11eb-07bf-05f964ffc325
md"
# Arrays
 
In this notebook we introduce some concepts important for work with arrays in julia. First though:

> What is an Array?

An array is a *container* for elements with **identical** data type. This is different from a spreadsheet or a `DataFrame`, which can have different data types in each column. It's also different from a `Dict`, which can have fully different datatypes (similar to a `python` dict or an `R` `list`)


"


# ╔═╡ 8140a376-5587-11eb-1372-cbf4cb8db60d
	md"""
	!!! info
	    The official [documentation](https://docs.julialang.org/en/v1/manual/arrays/) is **really** good. Check it out!
	"""

# ╔═╡ 2b1ab8e6-5588-11eb-03bc-0f1b03719f35
md"
## Creating Arrays
"

# ╔═╡ ded79d1e-5587-11eb-0e88-c56751e46abc
a = [1.0, 2.0, 3.0]

# ╔═╡ f8100ff0-5587-11eb-0731-f7b839004f3c
b = [1, 2, 3]

# ╔═╡ fdf808d2-5587-11eb-0302-1b739aaf6be5
typeof(a)

# ╔═╡ 0a8879f6-5588-11eb-023e-afe1ef6a1eb1
typeof(b)

# ╔═╡ 578ff698-5588-11eb-07fc-633d03c23062
md"
Notice the different types of those arrays.

We can get size and dimension info about an array with `size` and `ndims`. `length` returns the total number of elements.
"

# ╔═╡ 79c9e5ac-5588-11eb-0341-01ca0aa2e411
size(a)

# ╔═╡ 9e5ec180-5588-11eb-34a2-f325d272532d
typeof(size(a))

# ╔═╡ a6b4ad92-5588-11eb-233f-63c5737ad710
md"*Tuple*! That's useful for *short* containers and is used a lot in function arguments or when we want to return multiple things from a function. [here](https://docs.julialang.org/en/v1/manual/functions/#Tuples) is the relevant documentation. Notice that you cannot *change* a tuple once it's created. it is **immutable**:
"

# ╔═╡ f6b5bbe2-5588-11eb-08e7-59236a4a35bf
tu = (1,2,1)

# ╔═╡ ff363182-5588-11eb-2f7e-3180d1537941
tu[2] = 3  # I want to access the second element and change it to value 3!

# ╔═╡ 7fa5274e-55aa-11eb-0399-17c0fbccac1b
md"
There are also *named tuples*:
"

# ╔═╡ 8b36c8c4-55aa-11eb-357f-abc29a42300f
(a = 1.1, b = 2.3)

# ╔═╡ 1e3cf52a-5589-11eb-3d72-45b0f25f1edf
md"
### Array Creators

Here are some functions that create arrays:
"

# ╔═╡ 30db8bc4-5589-11eb-25cb-39afacd9bada
ones(2,3)

# ╔═╡ 499e8a78-55a5-11eb-337b-9bce2ab0a039


# ╔═╡ 35694d0c-5589-11eb-01a7-1d710daa7c08
zeros(2,4)

# ╔═╡ 99605a26-5589-11eb-031b-bd7709085dbf
falses(3)

# ╔═╡ 9dceb348-5589-11eb-254c-b1b8cf0781c2
trues(1,3)

# ╔═╡ a254b474-5589-11eb-101d-253f75350205
ra = rand(3,2)

# ╔═╡ aada0c02-5589-11eb-3d3f-d5d5cf9b8e0f
rand(1:5,3,4)

# ╔═╡ 39d9bfa4-5589-11eb-265b-5bedf62d6266
fill(1.3,2,5)

# ╔═╡ 40472640-5589-11eb-19eb-838837e9fdc0
Array{Float64}(undef, 3,3)

# ╔═╡ 4edab028-5589-11eb-1d4e-df07f93cd986
md"
Here you see how to create an *empty* array with the `undef` object. notice that those numbers are garbage (they are just whatever value is currently stored in a certain location of your computer's RAM). Sometimes it's useful to *preallocate* a large array, before we do a long-running computation, say. We will see that *allocating* memory is an expensive operation in terms of computing time and we should try to minimize it.

The function `similar` is a *similar* idea: you create an array of same shape and type as the input arg, but it's filled with garbage:
"

# ╔═╡ f5109d62-5588-11eb-0abc-2d21b95a3477
similar(ra)

# ╔═╡ fa21d678-5589-11eb-391a-633dbadf9558
md"
### Manually creating arrays

We can create arrays manually, as above, with the `[` operator:
"

# ╔═╡ cadb8b28-55a4-11eb-1600-5739c2a95a9c
a2 = [ 1 2 ;
	   3 4 ]

# ╔═╡ d97da51c-55a4-11eb-305c-1bb0e1634cb0
ndims(a2)

# ╔═╡ e27215b8-55a4-11eb-21c5-2738c0897729
a2'

# ╔═╡ 09e97578-55a5-11eb-1042-0d7ea64aa830
vcat(ones(3),1.1)

# ╔═╡ 516b88a0-55a5-11eb-105e-17c75539d751
hcat(rand(2,2),falses(2,3))

# ╔═╡ 6242d548-55a5-11eb-22e4-531a821011c9
 promote_type(Float64, Bool)

# ╔═╡ c66ff85c-55a5-11eb-042a-836b641d0bc9
hcat(rand(2,2),fill("hi",2,5))

# ╔═╡ bfd3cf90-55a6-11eb-189d-2b168f9581df


# ╔═╡ e9167d02-55a5-11eb-25cd-0dfc94f3e699
typeof(Any)

# ╔═╡ f53088a2-55a5-11eb-3b9b-8b91844c5384
md"
**Comprehensions** are a very powerful way to create arrays following from an expression:
"

# ╔═╡ 17ef63ac-55a6-11eb-3713-77dfe04fa601
comp = [ i + j for i in 1:3, j in 1:2]

# ╔═╡ 2cfde6f6-55a6-11eb-0a12-6b336a02c015
md"
so we could have also done before:
"

# ╔═╡ 3c4111cc-55a6-11eb-198b-458475b9b85f
hcat(rand(2,2), ["hi" for i in 1:2, j in 1:5 ])

# ╔═╡ ea8ada52-55b3-11eb-3994-5fe0313bd4a4
	md"""
	!!! warning
	    
	"""

# ╔═╡ f4c3f9fe-55b3-11eb-03ab-d5770e351648


# ╔═╡ 3f2fb44a-55a5-11eb-1bc6-aba9b4f794c7
md"
## Ranges

We have `Range` objects in julia. this is similar to `linspace`.
"

# ╔═╡ 2d7852b8-55a8-11eb-07e6-e1dc3016acfd
typeof(0:10)

# ╔═╡ 36e1f93a-55a8-11eb-3c41-bf266aa44859
collect(0:10)

# ╔═╡ 667ca08c-55a8-11eb-1944-dbd8171bd5ad
begin
	📍slider = @bind 📍 html"<input type=range min=2 max=10>"
	
	md"""**How many points do you want?**
	
	Choose: $(📍slider)
	"""
end

# ╔═╡ 54852468-55a9-11eb-0621-35560d6f3a46
md"
you chose $(📍) points
"

# ╔═╡ 3ba19e1c-55a8-11eb-0268-d333107b4ccc
scatter(exp.(range(log(2.0), stop = log(4.0), length = 📍)), ones(📍),ylims = (0.8,1.2))

# ╔═╡ ee1f71a8-55a4-11eb-352b-396905c8b20e
md"
## Array Indexing

We can use the square bracket operator `[ix]` to get element number `ix`. There is a difference between **linear indexing**, ie. traversing the array in *storage order*, vs **Cartesian indexing**, i.e. addressing the element by its location in the cartesian grid. Julia does both.  
"

# ╔═╡ 2920a9e6-55a7-11eb-235a-d9f89dbdc86d
x = [i + 3*(j-1) for i in 1:3, j in 1:4]

# ╔═╡ b65533cc-55a7-11eb-22ce-fb51e1d6a4c2
x[1,1]

# ╔═╡ c0efcdec-55a7-11eb-15a5-e700195132aa
x[1,3]

# ╔═╡ c749dcbe-55a7-11eb-3b6e-4917aaff9e88
x[1,:]

# ╔═╡ cace8fa8-55a7-11eb-083b-d5050a6ad009
x[:,1]

# ╔═╡ d087566c-55a7-11eb-22cf-493144e9f225
x[4]  # the 4th linear index in storage order. julia stores arrays column major

# ╔═╡ dff566ca-55a7-11eb-1b8c-335c0aaf3ac4
x[end] #reserved word `end`

# ╔═╡ 1a1eb880-55b0-11eb-31af-7572840ce1a2
md"
we can also use logical indexing by supplying  a boolean array saying which elements to pick:
"

# ╔═╡ 2fae8126-55b0-11eb-17ac-c54318e62c5e
which_ones = rand([true false], 12)

# ╔═╡ 430925e6-55b0-11eb-1637-6fe56713e397
slice = x[ which_ones ]

# ╔═╡ 1083eb14-55b1-11eb-1a61-9bd248cd93ba
typeof(slice)

# ╔═╡ 98752444-55b0-11eb-0dac-c5c30bf4c114
v = view(x, 1, :)

# ╔═╡ adf29b1c-55b0-11eb-31c3-7f6c77c45323
typeof(v)

# ╔═╡ 47f3a4c8-55b0-11eb-2ef3-f7aa996ebe4d
md"
slices vs views:

* `slice`: `x[1,:]` makes a *copy* of the data and returns a new object
* `view(x, dims)` or `@view x[ ]`: creates only a *view* into a **parent** array, thus avoids copying data. In large arrays, this can make a big difference for efficiency.
"

# ╔═╡ 357e1e12-55b1-11eb-1c6c-0372d03093a8
v[:] .= 100

# ╔═╡ 52d7230a-55b1-11eb-0fc6-59ae68a1be06
x

# ╔═╡ 59708954-55b1-11eb-0e62-b72b3f3f65f0


# ╔═╡ 54ba81e4-55b1-11eb-1334-69a2264c559c


# ╔═╡ Cell order:
# ╟─42ecaf0c-5585-11eb-07bf-05f964ffc325
# ╠═8140a376-5587-11eb-1372-cbf4cb8db60d
# ╟─2b1ab8e6-5588-11eb-03bc-0f1b03719f35
# ╠═ded79d1e-5587-11eb-0e88-c56751e46abc
# ╠═f8100ff0-5587-11eb-0731-f7b839004f3c
# ╠═fdf808d2-5587-11eb-0302-1b739aaf6be5
# ╠═0a8879f6-5588-11eb-023e-afe1ef6a1eb1
# ╟─578ff698-5588-11eb-07fc-633d03c23062
# ╠═79c9e5ac-5588-11eb-0341-01ca0aa2e411
# ╠═9e5ec180-5588-11eb-34a2-f325d272532d
# ╟─a6b4ad92-5588-11eb-233f-63c5737ad710
# ╠═f6b5bbe2-5588-11eb-08e7-59236a4a35bf
# ╠═ff363182-5588-11eb-2f7e-3180d1537941
# ╠═7fa5274e-55aa-11eb-0399-17c0fbccac1b
# ╠═8b36c8c4-55aa-11eb-357f-abc29a42300f
# ╟─1e3cf52a-5589-11eb-3d72-45b0f25f1edf
# ╠═30db8bc4-5589-11eb-25cb-39afacd9bada
# ╠═499e8a78-55a5-11eb-337b-9bce2ab0a039
# ╠═35694d0c-5589-11eb-01a7-1d710daa7c08
# ╠═99605a26-5589-11eb-031b-bd7709085dbf
# ╠═9dceb348-5589-11eb-254c-b1b8cf0781c2
# ╠═a254b474-5589-11eb-101d-253f75350205
# ╠═aada0c02-5589-11eb-3d3f-d5d5cf9b8e0f
# ╠═39d9bfa4-5589-11eb-265b-5bedf62d6266
# ╠═40472640-5589-11eb-19eb-838837e9fdc0
# ╟─4edab028-5589-11eb-1d4e-df07f93cd986
# ╠═f5109d62-5588-11eb-0abc-2d21b95a3477
# ╟─fa21d678-5589-11eb-391a-633dbadf9558
# ╠═cadb8b28-55a4-11eb-1600-5739c2a95a9c
# ╠═d97da51c-55a4-11eb-305c-1bb0e1634cb0
# ╠═e27215b8-55a4-11eb-21c5-2738c0897729
# ╠═09e97578-55a5-11eb-1042-0d7ea64aa830
# ╠═516b88a0-55a5-11eb-105e-17c75539d751
# ╠═6242d548-55a5-11eb-22e4-531a821011c9
# ╠═c66ff85c-55a5-11eb-042a-836b641d0bc9
# ╠═bfd3cf90-55a6-11eb-189d-2b168f9581df
# ╠═e9167d02-55a5-11eb-25cd-0dfc94f3e699
# ╠═f53088a2-55a5-11eb-3b9b-8b91844c5384
# ╠═17ef63ac-55a6-11eb-3713-77dfe04fa601
# ╠═2cfde6f6-55a6-11eb-0a12-6b336a02c015
# ╠═3c4111cc-55a6-11eb-198b-458475b9b85f
# ╠═ea8ada52-55b3-11eb-3994-5fe0313bd4a4
# ╠═f4c3f9fe-55b3-11eb-03ab-d5770e351648
# ╟─3f2fb44a-55a5-11eb-1bc6-aba9b4f794c7
# ╠═2d7852b8-55a8-11eb-07e6-e1dc3016acfd
# ╠═36e1f93a-55a8-11eb-3c41-bf266aa44859
# ╟─667ca08c-55a8-11eb-1944-dbd8171bd5ad
# ╟─54852468-55a9-11eb-0621-35560d6f3a46
# ╠═c348f288-55a9-11eb-1683-a7438f911f11
# ╠═3ba19e1c-55a8-11eb-0268-d333107b4ccc
# ╠═ee1f71a8-55a4-11eb-352b-396905c8b20e
# ╠═2920a9e6-55a7-11eb-235a-d9f89dbdc86d
# ╠═b65533cc-55a7-11eb-22ce-fb51e1d6a4c2
# ╠═c0efcdec-55a7-11eb-15a5-e700195132aa
# ╠═c749dcbe-55a7-11eb-3b6e-4917aaff9e88
# ╠═cace8fa8-55a7-11eb-083b-d5050a6ad009
# ╠═d087566c-55a7-11eb-22cf-493144e9f225
# ╠═dff566ca-55a7-11eb-1b8c-335c0aaf3ac4
# ╠═1a1eb880-55b0-11eb-31af-7572840ce1a2
# ╠═2fae8126-55b0-11eb-17ac-c54318e62c5e
# ╠═430925e6-55b0-11eb-1637-6fe56713e397
# ╠═1083eb14-55b1-11eb-1a61-9bd248cd93ba
# ╠═98752444-55b0-11eb-0dac-c5c30bf4c114
# ╠═adf29b1c-55b0-11eb-31c3-7f6c77c45323
# ╟─47f3a4c8-55b0-11eb-2ef3-f7aa996ebe4d
# ╠═357e1e12-55b1-11eb-1c6c-0372d03093a8
# ╠═52d7230a-55b1-11eb-0fc6-59ae68a1be06
# ╠═59708954-55b1-11eb-0e62-b72b3f3f65f0
# ╠═54ba81e4-55b1-11eb-1334-69a2264c559c
