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

# ╔═╡ 47c94cc0-6706-11eb-03ed-9937e34af506
using PlutoUI

# ╔═╡ c348f288-55a9-11eb-1683-a7438f911f11
using Plots

# ╔═╡ 42ecaf0c-5585-11eb-07bf-05f964ffc325
md"


# ScPo Numerical Methods Week 1

Florian Oswald, 2021
"


# ╔═╡ 7cf6ef84-6700-11eb-3cb3-6531e82cbb52
html"<button onclick='present()'>present</button>"

# ╔═╡ 6930bc28-6700-11eb-0505-f7c546f1acea

md"
# Arrays
 
In this notebook we introduce some concepts important for work with arrays in julia. First though:

> What is an Array?

An array is a *container* for elements with **identical** data type. This is different from a spreadsheet or a `DataFrame`, which can have different data types in each column. It's also different from a `Dict`, which can have fully different datatypes (similar to a `python` dict or an `R` `list`)

#
"


# ╔═╡ 8140a376-5587-11eb-1372-cbf4cb8db60d
	md"""
	!!! info
	    The official [documentation](https://docs.julialang.org/en/v1/manual/arrays/) is **really** good. Check it out!
	"""

# ╔═╡ 2b1ab8e6-5588-11eb-03bc-0f1b03719f35
md"
# Creating Arrays
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

# ╔═╡ ba5dcb58-6706-11eb-3b6f-c5f94de03e0d
md"#"

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
#


There are also *named tuples*:
"

# ╔═╡ 8b36c8c4-55aa-11eb-357f-abc29a42300f
(a = 1.1, b = 2.3)

# ╔═╡ 1e3cf52a-5589-11eb-3d72-45b0f25f1edf
md"
# Array Creators

Here are some functions that create arrays:
"

# ╔═╡ 30db8bc4-5589-11eb-25cb-39afacd9bada
ones(2,3)

# ╔═╡ 35694d0c-5589-11eb-01a7-1d710daa7c08
zeros(Int,2,4)

# ╔═╡ 99605a26-5589-11eb-031b-bd7709085dbf
falses(3)

# ╔═╡ 9dceb348-5589-11eb-254c-b1b8cf0781c2
trues(1,3)

# ╔═╡ b26ddb82-6700-11eb-2375-71c73ceeeea2
md"
#
"

# ╔═╡ a254b474-5589-11eb-101d-253f75350205
ra = rand(3,2)

# ╔═╡ aada0c02-5589-11eb-3d3f-d5d5cf9b8e0f
rand(1:5,3,4)

# ╔═╡ 39d9bfa4-5589-11eb-265b-5bedf62d6266
fill(1.3,2,5)

# ╔═╡ 4edab028-5589-11eb-1d4e-df07f93cd986
md"
#

Here you see how to create an *empty* array with the `undef` object. notice that those numbers are garbage (they are just whatever value is currently stored in a certain location of your computer's RAM). Sometimes it's useful to *preallocate* a large array, before we do a long-running computation, say. We will see that *allocating* memory is an expensive operation in terms of computing time and we should try to minimize it.

The function `similar` is a *similar* idea: you create an array of same shape and type as the input arg, but it's filled with garbage:
"

# ╔═╡ 40472640-5589-11eb-19eb-838837e9fdc0
Array{Float64}(undef, 3,3)

# ╔═╡ f5109d62-5588-11eb-0abc-2d21b95a3477
similar(ra)

# ╔═╡ fa21d678-5589-11eb-391a-633dbadf9558
md"
## Manually creating arrays

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

# ╔═╡ 2835338a-679a-11eb-1557-a75b2d2d657a
typeof(falses(2,2))

# ╔═╡ 6242d548-55a5-11eb-22e4-531a821011c9
 promote_type(Float64, Bool)

# ╔═╡ c66ff85c-55a5-11eb-042a-836b641d0bc9
hcat(rand(2,2),fill("hi",2,5))

# ╔═╡ f81c4ae0-6789-11eb-2289-333a2f14725b
md"

# `push`ing onto an existing (or empty) array

* Oftentimes it's useful to accumulate elements along the way, e.g. in a loop.
* In general *preallocating* arrays is good practice, particularly if you repeatedly modify the elements of an array.
* But sometimes we don't know the eventual size of an array (it may depend on the outcome of some other operation)."

# ╔═╡ 4a16caaa-678a-11eb-1818-5f72eb0bb0bb
z = Int[]   

# ╔═╡ 592f0b86-678a-11eb-39f4-55ec3bd7989a
typeof(z)

# ╔═╡ 5d2906c8-678a-11eb-3e11-05f1efd3f778
length(z)

# ╔═╡ 60b6d98a-678a-11eb-34f0-91b61835ccc2
push!(z, "hi")

# ╔═╡ b77ae9fe-679a-11eb-19de-678b04bcc042
pushfirst!(z, 1, 1)

# ╔═╡ be5f5cf0-679a-11eb-2b4f-f7d5ab85c6a0
z

# ╔═╡ 686337f0-678a-11eb-09bd-2badaadba117
pop!(z)

# ╔═╡ 6dd1411e-678a-11eb-047a-a1bc3fdea332
z

# ╔═╡ 7c1e52b4-678a-11eb-1a97-3d56ba213b3b
append!(z, 14)

# ╔═╡ f53088a2-55a5-11eb-3b9b-8b91844c5384
md"

# Comprehensions

are a very powerful way to create arrays following from an expression:
"

# ╔═╡ 17ef63ac-55a6-11eb-3713-77dfe04fa601
comp = [ i + j for i in 1:3, j in 1:2]

# ╔═╡ 2cfde6f6-55a6-11eb-0a12-6b336a02c015
md"
so we could have also done before:
"

# ╔═╡ 3c4111cc-55a6-11eb-198b-458475b9b85f
hcat(rand(2,2), ["hi" for i in 1:2, j in 1:5 ])

# ╔═╡ b5182a08-679c-11eb-01c4-21be55d14ac0
md"
$$\sum_{i=1}^3 \sum_{j=i}^{16} i^2 + y^3$$
"

# ╔═╡ 3f2fb44a-55a5-11eb-1bc6-aba9b4f794c7
md"
# Ranges

We have `Range` objects in julia. this is similar to `linspace`.
"

# ╔═╡ 2d7852b8-55a8-11eb-07e6-e1dc3016acfd
typeof(0:10)

# ╔═╡ 36e1f93a-55a8-11eb-3c41-bf266aa44859
collect(0:10)

# ╔═╡ fdb8e9b4-6706-11eb-2016-3b0323a35d58
myrange = 0:14

# ╔═╡ 07c4a8a8-6707-11eb-1c69-892820a63270
myrange[5]

# ╔═╡ 0aad3756-6707-11eb-1980-4b52a8129d5d
md"#"

# ╔═╡ 667ca08c-55a8-11eb-1944-dbd8171bd5ad
@bind 📍 Slider(2:10, show_value=true)

# ╔═╡ 3ba19e1c-55a8-11eb-0268-d333107b4ccc
scatter(exp.(range(log(2.0), stop = log(4.0), length = 📍)), ones(📍),ylims = (0.8,1.2))

# ╔═╡ ee1f71a8-55a4-11eb-352b-396905c8b20e
md"
# Array Indexing - *Getting* values

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
#

we can also use logical indexing by supplying  a boolean array saying which elements to pick:
"

# ╔═╡ 2fae8126-55b0-11eb-17ac-c54318e62c5e
which_ones = rand([true false], 12)

# ╔═╡ 430925e6-55b0-11eb-1637-6fe56713e397
x[ which_ones ]

# ╔═╡ 59708954-55b1-11eb-0e62-b72b3f3f65f0
md"
# Broadcasting and *setting* values

* How can we modify the value of an array?
"

# ╔═╡ 7295c79e-6701-11eb-3ec4-2ba3c1f45bfe
x[1,2] = -9

# ╔═╡ 85456dc2-6701-11eb-3348-379f534568c8
x

# ╔═╡ a21d3740-6701-11eb-3009-4da56916e76c
md"
#

Ok. But now consider this vector here and a range of indices:
"

# ╔═╡ d8a704bc-6701-11eb-2b46-090ac1815774
v = ones(Int, 10)

# ╔═╡ fe354836-6701-11eb-191c-2de25e7daacf
v[2:3] = 2

# ╔═╡ 074dccd8-6702-11eb-038c-0b8c14b62647
md"
## Broadcasting

* This operation was not allowed.
* Why? on the left we had an `Array` and on the right we had a scalar.
* What we really wanted was to use the operation `=` (*assign to object*) in an **element-by-element** fashion on the array on the LHS. 
* **element-by-element** means to *broadcast* over a colleciton in julia.
* We use the dot `.` to mark broadcasting:
"

# ╔═╡ 58d5e0ce-6702-11eb-2130-b1e9f84e5a65
v[2:3] .= 2

# ╔═╡ 64be4458-6702-11eb-02f9-37de11e2b41f
v

# ╔═╡ 7268d01a-6702-11eb-057d-a33ff7b6dff5
md"
## Question

* If you were followign along, what is this going to do?

```julia
v[4:7] = [0, 0, 0, 0]
```

#
"

# ╔═╡ 9aad3dec-6702-11eb-15c9-75d6b2d3713e
v[4:7] = [0, 0, 0, 0]

# ╔═╡ 9e35cf88-6702-11eb-2a92-0f9271383c8f
v

# ╔═╡ ad8175b6-6702-11eb-2edf-c9b3304a5ddd
md"
That worked because the right and left of `=` had the same type!
"

# ╔═╡ c8319ed4-6702-11eb-1651-5bb3675f8aa2
md"
## Working with Slices

* Let's give a name to that slice now
"

# ╔═╡ d91a964c-6702-11eb-12ad-f7d5398c1591
s = v[4:7]

# ╔═╡ 578f31c2-6703-11eb-05a7-11f472bc9ee5
md"
#

Well let's try of course. We are here currently:
"

# ╔═╡ 64ebb606-6703-11eb-332b-2d933e7bddec
s

# ╔═╡ 6696f222-6703-11eb-3862-8938b0a92edb
md"Now let's set one of those values:"

# ╔═╡ 6f9b3284-6703-11eb-1d85-9366ffbb66f0
s[2] = 3333

# ╔═╡ 7929685c-6703-11eb-3f14-61ebf493ecde
s

# ╔═╡ 7dc0ec32-6703-11eb-2cd7-27b1f1dcca3a
md"So far so good. But what happend to the original array `v`?

#

"

# ╔═╡ a37e8fd0-6703-11eb-3d69-85511daf9720
v

# ╔═╡ a6732366-6703-11eb-36da-110356d80106
md"Nothing!

Surprised? 

We take not that the operator `[]` makes a *copy* of the data. Our object `s` is allocated on a *different* set of memory locations than the original array `v`, hence changing `s` does not affect `v`.

Here is an alternative:

# Views

Sometimes we don't want to take a copy, but just operate on some subset of array. Literally *on the same memory* in RAM. a `view` creates a *window* into the original array:
"

# ╔═╡ c810b5ec-6703-11eb-293c-69fc0a8e573f
w = view(v, 4:7)

# ╔═╡ 35ae4a94-6704-11eb-3ebd-a75cbd2ca619
typeof(w)

# ╔═╡ 3d914642-6704-11eb-00a4-c1011ff61fdd

md"
#

Ok let's try to modify that again:
"

# ╔═╡ 9e79d118-6704-11eb-3cff-613653dd369b
w[3:4] .= -999

# ╔═╡ a1ca494c-6704-11eb-3bfd-b520e12ac37a
v

# ╔═╡ a7e0ee82-6704-11eb-258e-e77dee71e929
md"
It did modify the original array! ✅

# `@view` macro

* There is a nicer way to write it as well
"

# ╔═╡ 729bda00-6704-11eb-126a-439d41284c1c
w2 = @view v[3:5]

# ╔═╡ 9b2565fa-6703-11eb-26d0-d5a4c520bfef
md"
What's this trickery? The macro @view literally takes the piece of Julia code v[3:5] and replaces it with the new piece of code view(v, 3:5)

# Matrices

This works in the same way on matrices or higher dimensional arrays.
"

# ╔═╡ 5b3f73ac-6705-11eb-3fc3-a7132d790ddc
M = [10i + j for i in 0:5, j in 1:4]

# ╔═╡ 5c8f0d58-6705-11eb-0415-afa1032e72b3
M[3:5, 1:2]

# ╔═╡ 63b39b30-6705-11eb-11ef-31053508273a
md"
# Reshaping

* Reshaping a matrix or a general array is done with the `reshape` function:
"

# ╔═╡ 7e404c62-6705-11eb-2886-6b9da420f7b0
reshape(M, 3, 8)

# ╔═╡ a07a1634-6705-11eb-0342-61fd861a94a9
md"Notice that julia is *column-major* storage: i.e. we travers first columns, then rows through memory. This is reflected in how the reshaping picks elements

* Some times we also want to convert an array back to a simple vector (we strip all dimensionality info away from a vector)
* You can again see the storage order of the data in your computer's memory

"

# ╔═╡ e0180ac6-6705-11eb-3fe0-1ffdb3809f11
M[:]

# ╔═╡ e4dca698-6705-11eb-2e54-e751eb17d068
vec(M)

# ╔═╡ 7fb9ae2e-6704-11eb-0db7-abc17c115185
md"library"

# ╔═╡ 1e745340-6703-11eb-0853-2d40b6d3bc46
info(text) = Markdown.MD(Markdown.Admonition("info", "Info", [text]));

# ╔═╡ e48c1410-6702-11eb-06c4-8300fc6614af
info(md"
What's going to happen if you modify the values in `s`?
That's not a simple question!")

# ╔═╡ 7dc2bb38-6704-11eb-2aed-8563be2b3d89
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

# ╔═╡ Cell order:
# ╟─42ecaf0c-5585-11eb-07bf-05f964ffc325
# ╟─7cf6ef84-6700-11eb-3cb3-6531e82cbb52
# ╟─6930bc28-6700-11eb-0505-f7c546f1acea
# ╟─8140a376-5587-11eb-1372-cbf4cb8db60d
# ╟─2b1ab8e6-5588-11eb-03bc-0f1b03719f35
# ╠═ded79d1e-5587-11eb-0e88-c56751e46abc
# ╠═f8100ff0-5587-11eb-0731-f7b839004f3c
# ╠═fdf808d2-5587-11eb-0302-1b739aaf6be5
# ╠═0a8879f6-5588-11eb-023e-afe1ef6a1eb1
# ╟─578ff698-5588-11eb-07fc-633d03c23062
# ╟─ba5dcb58-6706-11eb-3b6f-c5f94de03e0d
# ╠═79c9e5ac-5588-11eb-0341-01ca0aa2e411
# ╠═9e5ec180-5588-11eb-34a2-f325d272532d
# ╟─a6b4ad92-5588-11eb-233f-63c5737ad710
# ╠═f6b5bbe2-5588-11eb-08e7-59236a4a35bf
# ╠═ff363182-5588-11eb-2f7e-3180d1537941
# ╟─7fa5274e-55aa-11eb-0399-17c0fbccac1b
# ╠═8b36c8c4-55aa-11eb-357f-abc29a42300f
# ╟─1e3cf52a-5589-11eb-3d72-45b0f25f1edf
# ╠═30db8bc4-5589-11eb-25cb-39afacd9bada
# ╠═35694d0c-5589-11eb-01a7-1d710daa7c08
# ╠═99605a26-5589-11eb-031b-bd7709085dbf
# ╠═9dceb348-5589-11eb-254c-b1b8cf0781c2
# ╟─b26ddb82-6700-11eb-2375-71c73ceeeea2
# ╠═a254b474-5589-11eb-101d-253f75350205
# ╠═aada0c02-5589-11eb-3d3f-d5d5cf9b8e0f
# ╠═39d9bfa4-5589-11eb-265b-5bedf62d6266
# ╟─4edab028-5589-11eb-1d4e-df07f93cd986
# ╠═40472640-5589-11eb-19eb-838837e9fdc0
# ╠═f5109d62-5588-11eb-0abc-2d21b95a3477
# ╟─fa21d678-5589-11eb-391a-633dbadf9558
# ╠═cadb8b28-55a4-11eb-1600-5739c2a95a9c
# ╠═d97da51c-55a4-11eb-305c-1bb0e1634cb0
# ╠═e27215b8-55a4-11eb-21c5-2738c0897729
# ╠═09e97578-55a5-11eb-1042-0d7ea64aa830
# ╠═516b88a0-55a5-11eb-105e-17c75539d751
# ╠═2835338a-679a-11eb-1557-a75b2d2d657a
# ╠═6242d548-55a5-11eb-22e4-531a821011c9
# ╠═c66ff85c-55a5-11eb-042a-836b641d0bc9
# ╟─f81c4ae0-6789-11eb-2289-333a2f14725b
# ╠═4a16caaa-678a-11eb-1818-5f72eb0bb0bb
# ╠═592f0b86-678a-11eb-39f4-55ec3bd7989a
# ╠═5d2906c8-678a-11eb-3e11-05f1efd3f778
# ╠═60b6d98a-678a-11eb-34f0-91b61835ccc2
# ╠═b77ae9fe-679a-11eb-19de-678b04bcc042
# ╠═be5f5cf0-679a-11eb-2b4f-f7d5ab85c6a0
# ╠═686337f0-678a-11eb-09bd-2badaadba117
# ╠═6dd1411e-678a-11eb-047a-a1bc3fdea332
# ╠═7c1e52b4-678a-11eb-1a97-3d56ba213b3b
# ╟─f53088a2-55a5-11eb-3b9b-8b91844c5384
# ╠═17ef63ac-55a6-11eb-3713-77dfe04fa601
# ╟─2cfde6f6-55a6-11eb-0a12-6b336a02c015
# ╠═3c4111cc-55a6-11eb-198b-458475b9b85f
# ╟─b5182a08-679c-11eb-01c4-21be55d14ac0
# ╟─3f2fb44a-55a5-11eb-1bc6-aba9b4f794c7
# ╠═2d7852b8-55a8-11eb-07e6-e1dc3016acfd
# ╠═36e1f93a-55a8-11eb-3c41-bf266aa44859
# ╠═fdb8e9b4-6706-11eb-2016-3b0323a35d58
# ╠═07c4a8a8-6707-11eb-1c69-892820a63270
# ╟─0aad3756-6707-11eb-1980-4b52a8129d5d
# ╠═47c94cc0-6706-11eb-03ed-9937e34af506
# ╠═667ca08c-55a8-11eb-1944-dbd8171bd5ad
# ╠═c348f288-55a9-11eb-1683-a7438f911f11
# ╠═3ba19e1c-55a8-11eb-0268-d333107b4ccc
# ╟─ee1f71a8-55a4-11eb-352b-396905c8b20e
# ╠═2920a9e6-55a7-11eb-235a-d9f89dbdc86d
# ╠═b65533cc-55a7-11eb-22ce-fb51e1d6a4c2
# ╠═c0efcdec-55a7-11eb-15a5-e700195132aa
# ╠═c749dcbe-55a7-11eb-3b6e-4917aaff9e88
# ╠═cace8fa8-55a7-11eb-083b-d5050a6ad009
# ╠═d087566c-55a7-11eb-22cf-493144e9f225
# ╠═dff566ca-55a7-11eb-1b8c-335c0aaf3ac4
# ╟─1a1eb880-55b0-11eb-31af-7572840ce1a2
# ╠═2fae8126-55b0-11eb-17ac-c54318e62c5e
# ╠═430925e6-55b0-11eb-1637-6fe56713e397
# ╟─59708954-55b1-11eb-0e62-b72b3f3f65f0
# ╠═7295c79e-6701-11eb-3ec4-2ba3c1f45bfe
# ╠═85456dc2-6701-11eb-3348-379f534568c8
# ╟─a21d3740-6701-11eb-3009-4da56916e76c
# ╠═d8a704bc-6701-11eb-2b46-090ac1815774
# ╠═fe354836-6701-11eb-191c-2de25e7daacf
# ╟─074dccd8-6702-11eb-038c-0b8c14b62647
# ╠═58d5e0ce-6702-11eb-2130-b1e9f84e5a65
# ╠═64be4458-6702-11eb-02f9-37de11e2b41f
# ╟─7268d01a-6702-11eb-057d-a33ff7b6dff5
# ╠═9aad3dec-6702-11eb-15c9-75d6b2d3713e
# ╠═9e35cf88-6702-11eb-2a92-0f9271383c8f
# ╟─ad8175b6-6702-11eb-2edf-c9b3304a5ddd
# ╟─c8319ed4-6702-11eb-1651-5bb3675f8aa2
# ╠═d91a964c-6702-11eb-12ad-f7d5398c1591
# ╟─e48c1410-6702-11eb-06c4-8300fc6614af
# ╟─578f31c2-6703-11eb-05a7-11f472bc9ee5
# ╠═64ebb606-6703-11eb-332b-2d933e7bddec
# ╟─6696f222-6703-11eb-3862-8938b0a92edb
# ╠═6f9b3284-6703-11eb-1d85-9366ffbb66f0
# ╠═7929685c-6703-11eb-3f14-61ebf493ecde
# ╟─7dc0ec32-6703-11eb-2cd7-27b1f1dcca3a
# ╠═a37e8fd0-6703-11eb-3d69-85511daf9720
# ╟─a6732366-6703-11eb-36da-110356d80106
# ╠═c810b5ec-6703-11eb-293c-69fc0a8e573f
# ╠═35ae4a94-6704-11eb-3ebd-a75cbd2ca619
# ╟─3d914642-6704-11eb-00a4-c1011ff61fdd
# ╠═9e79d118-6704-11eb-3cff-613653dd369b
# ╠═a1ca494c-6704-11eb-3bfd-b520e12ac37a
# ╟─a7e0ee82-6704-11eb-258e-e77dee71e929
# ╠═729bda00-6704-11eb-126a-439d41284c1c
# ╟─9b2565fa-6703-11eb-26d0-d5a4c520bfef
# ╠═5b3f73ac-6705-11eb-3fc3-a7132d790ddc
# ╠═5c8f0d58-6705-11eb-0415-afa1032e72b3
# ╟─63b39b30-6705-11eb-11ef-31053508273a
# ╠═7e404c62-6705-11eb-2886-6b9da420f7b0
# ╟─a07a1634-6705-11eb-0342-61fd861a94a9
# ╠═e0180ac6-6705-11eb-3fe0-1ffdb3809f11
# ╠═e4dca698-6705-11eb-2e54-e751eb17d068
# ╟─7fb9ae2e-6704-11eb-0db7-abc17c115185
# ╟─1e745340-6703-11eb-0853-2d40b6d3bc46
# ╟─7dc2bb38-6704-11eb-2aed-8563be2b3d89
