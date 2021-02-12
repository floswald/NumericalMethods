### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 082b1918-678b-11eb-1a78-2316091d6272
using PlutoUI

# ╔═╡ f855b1c0-670e-11eb-2a81-4de0846c0d93
using Dates   

# ╔═╡ 9e2fc76c-6727-11eb-2cab-7f81bdfb7343
using Distributions

# ╔═╡ 4723a50e-f6d1-11ea-0b1c-3d33a9b92f87
md"# Defining new types"

# ╔═╡ 91ba119e-670a-11eb-0bb2-15ac7192e7e7
html"<button onclick='present()'>present</button>"

# ╔═╡ 92cef0a4-670a-11eb-2df2-5b31a1602299
md"
# Types in Julia

* Julia has a very rich type system
* The [manual](https://docs.julialang.org/en/v1/manual/types/) is very good again.
* What's really cool with julia is the easy with which we can declare new types.
"

# ╔═╡ d19385e0-678a-11eb-3156-c51cb20abe3e
md"
# Julia Base Type System

Here is the basics of the `Number` type in julia. This is just one part of the type system. 

At the very top of all types (not shown) is the `DataType` `Any`, from which all others are derived.
"

# ╔═╡ e97a60e6-678c-11eb-0b37-fd80c3f0e8aa
Resource("https://upload.wikimedia.org/wikipedia/commons/d/d9/Julia-number-type-hierarchy.svg")

# ╔═╡ aa5adfba-678e-11eb-02e4-210b8b753e21
subtypes(Integer)

# ╔═╡ bff062f0-678e-11eb-131d-e11297e36d4b
supertypes(Integer)

# ╔═╡ bed2a154-f6d1-11ea-0812-1f5c628a9785
md"
# Why define new types?

Here is an example. Many mathematical and other objects can be represented by a pair of numbers, for example:

  - a rectangle has a width and a height;
  - a complex number has a real and imaginary part;
  - a position vector in 2D space has 2 components.

Each of these could naturally be represented by a pair $(x, y)$ of two numbers, or in Julia as a tuple:"

# ╔═╡ 81113974-f6d2-11ea-2ef8-fb2930402a74
begin 
	rectangle = (3, 4)   # (width, height)
	c = (3, 4)   # 3 + 4im
	x = (3, 4)   # position vector
end

# ╔═╡ a4296474-6714-11eb-0500-1d888fe18b51
md"#"

# ╔═╡ ac321d80-f6d2-11ea-2951-676e6e1aef56
md"


But from the fact that we have to remind ourselves using comments what each of these numbers represents, and the fact that they all look the same but should behave very differently, that there is a problem here. 

For example, we would like to have a function `width` that returns the width of a rectangle, but it makes no sense to apply that to a complex number.

In other words, we need a way to be able to distinguish between different *types of* objects with different *behaviours*.
"

# ╔═╡ 9a168a92-670b-11eb-34af-0121d1a252bd
md"
# Defining a New type
"

# ╔═╡ 66abc04e-f6d8-11ea-27d8-9f8b14659755
struct Rectangle
	width::Float64
	height::Float64
end

# ╔═╡ 7571be3a-f6d8-11ea-174c-9d65d5185153
md"""The keyword `struct` (short for "structure") tells Julia that we are defining a new type. We list the field names together with **type annotations** using `::` that specify which type each field can contain."""

# ╔═╡ 9f384ac2-f6d8-11ea-297e-4bf09acf9fe7
md"Once Julia has this template, we can create objects which have that type as follows:"

# ╔═╡ b0dac516-f6d8-11ea-1bdb-b59723107206
r = Rectangle(1, 2.5)

# ╔═╡ af236602-5ffa-11eb-0bec-cd944a602c70
md"the function `Rectangle` with identical name to our type is called a **constructor**."

# ╔═╡ b98b9faa-f6d8-11ea-3610-bf8a84af2b5a
md"
#

We can check that `r` is now a variable whose type is `Rectangle`, in other words `r` *is a* `Rectangle`:"

# ╔═╡ cf1c4aae-f6d8-11ea-3200-c5fb458c7c09
typeof(r)

# ╔═╡ d0749974-f6d8-11ea-2f41-074b6744f3d5
r isa Rectangle

# ╔═╡ d372342c-f6d8-11ea-10cd-573cf7eab992
md"""We can  extract from `r` the information about the values of the fields that it contains using "`.`": """

# ╔═╡ dd8f9e88-f6d8-11ea-15e3-f17f4af0d81b
r.width

# ╔═╡ e3e70064-f6d8-11ea-22fd-892bbc567ed4
r.height

# ╔═╡ e582eb02-f6d8-11ea-1fcc-89bbc9dfbb07
md"

#

We can create a new `Rectangle` with its own width and height:"

# ╔═╡ f2ed18b4-f6d8-11ea-3bc7-0b82eb5e8dc0
r2 = Rectangle(3.0, 4.0)

# ╔═╡ f9d192fe-f6d8-11ea-138d-3dcdff33c034
md"You should check that this does *not* affect the `width` and `height` variables belonging to `r`."

# ╔═╡ 6085144c-f6db-11ea-19fe-ed46dafb4562
md"Types like this are often called **composite types**; they consist of aggregating, or collecting together, different pieces of information that belong to a given object."

# ╔═╡ 1840898e-f6d9-11ea-3035-bb4dac496834
md"

## Mutable vs immutable


Now suppose we want to change the width of `r`. We would naturally try the following:

"

# ╔═╡ 63f76d28-f6d9-11ea-071c-458528c36008
r.width = 10

# ╔═╡ 68934a2a-f6d9-11ea-37ea-850304f6d3d6
md"But Julia complains that fields of objects of type `Rectangle` *cannot* be modified. This is because `struct` makes **immutable** (i.e. unchangeable) objects. The reason for this is that these usually lead to *faster code*."

# ╔═╡ a38a0164-f6d9-11ea-2e1d-a7a6e2106d0b
md"If we really want to have a **mutable** (modifiable) object, we can declare it using `mutable struct` instead of `struct` -- try it!"

# ╔═╡ 3fa7d8e2-f6d9-11ea-2e82-9f59b5cb9424
md"## Functions on types

We can now define functions that act only on given types. To restrict a given function argument to only accept objects of a certain type, we add a type annotation:"

# ╔═╡ 8c4beb6e-f6db-11ea-12d1-cf450181363b
width(r::Rectangle) = r.width

# ╔═╡ 91e21d28-f6db-11ea-1b0f-336719682f28
width(r)

# ╔═╡ 2ef7392e-f6dc-11ea-00e4-770cdf9a102e
md"Applying the function to objects of other types gives an error:"

# ╔═╡ 9371209e-f6db-11ea-3ba2-c3597d42d8ed
width(3)   # throws an error

# ╔═╡ b16a7348-6d16-11eb-1200-6967c76a9f5e


# ╔═╡ d1166a30-670b-11eb-320d-9b13b96824fb
md"#"

# ╔═╡ b916acb4-f6dc-11ea-3cdf-2b8ab3c34e03
md"""It is common in Julia to have "generic" versions of functions that apply to any object, and then specialised versions that apply only to certain types.

For example, we might want to define an `area` function with the correct definition for `Rectangle`s, but it might be convenient to fall back to a version that just returns the value itself for any other type:"""

# ╔═╡ 6fe1b332-f6dd-11ea-39d4-1954aeda6f08
begin
	# two different function bodies for the same function name!
	area(r::Rectangle) = r.width * r.height
	area(x) = x
end

# ╔═╡ c34355b8-f6dd-11ea-1089-cf5be2117ba8
area(r)

# ╔═╡ c4a7fae6-f6dd-11ea-1851-3bd445ebf677
area(17)

# ╔═╡ c83180d8-f6dd-11ea-32f7-634b781070f1
md"

#

But since we didn't restrict the type in the second method, we also have"

# ╔═╡ d3dc64b6-f6dd-11ea-1273-7fb3957e4964
area("hello")

# ╔═╡ 6d7ca590-f6f5-11ea-0a00-6128f971b546
area

# ╔═╡ d6e5a276-f6dd-11ea-34aa-b9e2d3805364
md"which does not make much sense."

# ╔═╡ 79fa562e-f6dd-11ea-2e97-df3c62c83685
md"Note that these are different versions of the function with the *same* function name; each version acting on different (combinations of) types is called a **method** of the (generic) function."

# ╔═╡ a1d2ae6c-f6dd-11ea-0216-ef9db5d9e29b
md"Suppose that later we create a `Circle` type. We can then just add a new method `area(c::Circle)` with the corresponding definition, and Julia will continue to choose the correct version (method) when we call `area(x)`, depending on the type of `x`.

[Note that at the time of writing, Pluto requires all methods of a function to be defined in the same cell.]
"

# ╔═╡ 41e21994-f6de-11ea-0e5c-0515a3a52f6f
md"## Multiple dispatch


The act of choosing which function to call based on the type of the arguments that are passed to the function is called **dispatch**. A central feature of Julia is **multiple dispatch**: this choice is made based on the types of *all* the arguments to a function.

For example, the following three calls to the `+` function each call *different* methods, defined in different locations. Click on the links to see the definition of each method in the Julia source code on GitHub!"

# ╔═╡ 814e328c-f6de-11ea-13c0-d1b97714c4f3
cc = 3 + 4im

# ╔═╡ cdd3e14e-f6f5-11ea-15e2-bd309e658823
cc + cc

# ╔═╡ e01e26f2-f6f5-11ea-13b0-95413a6f7290
+

# ╔═╡ 84cae75c-f6de-11ea-3cd4-1b263e34771f
@which cc + cc

# ╔═╡ 8ac4904a-f6de-11ea-105b-8925016ca6d5
@which cc + 3

# ╔═╡ 8cd9f438-f6de-11ea-2b58-93bbb860a005
@which 3 + cc

# ╔═╡ 549c63e4-670c-11eb-2574-55611c612e43
md"
# A `WorkerSpell` Type

* Let's look at something a bit more relevant for us
* [Robin, Piyapromdee and Lentz](https://www.dropbox.com/s/4ylyn1v7fe0jmn1/Piyapromdee_Anatomy.pdf?dl=0) write *On Worker and Firm Heterogeneity in Wages and Employment Mobility: Evidence from Danish Register Data*
* Their data are *worker spells* at firms in appendix E.1
* They have $I = 4,000,000$ workers and $J=400,000$ firms

# What is a `Spell`?

It's a period of time that a worker spent at a certain firm, earning a certain wage.

1. Start/end date of spell
1. ID of worker $i$ and firm $j$
1. A vector of wages for each year in the spell $w_i$
1. An indicator whether the worker changes firm after this spell, $D_{it}$

# Why Not a Spreadsheet?

* They have a likelihood function to estimate which encodes for worker $i$ the likelihood of observing the data

$$(w_{it}, j_{it}, x_{it})_{t=1}^{T_i}$$

* Simplifying a bit, this looks for worker $i$ like

$$L_i = \Pi_{t=1}^{T_i} f(w_{it} | j_{it}, x_{it}) \times \Pi_{t=1}^{T_i} S(i,j_{it},x_{it})^{1-D_{it}} M(j'|i,j_{it},x_{it})^{D_{it}}$$

* Can you see the $T_i$ there? 
* 👉different workers have differently long spells!
"

# ╔═╡ 1a5ca388-670e-11eb-0321-51c90fcbef35
md"## Defining a `Spell`"

# ╔═╡ 0b9cb9f4-670f-11eb-338a-5d6c16c99248
mutable struct Spell 
	start       :: Date
	stop        :: Date
	duration    :: Week
	firm        :: Int   # ∈ 1,2,...,L+1
	wage        :: Float64
	change_firm :: Bool   # switch firm after this spell?
	function Spell(t0::Date,fid::Int)  # this is the `inner constructor` method
		this = new()
		this.start = t0
		this.stop = t0
		this.duration = Week(0)
		this.firm = fid
		this.wage = 0.0
		this.change_firm = false
		return this 
	end
end

# ╔═╡ b083edfe-67a3-11eb-0924-4f44343ffceb


# ╔═╡ c63a5110-670d-11eb-0502-5fc01e7d59d5
md"#

Let's create one of those:
"

# ╔═╡ 320edd04-670f-11eb-2a98-353ca1501537
sp = Spell(Date("2015-03-21"), 34)

# ╔═╡ 3d2a7a92-670f-11eb-265c-77d1d30af869
md"ok, great. now we need a way to set some infos on this type. In particular, we want to record the wage the worker got, and how long the spell lasted. Here is function to call at the end of a spell:"

# ╔═╡ 50b68470-670f-11eb-2fd2-e9ac408adad2
md"#"

# ╔═╡ 52ecd1f4-670f-11eb-1843-7380dcf1ee54
function finish!(s::Spell,w::Float64,d::Week)
    @assert d >= Week(0)
    s.stop = s.start + d
    s.duration = d
    s.wage = w
end

# ╔═╡ 6359fe04-670f-11eb-1b4e-7d4193109484


# ╔═╡ 5afaa42c-670f-11eb-1e96-afa2a6466656
md"
let's say that particular spell lasted for 14 weeks and was characterised by a wage of 100.1 Euros
"

# ╔═╡ 5f97e3f8-670f-11eb-2b7b-e54286c0cbb3
finish!(sp, 100.1, Week(14))

# ╔═╡ 622172a6-670f-11eb-1910-8fe1652cf65b
sp

# ╔═╡ 666fba34-670f-11eb-19b0-6f5046362154
md"# Workers

Next, we need a worker. Workers accumulate `Spells`.
"

# ╔═╡ ddfab162-670f-11eb-1bff-037e2e001178
mutable struct Worker
    id :: Int   # name
    T  :: Int   # number of WEEKS observed
    l  :: Int   # current firm ∈ 1,2,...,L+1
    spells :: Vector{Spell}  # an array of type Spell
    function Worker(id,T,start::Date,l::Int)
        this = new()
        this.id = id
        this.T = T 
        this.l = l
        this.spells = Spell[Spell(start,l)]
        return this
    end
end

# ╔═╡ 1b971326-6710-11eb-2b7b-eba640d686ba
md"
#

Let's create a worker and make him work at a new firm:
"

# ╔═╡ 27629a70-6710-11eb-0527-2fe3a1fd89b5
w0 = Worker(13, 18, Dates.today(), 3)

# ╔═╡ 7119d84c-6710-11eb-0afd-b3695a395be1
md"and let's say that first spell lasts for 15 weeks at 500 Euros per week"

# ╔═╡ a5ec451e-6710-11eb-0b52-95ef2e17082e
finish!(w0.spells[end], 500.0,  Week(15))

# ╔═╡ d8ed9ee0-6710-11eb-33a0-d5648e5e3891
w0.spells

# ╔═╡ 5291c44c-6725-11eb-3508-5113b947e07b
md"
# Useful?

When we want to evaluate the function $L_i$ above, we need to sum over each workers $T_i$ weeks of work." 

# ╔═╡ e5635fae-6725-11eb-3bcc-491220b5e24a
begin
	N = 5 # workers
	F = 2 # firms
end

# ╔═╡ b8c625c8-6725-11eb-3853-2916ae6f4af5
begin
	starts = rand(Date(2014,1,29):Week(1):Dates.today(),N)
	Ts = rand(5:10,N)
	js = rand(1:F,N)
	wrks = [Worker(id,Ts[id],starts[id],js[id]) for id in 1:N]
end

# ╔═╡ f521adde-6726-11eb-1904-5d4a8cd90746
md"
#
"

# ╔═╡ fa1c3a34-6726-11eb-333b-21b7f91d524c
md"
now let's set some wages on those workers. Let's say with prob 0.5 they have 2 spells:
"

# ╔═╡ a2df9a4e-6727-11eb-129b-9bd913058795
Ln = LogNormal(1.0,	0.1)

# ╔═╡ 050b2ba8-6727-11eb-1a9a-738f657d78e6
begin
	# an empty array of type worker
	dates = Date(2014,1,29):Week(1):Dates.today()
    workers = Worker[]
	for iw in 1:N
		w = Worker(iw,rand(5:10),rand(dates),rand(1:F))
		dur = 0 # start with zero weeks
		for tx in 1:w.T
			dur += 1 # increase duration
			if rand() < 0.5
				# switches to another firm
				finish!(w.spells[end], rand(Ln), Week(dur))
				dur = 0 # reset duration
				w.spells[end].change_firm = true
				# new spell starts on the same day!
				newfirm = rand(1:F)
				push!(w.spells, Spell(w.spells[end].stop,newfirm))
				w.l = newfirm
			else
				# nothing to record: stay at same firm
			end
			if tx == w.T
				# finish last spell
				finish!(w.spells[end], rand(Ln), Week(dur))
			end
		end
		push!(workers,w)
	end
end

# ╔═╡ b8ccbe4e-6728-11eb-1f1f-7367e85017c6
workers

# ╔═╡ e2b3e500-6728-11eb-3b01-f769c8a4c799
md"# Summing over the Likelihood

Then, finally, we can iterate over our array of workers like this
"

# ╔═╡ fc2f870c-6728-11eb-06ff-ab6affcfaefe
begin
	L0 = 0.0
	for iw in workers
		# loops over multiple spells
		# for each worker
		for (idx, sp) in enumerate(iw.spells)
			L0 += logpdf(Ln, sp.wage)
		end						
	end
end

# ╔═╡ 5d72de64-672a-11eb-2b64-e73f79c72d9c
L0

# ╔═╡ 7226446c-6d19-11eb-19be-a1c0bf338929
y = rand(3)

# ╔═╡ 7d5c2826-6d19-11eb-1d1f-dfade2b194fe
begin
for (iy, vy) in enumerate(y)
	println("iy = $iy, vy = $vy")
end
end

# ╔═╡ f1f6aa62-6d19-11eb-03a5-cf982339b606
z = 1

# ╔═╡ f920d7e0-6d19-11eb-24dc-056472aae857
z += 1

# ╔═╡ Cell order:
# ╟─4723a50e-f6d1-11ea-0b1c-3d33a9b92f87
# ╠═91ba119e-670a-11eb-0bb2-15ac7192e7e7
# ╟─92cef0a4-670a-11eb-2df2-5b31a1602299
# ╟─d19385e0-678a-11eb-3156-c51cb20abe3e
# ╟─082b1918-678b-11eb-1a78-2316091d6272
# ╟─e97a60e6-678c-11eb-0b37-fd80c3f0e8aa
# ╠═aa5adfba-678e-11eb-02e4-210b8b753e21
# ╠═bff062f0-678e-11eb-131d-e11297e36d4b
# ╟─bed2a154-f6d1-11ea-0812-1f5c628a9785
# ╠═81113974-f6d2-11ea-2ef8-fb2930402a74
# ╟─a4296474-6714-11eb-0500-1d888fe18b51
# ╟─ac321d80-f6d2-11ea-2951-676e6e1aef56
# ╟─9a168a92-670b-11eb-34af-0121d1a252bd
# ╠═66abc04e-f6d8-11ea-27d8-9f8b14659755
# ╟─7571be3a-f6d8-11ea-174c-9d65d5185153
# ╟─9f384ac2-f6d8-11ea-297e-4bf09acf9fe7
# ╠═b0dac516-f6d8-11ea-1bdb-b59723107206
# ╟─af236602-5ffa-11eb-0bec-cd944a602c70
# ╟─b98b9faa-f6d8-11ea-3610-bf8a84af2b5a
# ╠═cf1c4aae-f6d8-11ea-3200-c5fb458c7c09
# ╠═d0749974-f6d8-11ea-2f41-074b6744f3d5
# ╟─d372342c-f6d8-11ea-10cd-573cf7eab992
# ╠═dd8f9e88-f6d8-11ea-15e3-f17f4af0d81b
# ╠═e3e70064-f6d8-11ea-22fd-892bbc567ed4
# ╟─e582eb02-f6d8-11ea-1fcc-89bbc9dfbb07
# ╠═f2ed18b4-f6d8-11ea-3bc7-0b82eb5e8dc0
# ╟─f9d192fe-f6d8-11ea-138d-3dcdff33c034
# ╟─6085144c-f6db-11ea-19fe-ed46dafb4562
# ╟─1840898e-f6d9-11ea-3035-bb4dac496834
# ╠═63f76d28-f6d9-11ea-071c-458528c36008
# ╟─68934a2a-f6d9-11ea-37ea-850304f6d3d6
# ╟─a38a0164-f6d9-11ea-2e1d-a7a6e2106d0b
# ╟─3fa7d8e2-f6d9-11ea-2e82-9f59b5cb9424
# ╠═8c4beb6e-f6db-11ea-12d1-cf450181363b
# ╠═91e21d28-f6db-11ea-1b0f-336719682f28
# ╟─2ef7392e-f6dc-11ea-00e4-770cdf9a102e
# ╠═9371209e-f6db-11ea-3ba2-c3597d42d8ed
# ╠═b16a7348-6d16-11eb-1200-6967c76a9f5e
# ╟─d1166a30-670b-11eb-320d-9b13b96824fb
# ╟─b916acb4-f6dc-11ea-3cdf-2b8ab3c34e03
# ╠═6fe1b332-f6dd-11ea-39d4-1954aeda6f08
# ╠═c34355b8-f6dd-11ea-1089-cf5be2117ba8
# ╠═c4a7fae6-f6dd-11ea-1851-3bd445ebf677
# ╟─c83180d8-f6dd-11ea-32f7-634b781070f1
# ╠═d3dc64b6-f6dd-11ea-1273-7fb3957e4964
# ╠═6d7ca590-f6f5-11ea-0a00-6128f971b546
# ╟─d6e5a276-f6dd-11ea-34aa-b9e2d3805364
# ╟─79fa562e-f6dd-11ea-2e97-df3c62c83685
# ╟─a1d2ae6c-f6dd-11ea-0216-ef9db5d9e29b
# ╟─41e21994-f6de-11ea-0e5c-0515a3a52f6f
# ╠═814e328c-f6de-11ea-13c0-d1b97714c4f3
# ╠═cdd3e14e-f6f5-11ea-15e2-bd309e658823
# ╠═e01e26f2-f6f5-11ea-13b0-95413a6f7290
# ╠═84cae75c-f6de-11ea-3cd4-1b263e34771f
# ╠═8ac4904a-f6de-11ea-105b-8925016ca6d5
# ╠═8cd9f438-f6de-11ea-2b58-93bbb860a005
# ╟─549c63e4-670c-11eb-2574-55611c612e43
# ╟─1a5ca388-670e-11eb-0321-51c90fcbef35
# ╠═f855b1c0-670e-11eb-2a81-4de0846c0d93
# ╠═0b9cb9f4-670f-11eb-338a-5d6c16c99248
# ╠═b083edfe-67a3-11eb-0924-4f44343ffceb
# ╟─c63a5110-670d-11eb-0502-5fc01e7d59d5
# ╠═320edd04-670f-11eb-2a98-353ca1501537
# ╟─3d2a7a92-670f-11eb-265c-77d1d30af869
# ╟─50b68470-670f-11eb-2fd2-e9ac408adad2
# ╠═52ecd1f4-670f-11eb-1843-7380dcf1ee54
# ╠═6359fe04-670f-11eb-1b4e-7d4193109484
# ╟─5afaa42c-670f-11eb-1e96-afa2a6466656
# ╠═5f97e3f8-670f-11eb-2b7b-e54286c0cbb3
# ╠═622172a6-670f-11eb-1910-8fe1652cf65b
# ╟─666fba34-670f-11eb-19b0-6f5046362154
# ╠═ddfab162-670f-11eb-1bff-037e2e001178
# ╟─1b971326-6710-11eb-2b7b-eba640d686ba
# ╠═27629a70-6710-11eb-0527-2fe3a1fd89b5
# ╟─7119d84c-6710-11eb-0afd-b3695a395be1
# ╠═a5ec451e-6710-11eb-0b52-95ef2e17082e
# ╠═d8ed9ee0-6710-11eb-33a0-d5648e5e3891
# ╟─5291c44c-6725-11eb-3508-5113b947e07b
# ╠═e5635fae-6725-11eb-3bcc-491220b5e24a
# ╠═b8c625c8-6725-11eb-3853-2916ae6f4af5
# ╟─f521adde-6726-11eb-1904-5d4a8cd90746
# ╟─fa1c3a34-6726-11eb-333b-21b7f91d524c
# ╠═9e2fc76c-6727-11eb-2cab-7f81bdfb7343
# ╠═a2df9a4e-6727-11eb-129b-9bd913058795
# ╠═050b2ba8-6727-11eb-1a9a-738f657d78e6
# ╠═b8ccbe4e-6728-11eb-1f1f-7367e85017c6
# ╟─e2b3e500-6728-11eb-3b01-f769c8a4c799
# ╠═fc2f870c-6728-11eb-06ff-ab6affcfaefe
# ╠═f1f6aa62-6d19-11eb-03a5-cf982339b606
# ╠═f920d7e0-6d19-11eb-24dc-056472aae857
# ╠═5d72de64-672a-11eb-2b64-e73f79c72d9c
# ╠═7226446c-6d19-11eb-19be-a1c0bf338929
# ╠═7d5c2826-6d19-11eb-1d1f-dfade2b194fe
