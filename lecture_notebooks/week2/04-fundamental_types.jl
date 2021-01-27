### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 44b0626e-502f-4597-95d4-5f0a697cb54b
using LinearAlgebra, Statistics

# ╔═╡ 9da58875-31ef-4559-bf1d-54c43a0c5a79
md"""
# Arrays, Tuples, Ranges, and Other Fundamental Types

**way too long**
"""

# ╔═╡ db12d2e9-48e0-4c4a-9644-eabefda2338e
md"""
## Contents

- [Arrays, Tuples, Ranges, and Other Fundamental Types](#Arrays,-Tuples,-Ranges,-and-Other-Fundamental-Types)  
  - [Overview](#Overview)  
  - [Array Basics](#Array-Basics)  
  - [Operations on Arrays](#Operations-on-Arrays)  
  - [Ranges](#Ranges)  
  - [Tuples and Named Tuples](#Tuples-and-Named-Tuples)  
  - [Nothing, Missing, and Unions](#Nothing,-Missing,-and-Unions)  
  - [Exercises](#Exercises)  
  - [Solutions](#Solutions)  
  
**This is lecture is a slightly modified version of https://lectures.quantecon.org/jl/fundamental_types.html**
**Thank you to the amazing Quantecon.org team!**
"""

# ╔═╡ 941ec474-e349-43fb-8605-8aa9b8a459ec
md"""
## Overview

In Julia, arrays and tuples are the most important data type for working with numerical data

In this lecture we give more details on

- creating and manipulating Julia arrays  
- fundamental array processing operations  
- basic matrix algebra  
- tuples and named tuples  
- ranges  
- nothing, missing, and unions  
"""

# ╔═╡ f9fd1271-c0a0-4122-8de1-033ff30722c7
md"""
## Array Basics

([See multi-dimensional arrays documentation](https://docs.julialang.org/en/v1/manual/arrays/))

Since it is one of the most important types, we will start with arrays

Later, we will see how arrays (and all other types in Julia) are handled in a generic and extensible way
"""

# ╔═╡ 5a92c652-ba81-4032-9da2-5b1133e88414
md"""
### Shape and Dimension

We’ve already seen some Julia arrays in action
"""

# ╔═╡ 8b0eb882-e787-4be3-b786-475d4513e5f2
md"""
The output tells us that the arrays are of types `Array{Int64,1}` and `Array{Float64,1}` respectively

Here `Int64` and `Float64` are types for the elements inferred by the compiler

We’ll talk more about types later

The `1` in `Array{Int64,1}` and `Array{Any,1}` indicates that the array is
one dimensional (i.e., a `Vector`)

This is the default for many Julia functions that create arrays
"""

# ╔═╡ 6bd6d1e0-e161-46c9-b16e-b9d25e9f5f60
typeof(randn(100))

# ╔═╡ 4a22df91-ab96-4802-a27e-8ff93cce78a8
md"""
In Julia, one dimensional vectors are best interpreted as column vectors, which we will see when we take transposes

We can check the dimensions of `a` using `size()` and `ndims()`
functions
"""

# ╔═╡ b250805b-bfde-444a-9413-d67545be5618
md"""
The syntax `(3,)` displays a tuple containing one element – the size along the one dimension that exists
"""

# ╔═╡ fe3c213c-361b-4d7e-8409-41b0a0bf6aa9
md"""
#### Array vs Vector vs Matrix

In Julia, `Vector` and `Matrix` are just aliases for one- and two-dimensional arrays
respectively
"""

# ╔═╡ ba7d88fe-27aa-4137-b3d0-05a439546da9
begin
	Array{Int64, 1} == Vector{Int64}
	Array{Int64, 2} == Matrix{Int64}
end

# ╔═╡ 5e0e1163-a61d-42ec-ac58-9a6873bdb287
md"""
Vector construction with `,` is then interpreted as a column vector

To see this, we can create a column vector and row vector more directly
"""

# ╔═╡ b690c4e9-9837-4e44-b968-089530657da0
[1, 2, 3] == [1; 2; 3]  # both column vectors

# ╔═╡ a05c5668-8984-4216-a730-52d45c47f51e
[1 2 3]  # a row vector is 2-dimensional

# ╔═╡ 2ff3c0c8-9c2f-4b9d-aca4-e0eecf96b7b6
md"""
As we’ve seen, in Julia we have both

- one-dimensional arrays (i.e., flat arrays)  
- arrays of size `(1, n)` or `(n, 1)` that represent row and column vectors respectively  


Why do we need both?

On one hand, dimension matters for matrix algebra

- Multiplying by a row vector is different to multiplying by a column vector  


On the other, we use arrays in many settings that don’t involve matrix algebra

In such cases, we don’t care about the distinction between row and column vectors

This is why many Julia functions return flat arrays by default


<a id='creating-arrays'></a>
"""

# ╔═╡ 4f432530-26de-47bd-a835-0d3eb25bfe85
md"""
## Creating Arrays

#### Functions that Create Arrays

We’ve already seen some functions for creating a vector filled with `0.0`
"""

# ╔═╡ 4347049b-8258-4368-9699-c52ad9760c71
zeros(3)

# ╔═╡ 4f41d825-a0a2-48f5-805c-c5b47c41b0ea
# This generalizes to matrices and higher dimensional arrays
zeros(2, 2)

# ╔═╡ 603aafd9-6f27-48d7-b422-96a5442161d7
# To return an array filled with a single value, use `fill`
fill(5.0, 2, 2)

# ╔═╡ 826a2c3d-1372-414c-888c-08844e2d2561
md"""
The printed values you see here are just garbage values.

(the existing contents of the allocated memory slots being interpreted as 64 bit floats)

If you need more control over the types, fill with a non-floating point:
"""

# ╔═╡ 5963c2e5-1d14-4cac-8227-e5d8abaab22c
fill(0, 2, 2)  # fills with 0, not 0.0

# ╔═╡ ba70403a-3797-4244-b9e6-3bd0d3b77762
# Or fill with a boolean type
fill(false, 2, 2)  # produces a boolean matrix

# ╔═╡ 46a6a796-da2e-4b13-bcfa-7e7f9878e61a
md"""
#### Creating Arrays from Existing Arrays

For the most part, we will avoid directly specifying the types of arrays, and let the compiler deduce the optimal types on its own

The reasons for this, discussed in more detail in [this lecture](https://lectures.quantecon.org/jl/generic_programming.html#), are to ensure both clarity and generality

One place this can be inconvenient is when we need to create an array based on an existing array

First, note that assignment in Julia binds a name to a value, but does not make a copy of that type
"""

# ╔═╡ cf1ead00-f208-47af-8d46-b90871d62954
x = [1, 2, 3]
y = x
y[1] = 2
x

# ╔═╡ 5eba9af6-79f9-4fdf-bc7a-b5f025d0a581
md"""
In the above, `y = x` simply creates a new named binding called `y` which refers to whatever `x` currently binds to

To copy the data, you need to be more explicit
"""

# ╔═╡ 02693f8e-9065-40c1-b500-5c9fe798e179
x = [1, 2, 3]
y = copy(x)
y[1] = 2
x

# ╔═╡ c5979a79-03b2-474f-bce2-03701130b002
md"""
However, rather than making a copy of `x`, you may want to just have a similarly sized array
"""

# ╔═╡ fb9c2d57-25f2-40f8-9de2-863ca1efbd8a
x = [1, 2, 3]
y = similar(x)
y

# ╔═╡ 5f5776a3-9622-47b3-a236-84d32008bdad
md"""
We can also use `similar` to pre-allocate a vector with a different size, but the same shape
"""

# ╔═╡ 41a363e0-f792-42f7-8b5e-3fafc1f1d25d
x = [1, 2, 3]
y = similar(x, 4)  # make a vector of length 4

# ╔═╡ 8e81e25c-424d-44c1-8349-f08ed79109ae
# higher dimensions
x = [1, 2, 3]
y = similar(x, 2, 2)  # make a 2x2 matrix

# ╔═╡ b122df17-37d8-4b2f-afad-b31829eaefab
md"""
#### Manual Array Definitions

As we’ve seen, you can create one dimensional arrays from manually specified data like so
"""

# ╔═╡ 2a8c5ae3-b8fb-4412-9b62-15f9151640cf
md"""
In two dimensions we can proceed as follows
"""

# ╔═╡ ddb2af37-70e9-40fe-a569-4673db40eccd
md"""
You might then assume that `a = [10; 20; 30; 40]` creates a two dimensional column vector but this isn’t the case
"""

# ╔═╡ b57b246e-e946-4d48-a62c-7294f9aea32d
md"""
Instead transpose the matrix (or adjoint if complex)
"""

# ╔═╡ 35bc68b2-2471-4e34-a4eb-5f0675fd3387
md"""
### Array Indexing

We’ve already seen the basics of array indexing
"""

# ╔═╡ 2e5669fa-4cdf-4635-b300-4d956a394465
a = [10 20 30 40]
a[end-1]

# ╔═╡ 4d8ead57-db0a-497e-be77-27b3e2cd4495
md"""
For 2D arrays the index syntax is straightforward
"""

# ╔═╡ c8215492-ae31-4954-90c0-6fe2edb01a29
a = randn(2, 2)
a[1, 1]

# ╔═╡ 46675778-edf2-4278-bb34-2bf2da1aba88
md"""
Booleans can be used to extract elements
"""

# ╔═╡ d70e0451-6be0-4dc6-8a77-eaa1a06d5f08
md"""
This is useful for conditional extraction, as we’ll see below

An aside: some or all elements of an array can be set equal to one number using slice notation
"""

# ╔═╡ 76704aa7-968d-44e1-95bd-63b6ede7fe2f
md"""
### Views and Slices

Using the `:` notation provides a slice of an array, copying the sub-array to a new array with a similar type
"""

# ╔═╡ dc3475a5-3fe9-4399-9233-7edbfb47bdda
a = [1 2; 3 4]
b = a[:, 2]
@show b
a[:, 2] = [4, 5] # modify a
@show a
@show b;

# ╔═╡ 0ed1ec30-2882-4a28-9a2d-9bb5a781b80c
md"""
A **view** on the other hand does not copy the value
"""

# ╔═╡ 8073c593-fcbf-4cf8-b895-0f1d6223b25f
a = [1 2; 3 4]
@views b = a[:, 2]
@show b
a[:, 2] = [4, 5]
@show a
@show b;

# ╔═╡ f20373ed-3cbf-490c-8aff-6ec5667a2f20
md"""
Note that the only difference is the `@views` macro, which will replace any slices with views in the expression

An alternative is to call the `view` function directly – though it is generally discouraged since it is a step away from the math
"""

# ╔═╡ 839fa9d6-f07f-4e76-a97e-c24a86811c9d
@views b = a[:, 2]
view(a, :, 2) == b

# ╔═╡ 18697bff-88ab-4295-b33e-a86e386a8e5c
md"""
As with most programming in Julia, it is best to avoid prematurely assuming that `@views` will have a significant impact on performance, and stress code clarity above all else

Another important lesson about `@views` is that they **are not** normal, dense arrays
"""

# ╔═╡ c5e54a82-1ad5-4b00-87f7-b372115e84c7
a = [1 2; 3 4]
b_slice = a[:, 2]
@show typeof(b_slice)
@show typeof(a)
@views b = a[:, 2]
@show typeof(b);

# ╔═╡ 09462935-5aa9-4f70-b670-8c306d094433
md"""
The type of `b` is a good example of how types are not as they may seem

Similarly
"""

# ╔═╡ 71cc1049-6f38-4fd9-a77a-3a967a83e128
a = [1 2; 3 4]
b = a'   # transpose
typeof(b)

# ╔═╡ d93baea1-38ab-424f-a5ee-2d30177f7c71
md"""
To copy into a dense array
"""

# ╔═╡ cfefbb99-454e-4c87-984e-6526ebb09ef4
a = [1 2; 3 4]
b = a' # transpose
c = Matrix(b)  # convert to matrix
d = collect(b) # also `collect` works on any iterable
c == d

# ╔═╡ 6ee0e819-00ed-4bf6-9718-f93441eaab68
md"""
### Special Matrices

As we saw with `transpose`, sometimes types that look like matrices are not stored as a dense array

As an example, consider creating a diagonal matrix
"""

# ╔═╡ f379c869-6f94-4c45-9e6c-d49b797db93d
d = [1.0, 2.0]
a = Diagonal(d)

# ╔═╡ aa65bbd4-cc93-4aa1-aaee-e3f87d57f596
md"""
As you can see, the type is `2×2 Diagonal{Float64,Array{Float64,1}}`, which is not a 2-dimensional array

The reasons for this are both efficiency in storage, as well as efficiency in arithmetic and matrix operations

In every important sense, matrix types such as `Diagonal` are just as much a “matrix” as the dense matrices we have using (see the [introduction to types lecture](https://lectures.quantecon.org/jl/introduction_to_types.html#) for more)
"""

# ╔═╡ 52482783-2a81-40d9-a056-0bcd4a86bf95
@show 2a
b = rand(2,2)
@show b * a;

# ╔═╡ 7bea4cb4-da56-4841-94ee-2d36c51c06d1
md"""
Another example is in the construction of an identity matrix, where a naive implementation is
"""

# ╔═╡ 98a9f682-14d4-47b9-be58-e5fee7275cfa
b = [1.0 2.0; 3.0 4.0]
b - Diagonal([1.0, 1.0])  # poor style, inefficient code

# ╔═╡ 717c24ac-939e-450c-a816-6ae08dd0cf57
# should to this instead
b = [1.0 2.0; 3.0 4.0]
b - I  # good style, and note the lack of dimensions of I

# ╔═╡ d9cb4a5a-b26b-4473-b9e3-5f1bdb7fea56
md"""
While the implementation of `I` is a little abstract to go into at this point, a hint is:
"""

# ╔═╡ fd2caba5-c543-492b-ab1e-90a691e08c3b
typeof(I)

# ╔═╡ 0e5a392b-93ed-4688-a6a6-03c2023ddd1f
md"""
This is a `UniformScaling` type rather than an identity matrix, making it much more powerful and general
"""

# ╔═╡ 0ca09099-a5b0-4490-b03f-d7ab1e6e1285
md"""
### Assignment and Passing Arrays

As discussed above, in Julia, the left hand side of an assignment is a “binding” or a label to a value
"""

# ╔═╡ d678cb9f-eda8-448b-bb0d-7243f409d4d9
x = [1 2 3]
y = x  # name `y` binds to whatever value `x` bound to

# ╔═╡ 915ec462-f1ad-4e17-ba5d-e382d107d570
md"""
The consequence of this, is that you can re-bind that name
"""

# ╔═╡ e4336818-7eb8-4d6f-be1a-9dd3aea9b6df
x = [1 2 3]
y = x        # name `y` binds to whatever `x` bound to
z = [2 3 4]
y = z        # only changes name binding, not value!
@show (x, y, z);

# ╔═╡ 9bfad4b6-cdf4-4324-be66-b176d211ca28
md"""
What this means is that if `a` is an array and we set `b = a` then `a` and `b` point to **exactly the same data in your RAM**!

In the above, suppose you had meant to change the value of `x` to the values of `y`, you need to assign the values rather than the name
"""

# ╔═╡ 0bf8f290-f9bc-4ba5-bebf-ddbb3e34db77
x = [1 2 3]
y = x       # name `y` binds to whatever `x` bound to
z = [2 3 4]
y .= z      # now dispatches the assignment of each element
@show (x, y, z);

# ╔═╡ 8dfd8d68-9e58-4c1a-a24b-1f145fd9c46e
md"""
Alternatively, you could have used `y[:] = z`

This applies to in-place functions as well

First, define a simple function for a linear map
"""

# ╔═╡ bb264090-c5e8-4f4a-9085-547d4fcf71d2
function f(x)
    return [1 2; 3 4] * x  # matrix * column vector
end

val = [1, 2]
f(val)

# ╔═╡ 92df24e5-7dfb-466b-b3f2-a028cdbd6b33
md"""
In general, these “out-of-place” functions are preferred to “in-place” functions, which modify the arguments
"""

# ╔═╡ 0d9a11d8-63ad-4fde-8103-b464a4b5b23d
function f(x)
    return [1 2; 3 4] * x # matrix * column vector
end

val = [1, 2]
y = similar(val)

function f!(out, x)
    out .= [1 2; 3 4] * x
end

f!(y, val)
y

# ╔═╡ 5bf57b84-4d38-4b83-bc47-42d397b334b1
md"""
This demonstrates a key convention in Julia: functions which modify any of the arguments have the name ending with `!` (e.g. `push!`)

We can also see a common mistake, where instead of modifying the arguments, the name binding is swapped
"""

# ╔═╡ ca5456a4-20f2-4f23-b1d3-ffa626cd6982
function f(x)
    return [1 2; 3 4] * x  # matrix * column vector
end

val = [1, 2]
y = similar(val)

function f!(out, x)
    out = [1 2; 3 4] * x   # MISTAKE! Should be .= or [:]
end
f!(y, val)
y

# ╔═╡ ae567de9-2ac1-4675-9d64-02cdc53e73f2
md"""
### In-place and Immutable Types

Note that scalars are always immutable, such that
"""

# ╔═╡ a2d98af2-90f2-4725-a4e3-b2f3d9a9d0dd
y = [1 2]
y .-= 2    # y .= y .- 2, no problem

x = 5
# x .-= 2  # Fails!
x = x - 2  # subtle difference - creates a new value and rebinds the variable

# ╔═╡ 7e2d765b-696a-4bd9-9e31-00094028065b
md"""
In particular, there is no way to pass any immutable into a function and have it modified
"""

# ╔═╡ 60db9959-705c-40e1-9e7a-1d477d645c60
x = 2

function f(x)
    x = 3     # MISTAKE! does not modify x, creates a new value!
end

f(x)          # cannot modify immutables in place
@show x;

# ╔═╡ 65279ef8-4ee7-45b5-9789-8aaeac9f33db
md"""
This is also true for other immutable types such as tuples, as well as some vector types
"""

# ╔═╡ adf05e01-9831-4b09-8527-bdd631966ed1
using StaticArrays
xdynamic = [1, 2]
xstatic = @SVector [1, 2]  # turns it into a highly optimized static vector

f(x) = 2x
@show f(xdynamic)
@show f(xstatic)

# inplace version
function g(x)
    x .= 2x
    return "Success!"
end
@show xdynamic
@show g(xdynamic)
@show xdynamic;

# g(xstatic) # fails, static vectors are immutable

# ╔═╡ 81b2de8d-07b7-440b-8614-c09021a0666f
md"""
## Operations on Arrays
"""

# ╔═╡ c7a26040-24f2-4c87-97b5-d9280735fc17
md"""
### Array Methods

Julia provides standard functions for acting on arrays, some of which we’ve
already seen
"""

# ╔═╡ cf1a068d-7332-4a66-9b65-ab3c67740c45
a = [-1, 0, 1]

@show length(a)
@show sum(a)
@show mean(a)
@show std(a)      # standard deviation
@show var(a)      # variance
@show maximum(a)
@show minimum(a)
@show extrema(a)  # (mimimum(a), maximum(a))

# ╔═╡ dda39690-e65a-43a4-a609-f98156b325b0
md"""
To sort an array
"""

# ╔═╡ 8acb096f-9566-42b9-a7db-3c17b4584768
md"""
### Matrix Algebra

For two dimensional arrays, `*` means matrix multiplication
"""

# ╔═╡ 7c7e1986-3ddc-47e4-8707-3812a42a9124
md"""
To solve the linear system $ A X = B $ for $ X $ use `A \ B`
"""

# ╔═╡ de2063fc-c339-41b0-a5b3-916822ae2e23
B = ones(2, 2)

# ╔═╡ d0cbb6fc-f560-4a18-8d38-b46685a051f0
md"""
Although the last two operations give the same result, the first one is numerically more stable and should be preferred in most cases

Multiplying two **one** dimensional vectors gives an error – which is reasonable since the meaning is ambiguous
"""

# ╔═╡ ff8f7b98-2d35-4b14-a34d-28ac82a9e9b7
ones(2) * ones(2)

# ╔═╡ 428dadd1-1169-476a-bac8-62ac839c7131
md"""
If you want an inner product in this setting use `dot()` or the unicode `\cdot<TAB>`
"""

# ╔═╡ f9eeacef-cf8e-48d6-bff9-9578376a0f06
dot(ones(2), ones(2))

# ╔═╡ 72c026a0-6efe-4456-b73e-78c6d41d0adb
md"""
Matrix multiplication using one dimensional vectors is a bit inconsistent –
pre-multiplication by the matrix is OK, but post-multiplication gives an error
"""

# ╔═╡ dd6e792a-7779-4907-af5b-4609b20e57df
md"""
### Elementwise Operations

#### Algebraic Operations

Suppose that we wish to multiply every element of matrix `A` with the corresponding element of matrix `B`

In that case we need to replace `*` (matrix multiplication) with `.*` (elementwise multiplication)

For example, compare
"""

# ╔═╡ 4e62792b-2d9d-4759-928c-c74f0de1559c
ones(2, 2) * ones(2, 2)   # matrix multiplication

# ╔═╡ 114f341c-5da5-4e3d-9078-9331f8e72ac9
ones(2, 2) .* ones(2, 2)   # element by element multiplication

# ╔═╡ 34954704-6042-4d41-937d-a4f590842a52
md"""
* This is a general principle: `.x` means apply operator `x` elementwise
* We have seen the `.x` before when talking about **Broadcasting**
* You remember that `.op(x)` this just applies operation `op` to all elements of argument `x`.
"""

# ╔═╡ 1bdc51b1-bb4a-4f27-88ab-75d66908a471
md"""
However in practice some operations are mathematically valid without broadcasting, and hence the `.` can be omitted
"""

# ╔═╡ cf7859c5-895b-4067-a6ff-89e9c656da0c
ones(2, 2) + ones(2, 2)  # same as ones(2, 2) .+ ones(2, 2)

# ╔═╡ 5a6a515b-2af4-436d-8668-df5d20e31041
md"""
Scalar multiplication is similar
"""

# ╔═╡ d3e1568b-8152-4f8a-8fe0-2e27e5434d3c
md"""
In fact you can omit the `*` altogether and just write `2A`

Unlike MATLAB and other languages, scalar addition requires the `.+` in order to correctly broadcast
"""

# ╔═╡ 9a769806-ae4c-44a1-b6b4-6da7a9bb9895
x = [1, 2]
x .+ 1     # not x + 1
x .- 1     # not x - 1

# ╔═╡ 8384a0b3-8f0b-4a4e-b041-d5129f517998
md"""
#### Elementwise Comparisons

Elementwise comparisons also use the `.x` style notation
"""

# ╔═╡ ceca8038-4054-47dc-bceb-8867cd23b606
md"""
We can also do comparisons against scalars with parallel syntax
"""

# ╔═╡ c3fed55f-3397-421d-b361-950673e4a5c8
md"""
This is particularly useful for *conditional extraction* – extracting the elements of an array that satisfy a condition
"""

# ╔═╡ c9d9ec45-5376-403d-9ede-f6ed27b8f874
md"""
#### Changing Dimensions

The primary function for changing the dimensions of an array is `reshape()`
"""

# ╔═╡ 9f911a3f-7dff-45af-928c-7371dce6a24c
md"""
Notice that this function returns a `view` on the existing array

This means that changing the data in the new array will modify the data in the
old one!
"""

# ╔═╡ 7099a6e3-b1f3-4ab9-8d2c-ff8f305f611c
md"""
To collapse an array along one dimension you can use `dropdims()`
"""

# ╔═╡ d6182b58-8a38-413d-b193-1f4e3272af7d
md"""
### Broadcasting Functions

Julia provides standard mathematical functions such as `log`, `exp`, `sin`, etc.
"""

# ╔═╡ 92f247b5-799e-4818-b28e-37730a8e7174
log(1.0)

# ╔═╡ 7b3699c5-5dc7-4413-b7da-e30c8fa5167d
md"""
By default, these functions act *elementwise* on arrays
"""

# ╔═╡ bed57736-88e9-44da-8b67-6d510790866f
log.(1:4)

# ╔═╡ e5834ec8-11c9-4e64-9bba-553283b8b3a7
md"""
Note that we can get the same result as with a comprehension or more explicit loop
"""

# ╔═╡ 7dd06041-99fa-4a18-8db6-4164adeb34c4
[ log(x) for x in 1:4 ]

# ╔═╡ 2854b3ea-7961-4fed-8190-0d92b105711a
md"""
### Comprehensions and Generators

* Those are very convenient for us to set up arrays
* consider those examples
"""

# ╔═╡ 72933591-adfd-4cba-a67c-9d10d8b4fb7e
# A Comprehension is in square brackets:
# LinSpace is depreciated in Julia v1.3

foo(x,y,z) = sin(x)*0.5y^0.5 + z^2
d = [foo(i,j,k) for i in 1:3, j in range(0.01,stop=0.1,length=5), k in [log(l) for l in 2:7]];

# ╔═╡ d04d06bc-16c1-4482-b1cf-894dd3e1fe0e
# generator expressions work in a similar way
# just leave the square brackets away
sum(1/n^2 for n=1:1000)  # this allocates no temp array: very efficient!

# ╔═╡ a6cb8a08-c69c-4d96-a2ef-5d9abce4068e
# can have indices depend on each other
[(i,j) for i=1:3 for j=1:i]   

# ╔═╡ 7b89a0df-5ad9-4bfe-a587-8b9af9415732
# you can even condition on the indices
[(i,j) for i=1:3 for j=1:i if i+j == 4]

# ╔═╡ faa0cbe8-6b52-4365-8bb9-22a3a61a53b1
md"""
### Linear Algebra

([See linear algebra documentation](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/))

Julia provides some a great deal of additional functionality related to linear operations
"""

# ╔═╡ e1ee5121-fc52-48a3-9d02-bc95ef42159e
A \ B

# ╔═╡ 10a82b79-30e8-4bd7-961f-b5e34dc85cf5
inv(A) * B

# ╔═╡ 94356303-7f9b-4835-af02-3e60e32e24e3
A.^2  # square every element

# ╔═╡ 1061ecdc-506d-48aa-8362-74b50265359c
2 * A  # same as 2 .* A

# ╔═╡ 2aec79bf-e6a0-43fb-8500-3b6374a867b1
det(A)

# ╔═╡ 2a75e320-7b10-4aba-aa62-9de17c3b760c
tr(A)

# ╔═╡ e9f8b372-de3d-4dc4-9216-91ff279ed0ea
eigvals(A)

# ╔═╡ 07e16567-5b5d-4add-9bc5-428bcc35b5e4
rank(A)

# ╔═╡ c1e4ec26-28b2-49ee-a5a4-d405d04c4896
md"""
## Ranges

As with many other types, a `Range` can act as a vector
"""

# ╔═╡ 97e0ad1d-6ddc-476c-b4c9-73e3ff1e6f96
a = 10:12        # a range, equivalent to 10:1:12
@show Vector(a)  # can convert, but shouldn't

b = Diagonal([1.0, 2.0, 3.0])
b * a .- [1.0; 2.0; 3.0]

# ╔═╡ 74889209-6203-4b04-aa83-e6576f0abd0c
ndims(a)

# ╔═╡ 71ef3e30-57ab-4765-86f2-56ced72edea3
size(a)

# ╔═╡ 75beef5e-0437-4d9d-90eb-de7573e722e3
ndims(a)

# ╔═╡ 033498e2-cea8-4f04-9baa-e00a46a77c5a
ndims(a)

# ╔═╡ 16e8ece4-324c-4dbc-a6e5-69f287f12272
ndims(a)

# ╔═╡ beea5832-2ad4-46a3-a80d-1284e96bdf41
a[1:3]

# ╔═╡ fa97d1c1-146d-4f93-b9c7-fd9de4b0e957
a[1, :]  # first row

# ╔═╡ 4186d1f8-f1df-45e3-a6be-0dd294870670
a[:, 1]  # first column

# ╔═╡ 46637100-b299-4d73-8518-57d6449ac8c6
a[2:end] .= 42

# ╔═╡ 588897de-f6ee-4f7a-a50f-e1403ccd241f
a

# ╔═╡ 01e2193f-dd92-4bc1-b3cc-65d5cf6e2eab
a .< 0

# ╔═╡ a46dde55-16f0-4d27-adeb-0b4af0e9da33
a[a .< 0]

# ╔═╡ 0db1740f-8023-4db9-9c88-8fb7f3ceaf9c
a[b]

# ╔═╡ 75207a4f-fa1b-4482-a9d8-9bb2dd67b7b5
b == a  # tests if have the same values

# ╔═╡ 0de1b7ed-cc42-40be-a27d-5ec8cdf6747b
b === a  # tests if arrays are identical (i.e share same memory)

# ╔═╡ 01246a47-677c-47b1-8212-b9b165e3e91c
a * b

# ╔═╡ e990a843-e839-43c6-b6b1-7ffea485be5b
b * a'

# ╔═╡ 57666329-a8a7-4297-818e-e62a5eaa9b17
b * ones(2)

# ╔═╡ 439d050f-1dc9-465f-a296-4614e4540776
ones(2,2) * b

# ╔═╡ dcb101c7-5612-49e4-9d67-be4fdee8c996
b .> a

# ╔═╡ ab90940a-cea0-4901-8b47-828bc88b6a3c
a .== b

# ╔═╡ 2aadea95-693a-4fed-b218-4398a5560d50
b

# ╔═╡ 67ab25f9-ea3c-489e-88e2-d0dc4f7a3fe4
b .> 1

# ╔═╡ 3d2fdafa-f484-40bd-a8bf-208c5a08701e
b

# ╔═╡ 7c3d36f2-efce-46a4-8b93-71d43dd8c121
b[1, 1] = 100  # continuing the previous example

# ╔═╡ d6e2493b-a250-4cb3-823a-6cc20962c36b
b

# ╔═╡ cb9b754a-656a-45ac-8e72-e6bfa56e1af4
a

# ╔═╡ e0528017-3547-457b-b839-c9c76ad6973c
# The return value is an array with the specified dimension “flattened”
dropdims(a, dims = 1)

# ╔═╡ 8195212c-6c41-4e8e-92c0-da37e173c666
md"""
But care should be taken if the terminal node is not a multiple of the set sizes
"""

# ╔═╡ 04e6aa99-f617-4093-91c2-87e7c959fbe3
maxval = 1.0
minval = 0.0
stepsize = 0.15
a = minval:stepsize:maxval # 0.0, 0.15, 0.3, ...
maximum(a) == maxval

# ╔═╡ ae198ca4-cb31-469a-ab48-03747c41f785
md"""
To evenly space points where the maximum value is important, i.e., `linspace` in other languages
"""

# ╔═╡ 238488fd-466b-4931-80dc-2663a622c2c6
maxval = 1.0
minval = 0.0
numpoints = 10
a = range(minval, maxval, length=numpoints)
# or range(minval, stop=maxval, length=numpoints)

maximum(a) == maxval

# ╔═╡ 1e0851c9-840b-4636-8205-3357d9c81921
md"""
* For the `range(minval, maxval, length=numpoints)` notation, until the release of Julia v1.1, you will need to have the `using Compat` in the header, as we do above.
* Absent that, and until then, you have to supply the keyword `stop`.
"""

# ╔═╡ 1bf1a0da-cfac-4e7e-8e9a-36982953eaf0
md"""
## Tuples and Named Tuples

([See tuples](https://docs.julialang.org/en/v1/manual/functions/#Tuples-1) and [named tuples documentation](https://docs.julialang.org/en/v1/manual/functions/#Named-Tuples-1))

We were introduced to tuples earlier, which provide high-performance immutable sets of distinct types
"""

# ╔═╡ 94ff9eeb-78d4-41c2-bd69-5d536734301b
t = (1.0, "test")
t[1]            # access by index
a, b = t        # unpack
# t[1] = 3.0    # would fail as tuples are immutable
println("a = $a and b = $b")

# ╔═╡ 548f8f6c-7977-4cbb-941f-a56e000aa378
md"""
As well as **named tuples**, which extend tuples with names for each argument
"""

# ╔═╡ 7c435b8d-daf5-47ec-90ff-702ffdf1ea4d
t = (val1 = 1.0, val2 = "test")
t.val1      # access by index
# a, b = t  # bad style, better to unpack by name with @unpack
println("val1 = $(t.val1) and val1 = $(t.val1)") # access by name

# ╔═╡ be942bce-68b4-4024-8c96-4c14e03f41c4
md"""
While immutable, it is possible to manipulate tuples and generate new ones
"""

# ╔═╡ 14c7ba45-a711-447a-8622-bd7a48f01eb5
t2 = (val3 = 4, val4 = "test!!")
t3 = merge(t, t2)  # new tuple

# ╔═╡ e9220511-2ef1-4bc7-81dd-bf1c0bc24a95
md"""
Named tuples are a convenient and high-performance way to manage and unpack sets of parameters
"""

# ╔═╡ 25cfe013-ceab-4312-bc1f-60e7c9f4824d
function f(parameters)
    α, β = parameters.α, parameters.β  # poor style
    # poor because we'll make errors once
    # we add more parameters!
    return α + β
end

parameters = (α = 0.1, β = 0.2)
f(parameters)

# ╔═╡ e92806e3-3ee2-4f4f-ad46-d6e5a7c3d6a0
md"""
This functionality is aided by the `Parameters.jl` package and the `@unpack` macro
"""

# ╔═╡ 7cf94b4a-bed3-4838-b0bf-8e65348133cb
using Parameters

function f(parameters)
    @unpack α, β = parameters  # good style, less sensitive to errors
    return α + β
end

parameters = (α = 0.1, β = 0.2)
f(parameters)

# ╔═╡ 8fc78255-f690-487a-93cf-efad5a697f78
md"""
In order to manage default values, use the `@with_kw` macro
"""

# ╔═╡ 51ac074e-3af8-404d-8251-d21d97d741a2
using Parameters
paramgen = @with_kw (α = 0.1, β = 0.2)  # create named tuples with defaults

# creates named tuples, replacing defaults
@show paramgen()  # calling without arguments gives all defaults
@show paramgen(α = 0.2)
@show paramgen(α = 0.2, β = 0.5);

# ╔═╡ e195e763-27c0-4e5d-93f6-985c830aa6ed
md"""
An alternative approach, defining a new type using `struct` tends to be more prone to accidental misuse, and leads to a great deal of boilerplate code

For that, and other reasons of generality, we will use named tuples for collections of parameters where possible
"""

# ╔═╡ 1788e724-62d7-42e2-9bc8-0c3b02b3d4a0
md"""
## Nothing, Missing, and Unions

Sometimes a variable, return type from a function, or value in an array needs to represent the absence of a value rather than a particular value

There are two distinct use cases for this

1. `nothing` (“software engineers null”): used where no value makes sense in a particular context due to a failure in the code, a function parameter not passed in, etc.  
1. `missing` (“data scientists null”): used when a value would make conceptual sense, but it isn’t available  



<a id='error-handling'></a>
"""

# ╔═╡ afc582c7-8afe-4ee6-afbc-afeb4ccb6ed4
md"""
### Nothing and Basic Error Handling

The value `nothing` is a single value of type `Nothing`
"""

# ╔═╡ 3f9abc19-7f0a-4c8d-a9b9-bda04e0f5e6e
typeof(nothing)

# ╔═╡ dde2249f-bf27-4680-84fe-548b87b195b9
md"""
An example of a reasonable use of `nothing` is if you need to have a variable defined in an outer scope, which may or may not be set in an inner one
"""

# ╔═╡ e019042f-5899-4c61-9445-163b105493c5
function f(y)
    x = nothing
    if y > 0.0
        # calculations to set `x`
        x = y
    end

    # later, can check `x`
    if x === nothing
        println("x was not set")
    else
        println("x = $x")
    end
    x
end

@show f(1.0)
@show f(-1.0);

# ╔═╡ c46b746e-8e09-4ef5-a80b-1f6e587fae84
md"""
While in general you want to keep a variable name bound to a single type in Julia, this is a notable exception

Similarly, if needed, you can return a `nothing` from a function to indicate that it did not calculate as expected
"""

# ╔═╡ 21f5ad82-c94f-4985-91ca-068341738ba8
function f(x)
    if x > 0.0
        return sqrt(x)
    else
        return nothing
    end
end
x1 = 1.0
x2 = -1.0
y1 = f(x1)
y2 = f(x2)

# check results with === nothing
if y1 === nothing
    println("f($x2) successful")
else
    println("f($x2) failed");
end

# ╔═╡ c0ffb97f-a6f7-4391-b27f-8ea7349532ce
md"""
As an aside, an equivalent way to write the above function is to use the
[ternary operator](https://docs.julialang.org/en/v1/manual/control-flow/index.html#man-conditional-evaluation-1),
which gives a compact if/then/else structure
"""

# ╔═╡ 7bdeca04-fcc0-42ce-8d45-12572cd37358
function f(x)
    x > 0.0 ? sqrt(x) : nothing  # the "a ? b : c" pattern is the ternary
end

f(1.0)

# ╔═╡ 1d96bf36-00ae-485d-a338-520b458d0f4d
md"""
We will sometimes use this form when it makes the code more clear (and it will occasionally make the code higher performance)

Regardless of how `f(x)` is written,  the return type is an example of a union, where the result could be one of an explicit set of types

In this particular case, the compiler would deduce that the type would be a `Union{Nothing,Float64}` – that is, it returns either a floating point or a `nothing`

You will see this type directly if you use an array containing both types
"""

# ╔═╡ 44ca4731-cb01-4528-a201-214d4e9ca3b7
md"""
When considering error handling, whether you want a function to return `nothing` or simply fail depends on whether the code calling `f(x)` is carefully checking the results

For example, if you were calling on an array of parameters where a priori you were not sure which ones will succeed, then
"""

# ╔═╡ ef94faf8-38d4-465f-baef-942cb1a71144
x = [0.1, -1.0, 2.0, -2.0]
y = f.(x)

# presumably check `y`

# ╔═╡ ca15a5c5-b567-49dc-924b-8b1843b18747
md"""
On the other hand, if the parameter passed is invalid and you would prefer not to handle a graceful failure, then using an assertion is more appropriate
"""

# ╔═╡ f6a9edbb-b7b1-4ba5-ab99-24247843f7ea
function f(x)
    @assert x > 0.0
    sqrt(x)
end

f(1.0)

# ╔═╡ 1a226619-3655-4685-b96c-b58a1c7ae52a
md"""
Finally, `nothing` is a good way to indicate an optional parameter in a function
"""

# ╔═╡ 4349f64c-8892-4616-9b27-8205579c59ea
function f(x; z = nothing)

    if(z === nothing)
        println("No z given with $x")
    else
        println("z = $z given with $x")
    end
end

f(1.0)
f(1.0, z=3.0)

# ╔═╡ d7a59a0f-8de8-42c4-afbe-043d92d5e334
md"""
An alternative to `nothing`, which can be useful and sometimes higher performance,
is to use `NaN` to signal that a value is invalid returning from a function
"""

# ╔═╡ 3ec97746-d1b2-45df-bf2d-44c49a5b1c6c
function f(x)
    if x > 0.0
        return x
    else
        return NaN
    end
end

f(0.1)
f(-1.0)

@show typeof(f(-1.0))
@show f(-1.0) == NaN  # note, this fails!
@show isnan(f(-1.0))  # check with this

# ╔═╡ bcf973b7-20ad-41ed-b9e2-a6c73c9e711e
md"""
Note that in this case, the return type is `Float64` regardless of the input for `Float64` input

Keep in mind, though, that this only works if the return type of a function is `Float64`
"""

# ╔═╡ 77f89197-4017-49d6-a329-06ba0dc75910
md"""
### Exceptions

(See [exceptions documentation](https://docs.julialang.org/en/v1/manual/control-flow/index.html#Exception-Handling-1))

While returning a `nothing` can be a good way to deal with functions which may or may not return values, a more robust error handling method is to use exceptions

Unless you are writing a package, you will rarely want to define and throw your own exceptions, but will need to deal with them from other libraries

The key distinction for when to use an exceptions vs. return a `nothing` is whether an error is unexpected rather than a normal path of execution

An example of an exception is a `DomainError`, which signifies that a value passed to a function is invalid
"""

# ╔═╡ 8577bffe-96f0-4b7a-aab2-9755dc334545
# throws exception, turned off to prevent breaking notebook
# sqrt(-1.0)

# to see the error
try sqrt(-1.0); catch err; err end  # catches the exception and prints it

# ╔═╡ be0df836-c55a-4832-b0b2-2faa08ea3028
md"""
Another example you will see is when the compiler cannot convert between types
"""

# ╔═╡ 8b0251c1-61ae-4fd8-a856-4fb9f925e2f9
# throws exception, turned off to prevent breaking notebook
# convert(Int64, 3.12)

# to see the error
try convert(Int64, 3.12); catch err; err end  # catches the exception and prints it.

# ╔═╡ d14ca331-a60f-4524-8c1e-62f6f2358740
md"""
If these exceptions are generated from unexpected cases in your code, it may be appropriate simply let them occur and ensure you can read the error

Occasionally you will want to catch these errors and try to recover, as we did above in the `try` block
"""

# ╔═╡ eb267238-5583-11eb-1646-8d4d8e54c2ae


# ╔═╡ ed02d5f4-5583-11eb-1f80-c56a7c1f0b0d


# ╔═╡ e0861edc-56ef-4a16-b432-96fd2b45ce9e
md"""
### Missing

(see [“missing” documentation](https://docs.julialang.org/en/v1/manual/missing/))

The value `missing` of type `Missing` is used to represent missing value in a statistical sense

For example, if you loaded data from a panel, and gaps existed
"""

# ╔═╡ c8c63662-9c5e-4e6d-a963-a8d6baa6777f
md"""
A key feature of `missing` is that it propagates through other function calls - unlike `nothing`
"""

# ╔═╡ e0105942-7576-448c-b0ed-39dc77081707
md"""
The purpose of this is to ensure that failures do not silently fail and provide meaningless numerical results

This even applies for the comparison of values, which
"""

# ╔═╡ 789fa62e-6f06-47b6-b126-bbab2aa5e7bc
md"""
Where `ismissing` is the canonical way to test the value

In the case where you would like to calculate a value without the missing values, you can use `skipmissing`
"""

# ╔═╡ 2f215448-effb-4726-8d55-8f6c776f1b97
md"""
As `missing` is similar to R’s `NA` type, we will see more of `missing` when we cover `DataFrames`
"""

# ╔═╡ 294da076-6bcf-433d-b4ec-170913e61c3f
md"""
## Exercises


<a id='np-ex1'></a>
"""

# ╔═╡ e477174a-04bf-4ab4-b425-0d90313268f8
md"""
### Exercise 1

This exercise uses matrix operations that arise in certain problems,
including when dealing with linear stochastic difference equations

If you aren’t familiar with all the terminology don’t be concerned – you can
skim read the background discussion and focus purely on the matrix exercise

With that said, consider the stochastic difference equation


<a id='equation-ja-sde'></a>
$$
X_{t+1} = A X_t + b + \Sigma W_{t+1} \tag{1}
$$

Here

- $ X_t, b $ and $ X_{t+1} $ are $ n \times 1 $  
- $ A $ is $ n \times n $  
- $ \Sigma $ is $ n \times k $  
- $ W_t $ is $ k \times 1 $ and $ \{W_t\} $ is iid with zero mean and variance-covariance matrix equal to the identity matrix  


Let $ S_t $ denote the $ n \times n $ variance-covariance matrix of $ X_t $

Using the rules for computing variances in matrix expressions, it can be shown from [(1)](#equation-ja-sde) that $ \{S_t\} $ obeys


<a id='equation-ja-sde-v'></a>
$$
S_{t+1} = A S_t A' + \Sigma \Sigma' \tag{2}
$$

It can be shown that, provided all eigenvalues of $ A $ lie within the unit circle, the sequence $ \{S_t\} $ converges to a unique limit $ S $

This is the **unconditional variance** or **asymptotic variance** of the stochastic difference equation

As an exercise, try writing a simple function that solves for the limit $ S $ by iterating on [(2)](#equation-ja-sde-v) given $ A $ and $ \Sigma $

To test your solution, observe that the limit $ S $ is a solution to the matrix equation


<a id='equation-ja-dle'></a>
$$
S = A S A' + Q
\quad \text{where} \quad Q := \Sigma \Sigma' \tag{3}
$$

This kind of equation is known as a **discrete time Lyapunov equation**

The [QuantEcon package](http://quantecon.org/julia_index.html)
provides a function called `solve_discrete_lyapunov` that implements a fast
“doubling” algorithm to solve this equation

Test your iterative method against `solve_discrete_lyapunov` using matrices

$$
A =
\begin{bmatrix}
    0.8 & -0.2  \\
    -0.1 & 0.7
\end{bmatrix}
\qquad
\Sigma =
\begin{bmatrix}
    0.5 & 0.4 \\
    0.4 & 0.6
\end{bmatrix}
$$
"""

# ╔═╡ 02a9ca1b-ddeb-4a98-8031-5ad0afdc1643
md"""
### Exercise 2

Take a stochastic process for $ \{y_t\}_{t=0}^T $

$$
y_{t+1} = \gamma + \theta y_t + \sigma w_{t+1}
$$

where

- $ w_{t+1} $ is distributed `Normal(0,1)`  
- $ \gamma=1, \sigma=1, y_0 = 0 $  
- $ \theta \in \Theta \equiv \{0.8, 0.9, 0.98\} $  


Given these parameters

- Simulate a single $ y_t $ series for each $ \theta \in \Theta $
  for $ T = 150 $.  Feel free to experiment with different $ T $  
- Overlay plots of the rolling mean of the process for each $ \theta \in \Theta $,
  i.e. for each $ 1 \leq \tau \leq T $ plot  


$$
\frac{1}{\tau}\sum_{t=1}^{\tau}y_T
$$

- Simulate $ N=200 $ paths of the stochastic process above to the $ T $,
  for each $ \theta \in \Theta $, where we refer to an element of a particular
  simulation as $ y^n_t $  
- Overlay plots a histogram of the stationary distribution of the final
  $ y^n_T $ for each $ \theta \in \Theta $.  Hint: pass `alpha`
  to a plot to make it transparent (e.g. `histogram(vals, alpha = 0.5)`) or
  use `stephist(vals)` to show just the step function for the histogram  
- Numerically find the mean and variance of this as an ensemble average, i.e.
  $ \sum_{n=1}^N\frac{y^n_T}{N} $ and
  $ \sum_{n=1}^N\frac{(y_T^n)^2}{N} -\left(\sum_{n=1}^N\frac{y^n_T}{N}\right)^2 $  


Later, we will interpret some of these in [this lecture](https://lectures.quantecon.org/jl/lln_clt.html#)
"""

# ╔═╡ 38e4bb20-85f1-4c41-9725-7b4e1af672aa
md"""
### Exercise 3

Let the data generating process for a variable be

$$
y = a x_1 + b x_1^2 + c x_2 + d + \sigma w
$$

where $ y, x_1, x_2 $ are scalar observables, $ a,b,c,d $ are parameters to estimate, and $ w $ are iid normal with mean 0 and variance 1

First, let’s simulate data we can use to estimate the parameters

- Draw $ N=50 $ values for $ x_1, x_2 $ from iid normal distributions  


Then, simulate with different $ w $
* Draw a $ w $ vector for the `N` values and then `y` from this simulated data if the parameters were $ a = 0.1, b = 0.2 c = 0.5, d = 1.0, \sigma = 0.1 $
* Repeat that so you have `M = 20` different simulations of the `y` for the `N` values

Finally, calculate order least squares manually (i.e., put the observables
into matrices and vectors, and directly use the equations for
[OLS](https://en.wikipedia.org/wiki/Ordinary_least_squares) rather than a package)

- For each of the `M=20` simulations, calculate the OLS estimates for $ a, b, c, d, \sigma $  
- Plot a histogram of these estimates for each variable  
"""

# ╔═╡ b6d19689-f299-41c5-a2c3-b8cc3dafd50d
begin
	x = [1.0, missing, 2.0, missing, missing, 5.0]
	
	@show mean(x)
	@show mean(skipmissing(x))
	@show coalesce.(x, 0.0);  # replace missing with 0.0;
end

# ╔═╡ 0e2d9b8a-bb59-48cc-864f-19bec0afe339
a = [10; 20; 30; 40]

# ╔═╡ 4d7b7a51-46e8-4f32-bf97-f9220c98bfb5
a = randn(4)

# ╔═╡ 4a1c1a7c-509d-4614-8eba-107af00425b1
A = [1 2; 3 4]

# ╔═╡ 9d15b424-5b9e-440d-b35e-27344ac7ab00
b = sort(a, rev = true)  # returns new array, original not modified

# ╔═╡ b08155b2-5583-11eb-1b80-857ef0dd906a
a = [1.1,2.0,3.0]

# ╔═╡ e1b86d4a-0aa0-46a7-ad07-9b9ae2b9616e
b = reshape(a, 2, 2)

# ╔═╡ 28c6da77-d418-4f9c-9537-9bc865c28e3c
# 
a = 0.0:0.1:1.0  # 0.0, 0.1, 0.2, ... 1.0

# ╔═╡ 531fb1d2-a720-4af2-9c45-e21331fbc0e8
a = [10 20 30 40]  # two dimensional, shape is 1 x n

# ╔═╡ 0f7912a1-cd59-41f3-8cad-9e59932c571e
b = ones(2, 2)

# ╔═╡ dd2cdd3f-30a5-4682-8134-0eb9009faf6e
a = [10 20 30 40]'

# ╔═╡ dbd3202b-c660-4afe-909f-f8c957b65e62
a = [10, 20, 30, 40]

# ╔═╡ c9cf0e1f-aad0-408c-8704-4fec13b88dd5
b = [-100, 0, 100]

# ╔═╡ 6ec3e1dc-c000-4f78-a59d-8511969f26c5
A = [1 2; 2 3]

# ╔═╡ c5e9bf1e-6bef-4cb8-a145-200e93b69079
a = zeros(4)

# ╔═╡ deed19d5-d6b1-4eaf-aa4a-fb1c8bb7b032
a = [10, 20, 30, 40]

# ╔═╡ df98ed9e-db7d-48ba-86f3-a6fde4b763c5
a = ones(1, 2)

# ╔═╡ a979ff8a-70c9-4f94-a261-7420b314557c

function f(x)
	try
		sqrt(x)
	catch err                # enters if exception thrown
		sqrt(complex(x, 0))  # convert to complex number
	end
end
	

	


# ╔═╡ bfd07ec1-f29d-4a33-9294-2f13e3085669
begin
	x = missing
	
	@show x == missing
	@show x === missing  # an exception
	@show ismissing(x);
end

# ╔═╡ 89916127-aef3-4d79-9020-45b91de4a6d2
begin
	f(x) = x^2
	
	@show missing + 1.0
	@show missing * 2
	@show missing * "test"
	@show f(missing);      # even user-defined functions
	@show mean(x);
end

# ╔═╡ ed2edc14-0e6c-4010-95c1-7b3749ad02a9
b = [true false; false true]

# ╔═╡ 3918b4dd-c72d-42e4-82b5-1db0eb58db6a
x = [3.0, missing, 5.0, missing, missing]

# ╔═╡ 0927ba67-19fc-4a7c-bdb1-18466208b3a2
# You can create an empty array using the `Array()` constructor
# Need to add undef
x = Array{Float64}(undef, 2, 2)

# ╔═╡ 8910b4e6-9d51-4aa5-b699-2d4cb76d8a53
a = randn(2, 2)

# ╔═╡ eace2ae0-491b-4cb3-83eb-f360861a3ba5
a = [10, 20, 30]

# ╔═╡ 3a4a0983-e3d1-4449-8aa2-bde987a34cd9
b = ones(2, 2)

# ╔═╡ aa7fe199-2523-4506-8104-7f11d0cddde0
A = -ones(2, 2)

# ╔═╡ 1a7ff4b7-56cf-43c8-abd3-58ac9506d92f
a = [10 20; 30 40]  # 2 x 2

# ╔═╡ 31408ba7-7b98-4a73-ac65-e569d5c6927b
A = ones(2, 2)

# ╔═╡ 1d1652a8-d075-40c9-90d8-3303188f6773
x = [1.0, nothing]

# ╔═╡ 403c81aa-f2e0-43ff-a327-dd4d74d2762b
b = sort!(a, rev = true)  # returns *modified original* array

# ╔═╡ 4a6d4daa-96a5-4ee1-8a74-d64ab363b62d
a = [1 2 3 4]  # two dimensional

# ╔═╡ Cell order:
# ╠═9da58875-31ef-4559-bf1d-54c43a0c5a79
# ╟─db12d2e9-48e0-4c4a-9644-eabefda2338e
# ╟─941ec474-e349-43fb-8605-8aa9b8a459ec
# ╠═44b0626e-502f-4597-95d4-5f0a697cb54b
# ╟─f9fd1271-c0a0-4122-8de1-033ff30722c7
# ╠═5a92c652-ba81-4032-9da2-5b1133e88414
# ╠═b08155b2-5583-11eb-1b80-857ef0dd906a
# ╟─8b0eb882-e787-4be3-b786-475d4513e5f2
# ╠═6bd6d1e0-e161-46c9-b16e-b9d25e9f5f60
# ╟─4a22df91-ab96-4802-a27e-8ff93cce78a8
# ╠═74889209-6203-4b04-aa83-e6576f0abd0c
# ╠═71ef3e30-57ab-4765-86f2-56ced72edea3
# ╟─b250805b-bfde-444a-9413-d67545be5618
# ╟─fe3c213c-361b-4d7e-8409-41b0a0bf6aa9
# ╠═ba7d88fe-27aa-4137-b3d0-05a439546da9
# ╟─5e0e1163-a61d-42ec-ac58-9a6873bdb287
# ╠═b690c4e9-9837-4e44-b968-089530657da0
# ╠═a05c5668-8984-4216-a730-52d45c47f51e
# ╟─2ff3c0c8-9c2f-4b9d-aca4-e0eecf96b7b6
# ╟─4f432530-26de-47bd-a835-0d3eb25bfe85
# ╠═4347049b-8258-4368-9699-c52ad9760c71
# ╠═4f41d825-a0a2-48f5-805c-c5b47c41b0ea
# ╠═603aafd9-6f27-48d7-b422-96a5442161d7
# ╠═0927ba67-19fc-4a7c-bdb1-18466208b3a2
# ╟─826a2c3d-1372-414c-888c-08844e2d2561
# ╠═5963c2e5-1d14-4cac-8227-e5d8abaab22c
# ╠═ba70403a-3797-4244-b9e6-3bd0d3b77762
# ╟─46a6a796-da2e-4b13-bcfa-7e7f9878e61a
# ╠═cf1ead00-f208-47af-8d46-b90871d62954
# ╟─5eba9af6-79f9-4fdf-bc7a-b5f025d0a581
# ╠═02693f8e-9065-40c1-b500-5c9fe798e179
# ╟─c5979a79-03b2-474f-bce2-03701130b002
# ╠═fb9c2d57-25f2-40f8-9de2-863ca1efbd8a
# ╟─5f5776a3-9622-47b3-a236-84d32008bdad
# ╠═41a363e0-f792-42f7-8b5e-3fafc1f1d25d
# ╠═8e81e25c-424d-44c1-8349-f08ed79109ae
# ╟─b122df17-37d8-4b2f-afad-b31829eaefab
# ╠═deed19d5-d6b1-4eaf-aa4a-fb1c8bb7b032
# ╟─2a8c5ae3-b8fb-4412-9b62-15f9151640cf
# ╠═531fb1d2-a720-4af2-9c45-e21331fbc0e8
# ╠═75beef5e-0437-4d9d-90eb-de7573e722e3
# ╠═1a7ff4b7-56cf-43c8-abd3-58ac9506d92f
# ╟─ddb2af37-70e9-40fe-a569-4673db40eccd
# ╠═0e2d9b8a-bb59-48cc-864f-19bec0afe339
# ╠═033498e2-cea8-4f04-9baa-e00a46a77c5a
# ╟─b57b246e-e946-4d48-a62c-7294f9aea32d
# ╠═dd2cdd3f-30a5-4682-8134-0eb9009faf6e
# ╠═16e8ece4-324c-4dbc-a6e5-69f287f12272
# ╟─35bc68b2-2471-4e34-a4eb-5f0675fd3387
# ╠═2e5669fa-4cdf-4635-b300-4d956a394465
# ╠═beea5832-2ad4-46a3-a80d-1284e96bdf41
# ╟─4d8ead57-db0a-497e-be77-27b3e2cd4495
# ╠═c8215492-ae31-4954-90c0-6fe2edb01a29
# ╠═fa97d1c1-146d-4f93-b9c7-fd9de4b0e957
# ╠═4186d1f8-f1df-45e3-a6be-0dd294870670
# ╟─46675778-edf2-4278-bb34-2bf2da1aba88
# ╠═8910b4e6-9d51-4aa5-b699-2d4cb76d8a53
# ╠═ed2edc14-0e6c-4010-95c1-7b3749ad02a9
# ╠═0db1740f-8023-4db9-9c88-8fb7f3ceaf9c
# ╟─d70e0451-6be0-4dc6-8a77-eaa1a06d5f08
# ╠═c5e9bf1e-6bef-4cb8-a145-200e93b69079
# ╠═46637100-b299-4d73-8518-57d6449ac8c6
# ╠═588897de-f6ee-4f7a-a50f-e1403ccd241f
# ╟─76704aa7-968d-44e1-95bd-63b6ede7fe2f
# ╠═dc3475a5-3fe9-4399-9233-7edbfb47bdda
# ╟─0ed1ec30-2882-4a28-9a2d-9bb5a781b80c
# ╠═8073c593-fcbf-4cf8-b895-0f1d6223b25f
# ╟─f20373ed-3cbf-490c-8aff-6ec5667a2f20
# ╠═839fa9d6-f07f-4e76-a97e-c24a86811c9d
# ╟─18697bff-88ab-4295-b33e-a86e386a8e5c
# ╠═c5e54a82-1ad5-4b00-87f7-b372115e84c7
# ╟─09462935-5aa9-4f70-b670-8c306d094433
# ╠═71cc1049-6f38-4fd9-a77a-3a967a83e128
# ╟─d93baea1-38ab-424f-a5ee-2d30177f7c71
# ╠═cfefbb99-454e-4c87-984e-6526ebb09ef4
# ╟─6ee0e819-00ed-4bf6-9718-f93441eaab68
# ╠═f379c869-6f94-4c45-9e6c-d49b797db93d
# ╟─aa65bbd4-cc93-4aa1-aaee-e3f87d57f596
# ╠═52482783-2a81-40d9-a056-0bcd4a86bf95
# ╟─7bea4cb4-da56-4841-94ee-2d36c51c06d1
# ╠═98a9f682-14d4-47b9-be58-e5fee7275cfa
# ╠═717c24ac-939e-450c-a816-6ae08dd0cf57
# ╟─d9cb4a5a-b26b-4473-b9e3-5f1bdb7fea56
# ╠═fd2caba5-c543-492b-ab1e-90a691e08c3b
# ╟─0e5a392b-93ed-4688-a6a6-03c2023ddd1f
# ╟─0ca09099-a5b0-4490-b03f-d7ab1e6e1285
# ╠═d678cb9f-eda8-448b-bb0d-7243f409d4d9
# ╟─915ec462-f1ad-4e17-ba5d-e382d107d570
# ╠═e4336818-7eb8-4d6f-be1a-9dd3aea9b6df
# ╟─9bfad4b6-cdf4-4324-be66-b176d211ca28
# ╠═0bf8f290-f9bc-4ba5-bebf-ddbb3e34db77
# ╟─8dfd8d68-9e58-4c1a-a24b-1f145fd9c46e
# ╠═bb264090-c5e8-4f4a-9085-547d4fcf71d2
# ╟─92df24e5-7dfb-466b-b3f2-a028cdbd6b33
# ╠═0d9a11d8-63ad-4fde-8103-b464a4b5b23d
# ╟─5bf57b84-4d38-4b83-bc47-42d397b334b1
# ╠═ca5456a4-20f2-4f23-b1d3-ffa626cd6982
# ╟─ae567de9-2ac1-4675-9d64-02cdc53e73f2
# ╠═a2d98af2-90f2-4725-a4e3-b2f3d9a9d0dd
# ╟─7e2d765b-696a-4bd9-9e31-00094028065b
# ╠═60db9959-705c-40e1-9e7a-1d477d645c60
# ╟─65279ef8-4ee7-45b5-9789-8aaeac9f33db
# ╠═adf05e01-9831-4b09-8527-bdd631966ed1
# ╟─81b2de8d-07b7-440b-8614-c09021a0666f
# ╟─c7a26040-24f2-4c87-97b5-d9280735fc17
# ╠═cf1a068d-7332-4a66-9b65-ab3c67740c45
# ╟─dda39690-e65a-43a4-a609-f98156b325b0
# ╠═9d15b424-5b9e-440d-b35e-27344ac7ab00
# ╠═403c81aa-f2e0-43ff-a327-dd4d74d2762b
# ╠═75207a4f-fa1b-4482-a9d8-9bb2dd67b7b5
# ╠═0de1b7ed-cc42-40be-a27d-5ec8cdf6747b
# ╟─8acb096f-9566-42b9-a7db-3c17b4584768
# ╠═df98ed9e-db7d-48ba-86f3-a6fde4b763c5
# ╠═3a4a0983-e3d1-4449-8aa2-bde987a34cd9
# ╠═01246a47-677c-47b1-8212-b9b165e3e91c
# ╠═e990a843-e839-43c6-b6b1-7ffea485be5b
# ╟─7c7e1986-3ddc-47e4-8707-3812a42a9124
# ╠═6ec3e1dc-c000-4f78-a59d-8511969f26c5
# ╠═de2063fc-c339-41b0-a5b3-916822ae2e23
# ╠═e1ee5121-fc52-48a3-9d02-bc95ef42159e
# ╠═10a82b79-30e8-4bd7-961f-b5e34dc85cf5
# ╟─d0cbb6fc-f560-4a18-8d38-b46685a051f0
# ╠═ff8f7b98-2d35-4b14-a34d-28ac82a9e9b7
# ╟─428dadd1-1169-476a-bac8-62ac839c7131
# ╠═f9eeacef-cf8e-48d6-bff9-9578376a0f06
# ╟─72c026a0-6efe-4456-b73e-78c6d41d0adb
# ╠═0f7912a1-cd59-41f3-8cad-9e59932c571e
# ╠═57666329-a8a7-4297-818e-e62a5eaa9b17
# ╠═439d050f-1dc9-465f-a296-4614e4540776
# ╟─dd6e792a-7779-4907-af5b-4609b20e57df
# ╠═4e62792b-2d9d-4759-928c-c74f0de1559c
# ╠═114f341c-5da5-4e3d-9078-9331f8e72ac9
# ╟─34954704-6042-4d41-937d-a4f590842a52
# ╠═aa7fe199-2523-4506-8104-7f11d0cddde0
# ╠═94356303-7f9b-4835-af02-3e60e32e24e3
# ╟─1bdc51b1-bb4a-4f27-88ab-75d66908a471
# ╠═cf7859c5-895b-4067-a6ff-89e9c656da0c
# ╟─5a6a515b-2af4-436d-8668-df5d20e31041
# ╠═31408ba7-7b98-4a73-ac65-e569d5c6927b
# ╠═1061ecdc-506d-48aa-8362-74b50265359c
# ╟─d3e1568b-8152-4f8a-8fe0-2e27e5434d3c
# ╠═9a769806-ae4c-44a1-b6b4-6da7a9bb9895
# ╟─8384a0b3-8f0b-4a4e-b041-d5129f517998
# ╠═eace2ae0-491b-4cb3-83eb-f360861a3ba5
# ╠═c9cf0e1f-aad0-408c-8704-4fec13b88dd5
# ╠═dcb101c7-5612-49e4-9d67-be4fdee8c996
# ╠═ab90940a-cea0-4901-8b47-828bc88b6a3c
# ╟─ceca8038-4054-47dc-bceb-8867cd23b606
# ╠═2aadea95-693a-4fed-b218-4398a5560d50
# ╠═67ab25f9-ea3c-489e-88e2-d0dc4f7a3fe4
# ╟─c3fed55f-3397-421d-b361-950673e4a5c8
# ╠═4d7b7a51-46e8-4f32-bf97-f9220c98bfb5
# ╠═01e2193f-dd92-4bc1-b3cc-65d5cf6e2eab
# ╠═a46dde55-16f0-4d27-adeb-0b4af0e9da33
# ╟─c9d9ec45-5376-403d-9ede-f6ed27b8f874
# ╠═dbd3202b-c660-4afe-909f-f8c957b65e62
# ╠═e1b86d4a-0aa0-46a7-ad07-9b9ae2b9616e
# ╠═3d2fdafa-f484-40bd-a8bf-208c5a08701e
# ╟─9f911a3f-7dff-45af-928c-7371dce6a24c
# ╠═7c3d36f2-efce-46a4-8b93-71d43dd8c121
# ╠═d6e2493b-a250-4cb3-823a-6cc20962c36b
# ╠═cb9b754a-656a-45ac-8e72-e6bfa56e1af4
# ╟─7099a6e3-b1f3-4ab9-8d2c-ff8f305f611c
# ╠═4a6d4daa-96a5-4ee1-8a74-d64ab363b62d
# ╠═e0528017-3547-457b-b839-c9c76ad6973c
# ╟─d6182b58-8a38-413d-b193-1f4e3272af7d
# ╠═92f247b5-799e-4818-b28e-37730a8e7174
# ╟─7b3699c5-5dc7-4413-b7da-e30c8fa5167d
# ╠═bed57736-88e9-44da-8b67-6d510790866f
# ╟─e5834ec8-11c9-4e64-9bba-553283b8b3a7
# ╠═7dd06041-99fa-4a18-8db6-4164adeb34c4
# ╟─2854b3ea-7961-4fed-8190-0d92b105711a
# ╠═72933591-adfd-4cba-a67c-9d10d8b4fb7e
# ╠═d04d06bc-16c1-4482-b1cf-894dd3e1fe0e
# ╠═a6cb8a08-c69c-4d96-a2ef-5d9abce4068e
# ╠═7b89a0df-5ad9-4bfe-a587-8b9af9415732
# ╟─faa0cbe8-6b52-4365-8bb9-22a3a61a53b1
# ╠═4a1c1a7c-509d-4614-8eba-107af00425b1
# ╠═2aec79bf-e6a0-43fb-8500-3b6374a867b1
# ╠═2a75e320-7b10-4aba-aa62-9de17c3b760c
# ╠═e9f8b372-de3d-4dc4-9216-91ff279ed0ea
# ╠═07e16567-5b5d-4add-9bc5-428bcc35b5e4
# ╟─c1e4ec26-28b2-49ee-a5a4-d405d04c4896
# ╠═97e0ad1d-6ddc-476c-b4c9-73e3ff1e6f96
# ╠═28c6da77-d418-4f9c-9537-9bc865c28e3c
# ╟─8195212c-6c41-4e8e-92c0-da37e173c666
# ╠═04e6aa99-f617-4093-91c2-87e7c959fbe3
# ╟─ae198ca4-cb31-469a-ab48-03747c41f785
# ╠═238488fd-466b-4931-80dc-2663a622c2c6
# ╟─1e0851c9-840b-4636-8205-3357d9c81921
# ╟─1bf1a0da-cfac-4e7e-8e9a-36982953eaf0
# ╠═94ff9eeb-78d4-41c2-bd69-5d536734301b
# ╟─548f8f6c-7977-4cbb-941f-a56e000aa378
# ╠═7c435b8d-daf5-47ec-90ff-702ffdf1ea4d
# ╟─be942bce-68b4-4024-8c96-4c14e03f41c4
# ╠═14c7ba45-a711-447a-8622-bd7a48f01eb5
# ╟─e9220511-2ef1-4bc7-81dd-bf1c0bc24a95
# ╠═25cfe013-ceab-4312-bc1f-60e7c9f4824d
# ╟─e92806e3-3ee2-4f4f-ad46-d6e5a7c3d6a0
# ╠═7cf94b4a-bed3-4838-b0bf-8e65348133cb
# ╟─8fc78255-f690-487a-93cf-efad5a697f78
# ╠═51ac074e-3af8-404d-8251-d21d97d741a2
# ╟─e195e763-27c0-4e5d-93f6-985c830aa6ed
# ╟─1788e724-62d7-42e2-9bc8-0c3b02b3d4a0
# ╟─afc582c7-8afe-4ee6-afbc-afeb4ccb6ed4
# ╠═3f9abc19-7f0a-4c8d-a9b9-bda04e0f5e6e
# ╟─dde2249f-bf27-4680-84fe-548b87b195b9
# ╠═e019042f-5899-4c61-9445-163b105493c5
# ╟─c46b746e-8e09-4ef5-a80b-1f6e587fae84
# ╠═21f5ad82-c94f-4985-91ca-068341738ba8
# ╟─c0ffb97f-a6f7-4391-b27f-8ea7349532ce
# ╠═7bdeca04-fcc0-42ce-8d45-12572cd37358
# ╟─1d96bf36-00ae-485d-a338-520b458d0f4d
# ╠═1d1652a8-d075-40c9-90d8-3303188f6773
# ╟─44ca4731-cb01-4528-a201-214d4e9ca3b7
# ╠═ef94faf8-38d4-465f-baef-942cb1a71144
# ╟─ca15a5c5-b567-49dc-924b-8b1843b18747
# ╠═f6a9edbb-b7b1-4ba5-ab99-24247843f7ea
# ╟─1a226619-3655-4685-b96c-b58a1c7ae52a
# ╠═4349f64c-8892-4616-9b27-8205579c59ea
# ╟─d7a59a0f-8de8-42c4-afbe-043d92d5e334
# ╠═3ec97746-d1b2-45df-bf2d-44c49a5b1c6c
# ╟─bcf973b7-20ad-41ed-b9e2-a6c73c9e711e
# ╟─77f89197-4017-49d6-a329-06ba0dc75910
# ╠═8577bffe-96f0-4b7a-aab2-9755dc334545
# ╟─be0df836-c55a-4832-b0b2-2faa08ea3028
# ╠═8b0251c1-61ae-4fd8-a856-4fb9f925e2f9
# ╟─d14ca331-a60f-4524-8c1e-62f6f2358740
# ╠═a979ff8a-70c9-4f94-a261-7420b314557c
# ╠═eb267238-5583-11eb-1646-8d4d8e54c2ae
# ╠═ed02d5f4-5583-11eb-1f80-c56a7c1f0b0d
# ╟─e0861edc-56ef-4a16-b432-96fd2b45ce9e
# ╠═3918b4dd-c72d-42e4-82b5-1db0eb58db6a
# ╟─c8c63662-9c5e-4e6d-a963-a8d6baa6777f
# ╠═89916127-aef3-4d79-9020-45b91de4a6d2
# ╟─e0105942-7576-448c-b0ed-39dc77081707
# ╠═bfd07ec1-f29d-4a33-9294-2f13e3085669
# ╟─789fa62e-6f06-47b6-b126-bbab2aa5e7bc
# ╠═b6d19689-f299-41c5-a2c3-b8cc3dafd50d
# ╟─2f215448-effb-4726-8d55-8f6c776f1b97
# ╟─294da076-6bcf-433d-b4ec-170913e61c3f
# ╟─e477174a-04bf-4ab4-b425-0d90313268f8
# ╟─02a9ca1b-ddeb-4a98-8031-5ad0afdc1643
# ╟─38e4bb20-85f1-4c41-9725-7b4e1af672aa
