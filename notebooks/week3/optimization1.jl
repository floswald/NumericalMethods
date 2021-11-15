### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ b1a45f30-beec-4089-904c-488b86b56a9e
begin
	using Plots
	using LaTeXStrings
	using PlutoUI
end

# ╔═╡ d4bb171d-3c1c-463a-9360-c78bdfc83363
begin
	using Calculus
	# also can compute gradients for multidim functions
	Calculus.gradient(x->x[1]^2 * exp(3x[2]),ones(2)), Calculus.hessian( x->x[1]^2 * exp(3x[2]),ones(2))
end

# ╔═╡ 068dd98e-8507-4380-a4b2-f6fee80adaaa
begin
	using SymEngine
	x = symbols("x");
	f = x^2 + x/2 - sin(x)/x; diff(f, x)
end

# ╔═╡ 86440ba5-4b5f-440b-87e4-5446217dd073
using ForwardDiff    # one particular AD package in julia

# ╔═╡ 5c316980-d18d-4698-a841-e732f7632cec
html"<button onclick='present()'>present</button>"

# ╔═╡ 53ef0bfc-4239-11ec-0b4c-23f451fff4a6
md"""
# Optimization 1

* This lecture reminds you of some optimization theory.
* The focus here is to illustrate use cases with julia.
* We barely scratch the surface of optimization, and I refer you to Nocedal and Wright for a more thorough exposition in terms of theory.
* This 2-part lecture is heavily based on [Algorithms for Optimization](https://mitpress.mit.edu/books/algorithms-optimization) by Kochenderfer and Wheeler.

This is a 2 part lecture.

## Optimization I: Basics

1. Intro
2. Conditions for Optima
3. Derivatives and Gradients
4. Numerical Differentiation
5. Optim.jl


## Optimization II: Algorithms

1. Bracketing
2. Local Descent
3. First/Second Order and Direct Methods
4. Constraints

## The Optimization Process

```
1. Problem Specification
2. Initial Design
3. Optimization Proceedure:
    a) Evaluate Performance
    b) Good?
        i. yes: final design
        ii. no: 
            * Change design
            * go back to a)
```            

We want to automate step 3.

## Optimization Algorithms

* All of the algorithms we are going to see employ some kind of *iterative* proceedure. 
* They try to improve the value of the objective function over successive steps.
* The way the algorithm goes about generating the next step is what distinguishes algorithms from one another.
	* Some algos only use the objective function
	* Some use both objective and gradients
	* Some add the Hessian
	* and many variants more

## Desirable Features of any Algorithm

* Robustness: We want good performance on a wide variety of problems in their class, and starting from *all* reasonable starting points.
* Efficiency: They should be fast and not use an excessive amount of memory.
* Accuracy: They should identify the solution with high precision.
"""

# ╔═╡ 9b3eee98-e481-4fb6-98c2-6ac408dcfe54
md"""

## Optimisation Basics

* Recall our generic definition of an optimization problem:

$$\min_{x\in\mathbb{R}^n} f(x)  \text{ s.t. } x \in \mathcal{X}$$

  symbol |  meaning
  --- | ---- 
 $x$ | *choice variable* or a *design point*
 $\mathcal{X}$ | feasible set
$f$  | objective function
 $x^*$ | *solution* or a *minimizer*

$x^*$ is *solution* or a *minimizer* to this problem if $x^*$ is *feasible* and $x^*$ minimizes $f$.

Maximization is just minimizing $(-1)f$:

$$\min_{x\in\mathbb{R}^n} f(x)  \text{ s.t. } x \in \mathcal{X} \equiv \max_{x\in\mathbb{R}^n} -f(x)  \text{ s.t. } x \in \mathcal{X}$$

"""

# ╔═╡ 6163277d-70d3-4a73-89df-65c329c2b818


# ╔═╡ 843fef36-611c-4411-b31b-8a11e128881b
@bind B Slider(0:0.1:10,default = 3.0)

# ╔═╡ 3ba9cc34-0dcd-4c2e-b428-242e456bd436
let
	npoints = 100
	a,b = (0,10)
	x = range(a,b,length = npoints)
	f₀(x) = x .* sin.(x)
	plot(x, f₀.(x), leg=false,color=:black,lw = 2,title = "Finding the Max is Easy! Right?")
	xtest = x[x .<= B]
	fmax,ix = findmax(f₀.(xtest))	
	scatter!([xtest[ix]], [fmax], color = :red, ms = 5)
	vline!([B],lw = 3)
end

# ╔═╡ 601e4aa9-e380-41a1-96a2-7089603889c3
md"""
## Constraints

* We often have constraints on problems in economics.

$$\max_{x_1,x_2} u(x_1,x_2)  \text{ s.t. } p_1 x_1 + p_2 x_2 \leq y$$

* Constraints define the feasible set $\mathcal{X}$.
* It's better to write *weak inequalities* (i.e. $\leq$) rather than strict ones ($<$). 
"""

# ╔═╡ fefd6403-f46c-4eb1-b754-a85bcb75914c
md"""
## Example
$$\min_{x_1,x_2} -\exp(-(x_1 x_2 - 3/2)^2 - (x_2-3/2)^2) \text{ s.t. } x_2 \leq \sqrt{x_1}$$



"""

# ╔═╡ 2ac3348d-196a-4507-b2f7-c575e42d7e7b
let
	x=0:0.01:3.5
	f0(x1,x2) = -exp.(-(x1.*x2 - 3/2).^2 - (x2-3/2).^2)
	c(z) = sqrt(z)

	p1 = surface(x,x,(x,y)->f0(x,y),xlab = L"x_1", ylab = L"x_2")
	p2 = contour(x,x,(x,y)->f0(x,y),lw=1.5,levels=[collect(0:-0.1:-0.85)...,-0.887,-0.95,-1],xlab = L"x_1", ylab = L"x_2")
	plot!(p2,c,0.01,3.5,label="",lw=2,color=:black,fill=(0,0.5,:blue))
	scatter!(p2,[1.358],[1.165],markersize=5,markercolor=:red,label="Constr. Optimum")
	plot(p1,p2,size=(900,300))
end

# ╔═╡ 09582278-6fed-4cac-9aaa-45cf0ac9fb6c
md"""
## Conditions for Local Minima

We can define *first and second order necessary conditions*, FONC and SONC. This definition is to point out that those conditions are not sufficient for optimality (only necessary).

### Univariate $f$

1. **FONC:** $f'(x^*) =0$
2. **SONC** $f''(x^*) \geq 0$ (and $f''(x^*) \leq 0$ for local maxima)
2. (**SOSC** $f''(x^*) > 0$ (and $f''(x^*) < 0$ for local maxima))

"""

# ╔═╡ 58cb6931-91a5-4325-8be4-6675f7e142ed
md"""
### Multivariate $f$

1. **FONC:** $\nabla f(x^*) =0$
2. **SONC** $\nabla^2f(x^*)$ is positive semidefinite (negative semidefinite for local maxima)
2. (**SOSC** $\nabla^2f(x^*)$ is positive definite (negative definite for local maxima))
"""

# ╔═╡ 3af8c139-2e6c-4830-9b67-96f78356f521
md"""
## Example Time: Rosenbrock's Banana Function

A well-known test function for numerical optimization algorithms is the Rosenbrock banana function developed by Rosenbrock in 1960. it is defined by 

$$f(\mathbf{x}) = (1-x_1)^2  + 5(x_2-x_1^2)^2$$
    
"""

# ╔═╡ dd9bfbb1-aecf-458f-9a05-a93ff78fd741
md"""
## How to write a julia function?

* We talked briefly about this - so let's try out the various forms:
* (and don't forget to [look at the manual](https://docs.julialang.org/en/v1/manual/functions/) as always!)
"""

# ╔═╡ 34b7e91e-d67f-4554-b985-b9100adda733
begin
	# long form taking a vector x
	function rosen₁(x)
		(1-x[1])^2 + 5*(x[2] - x[1]^2)^2
	end
	# short form taking a vector x
	rosen₂(x) = (1-x[1])^2 + 5*(x[2] - x[1]^2)^2
end

# ╔═╡ 4d2a5726-2704-4b63-b334-df5175278b18
begin
	using Optim
	result = optimize(rosen₁, zeros(2), NelderMead())
end

# ╔═╡ 2dbb5b13-790a-4ab7-95b1-b833c4cb027a
rosen₁([1.1,0.4]) == rosen₂([1.1,0.4])

# ╔═╡ f51233c4-ec66-4517-9109-5309601d1d87
md"""
* but the stuff with `x[1]` and `x[2]` is ugly to read
* no? 🤷🏿‍♂️ well I'd like to read this instead

$$f(x,y) = (1-x)^2  + 5(y-x^2)^2$$

* fear not. we can do better here.
"""

# ╔═╡ 3729833f-80d4-4948-8d81-750008c8f16d
begin
	# long form taking an x and a y
	function rosen₃(x,y)
		(1-x)^2 + 5*(y - x^2)^2
	end
	# short form taking a vector x
	rosen₄(x,y) = (1-x[1])^2 + 5*(x[2] - x[1]^2)^2
end

# ╔═╡ 7172d082-e6d2-419b-8bb6-75e30f1b4dfe
md"""
ok fine, but it's often useful to keep data in a vector. Can we have the readibility of the `x,y` formulation, with the vector input?

➡️ We can! here's a cool feature called *argument destructuring*:
"""

# ╔═╡ e7841458-f641-48cf-8667-1e5b38cbd9f6
rosen₅((x,y)) = (1-x)^2 + 5*(y - x^2)^2  # the argument is a `tuple`, i.e. a single object!

# ╔═╡ abbc5a52-a02c-4f5b-bd1e-af5596455762
@which rosen₅([1.0, 1.3])

# ╔═╡ 95e688e2-9607-41a2-9098-626590bcf435
rosen₅( [1.0, 1.3] )  # assigns x = 1.0 , y = 1.3 inside the function

# ╔═╡ 8279fd8a-e447-49b6-b729-6e7b8883f5e4
md"""
Ok enough of that. Let's get a visual of the Rosenbrock function finally!
"""

# ╔═╡ ed2ee298-ac4f-4ae3-a9e3-300040a706a8
md"""
### Keyword Arguments

In fact, the numbers `1` and `5` in 

$$f(x,y) = (1-x)^2  + 5(y-x^2)^2$$

are just *parameters*, i.e. the function definition can be changed by varying those. Let's get a version of `rosen()` which allows this, then let's investigate the plot again:

"""

# ╔═╡ 0bbaa5a8-8082-4697-ae98-92b2ae3769af
rosenkw(x,y ; a = 1, b = 5) = (a - x)^2 + b*(y - x^2)^2  # notice the ; 

# ╔═╡ dd0c1982-38f4-4752-916f-c05da365bade
md"""
* alright, not bad. but how can I change the a and b values now?
* One solution is to pass an *anonymous function* which will *enclose* the values for `a` and `b` (it is hence called a `closure`):
"""

# ╔═╡ 202dc3b6-ddcb-463d-b8f2-a285a2ecb112
md"""
This wouldn't be a proper pluto session if we wouldn't hook those values up to a slider, would it? Let's do it!
"""

# ╔═╡ 91fd09a1-8b3a-4772-b6a5-7b149d91eb4d
md"""
	a = $(@bind a Slider(0.05:0.1:10.5, default=1, show_value=true))
	"""

# ╔═╡ b49ca3b1-0d1b-4edb-8064-e8cd8d4db727
md"""
	b = $(@bind b Slider(0.1:0.5:20, default=1, show_value=true))
	"""

# ╔═╡ 86f0e396-f81b-45be-94a7-90e40a8ba251
md"""
## Finding Optima

Ok, tons of fun. Now let's see where the optimum of this function is located. In this instance, *optimum* means the *lowest value* on the $z$ axis. Let's project the 3D graph down into 2D via a contour plot to see this better:
"""

# ╔═╡ 9806ec5e-a884-41a1-980a-579915a33b8e
md"""
* The optimum is at point $(1,1)$ (I know it.)
* it's not great to see the contour lines on this plot though, so let's try a bit harder.
* Let's choose a different color scheme and also let's bit a bit smarter at which levels we want to measure the function:
"""

# ╔═╡ 8300dbb5-0eb6-4f84-80c6-24c4443b1f29
md"""
## Derivatives and Gradients

* 😱
* You all know this, so no panic.
* The derivative of a univariate function $f$ at point $x$, $f'(x)$ gives the rate with which $f$ changes at point $x$.
* Think of a tangent line to a curve, to economists known as the omnipresent and omnipotent expression : `THE SLOPE`. Easy. Peanuts. 🥜
* Here is the definition of $f'$
$$f'(x) \equiv \lim_{h\to0}\frac{f(x+h)-f(x)}{h}$$

* Like, if I gave you function like $u(c) = \frac{c^{1-\sigma}}{1-\sigma}$ , I bet you guys could shoot back in your sleep that $u'(c) = \frac{\partial u(c)}{\partial c} = ?$
* Of course you know all the differentiation rules, so no problem. But a computer?
* In fact, there are several ways. Let's illustrate the easiest one first, called *finite differencing*:
"""

# ╔═╡ 3a40b68f-83cf-4db9-b301-492e5cedcd13
u(c; σ = 2) = (c^(1-σ)) / (1-σ)

# ╔═╡ b901c4aa-38f8-476a-8c9e-7eb523f59438
eps()

# ╔═╡ d4af5141-422b-4941-8dc7-f2b4b09029c0
md"""
ϵ = $(@bind ϵ Slider(-6:-1, show_value = true, default = -1))
"""

# ╔═╡ 3fd2f03a-fc52-4009-b284-0def00be601f
h = 10.0^ϵ

# ╔═╡ 27d955de-8d97-43e4-9176-aad5456eb797
let
	c = 1.5
	∂u∂c = (u(c + h) - u(c)) / h  # definition from above!
	Dict(:finite_diff => ∂u∂c, :truth_enrico => c^-2)
end

# ╔═╡ 645ef857-aff9-4dee-bfd6-72fe9d542375
md"""
## Multiple Dimensions: 

* Let's add more notation to have more than 1 dimensional functions.

### $f$ that takes a vector and outputs a number

* Unless otherwise noted, we have $x \in \mathbb{R}^n$ as an $n$ element vector.
* The **gradient** of a function $f : \mathbb{R}^n \mapsto \mathbb{R}$ is denoted $\nabla f:\mathbb{R}^n \mapsto \mathbb{R}^n$ and it returns a vector
	
$$\nabla f(x) = \left(\frac{\partial f}{\partial x_1}(x),\frac{\partial f}{\partial x_2}(x),\dots,\frac{\partial f}{\partial x_n}(x) \right)$$

* So that's just taking the partial derivative wrt to *each* component in $x$.

### $f$ that takes a vector and outputs *another vector* 🤪

* In this case we talk of the **Jacobian** matrix.
* You can easily see that if $f$ is s.t. it maps $n$ numbers (in) to $m$ numbers (out), now *taking the derivative* means keeping track of how all those numbers change as we change each of the $n$ input components in $x$.
* One particularly relevant Jacobian in optimization is the so-called **Hessian** matrix. 
* You can think of the hessian either as a function $H_f :\mathbb{R}^n \mapsto \mathbb{R}^{n\times n}$ and returns an $(n,n)$ matrix, where the elements are

$$H_f(x) = \left( \begin{array}{cccc} 
\frac{\partial^2 f}{\partial x_1 \partial x_1}(x)  &  \frac{\partial^2 f}{\partial x_2 \partial x_1}(x) & \dots & \frac{\partial^2 f}{\partial x_n \partial x_1}(x) \\
\frac{\partial^2 f}{\partial x_1 \partial x_2}(x)  &  \frac{\partial^2 f}{\partial x_2 \partial x_2}(x) & \dots & \frac{\partial^2 f}{\partial x_n \partial x_2}(x) \\
\vdots & \vdots & \dots & \vdots \\
\frac{\partial^2 f}{\partial x_1 \partial x_n}(x)  &  \frac{\partial^2 f}{\partial x_2 \partial x_n}(x) & \dots & \frac{\partial^2 f}{\partial x_n \partial x_n}(x) 
\end{array}\right)$$

* or you just imagine the gradien from above, and then differentiate each element *again* wrt to all components of $x$.
    
"""

# ╔═╡ 06ca10a8-c922-4252-91d2-e025ab306f02
md"""

## Time for a Proof! 😨

* We mentioned above the FOC and SOC conditions. 
* We should be able to *prove* that the point (1,1) is an optimum, right?
* Let's do it! Everybody derive the gradient *and* the hessian of the rosenbrock function $$f(x,y) = (1-x)^2  + 5(y-x^2)^2$$ to show that $(1,1)$ is a candidate optimum! As a homework! 😄
"""

# ╔═╡ ab589e93-a4ca-45be-882c-bc3da47e4d1c
md"""
### Calculus.jl package

* Meanwhile, here is a neat package to help out with finite differencing:
"""

# ╔═╡ b600aafb-7d23-417a-a8c9-597d95182469
md"""
## Approaches to Differentiation

1. We have seen *numerical Differentiation* or *finite differencing*. We have seen the issues with choosing the right step size. Also we need to evaluate the function many times, which is costly.
1. Symbolical Differentiation: We can teach the computer the rules, declare *symbols*, then then manipulate those expressions. We'll do that next.
1. Finally, there is **Automatic Differentiation (AD)**. That's the 💣 future! More later.
"""

# ╔═╡ bf8dfa21-29e4-4d6e-a876-ba1a6ca313b1
md"""
## Symbolic Differentiation on a Computer

* If you can write down an analytic form of $f$, there are ways to *symbolically* differentiate it on a computer.
* This is as if you would do the derivation on paper.
* Mathematica, python, and julia all have packages for that.
"""

# ╔═╡ 4b3f4b1b-1b22-4e2e-be5b-d44d74d8da0e
md"""
## Automatic Differentiation (AD)

* Breaks down the actual `code` that defines a function and performs elementary differentiation rules, after disecting expressions via the chain rule:
$$\frac{d}{dx}f(g(x)) = \frac{df}{dg}\frac{dg}{dx}$$
* This produces **analytic** derivatives, i.e. there is **no** approximation error.
* Very accurate, very fast.
* The idea is to be able to *unpick* **expressions** in your code.
* **Machine Learning** depends very strongly on this technology.
* Let's look at an example


"""

# ╔═╡ 3e480576-ed7d-4f2d-bcd1-d7d1cbbeccf9
let
	c = 1.5
	∂u∂c = (u(c + h) - u(c)) / h  # definition from above!
	(∂u∂c, c^-2, ForwardDiff.derivative(u,c))
end

# ╔═╡ bc52bf0c-6cd1-488d-a9c1-7a91a582dda9
md"""
* I find this mind blowing 🤯

### AD Example

Consider the function $f(x,y) = \ln(xy + \max(x,2))$. Let's get the partial derivative wrt $x$:

$$\begin{aligned} \frac{\partial f}{\partial x} &= \frac{1}{xy + \max(x,2)} \frac{\partial}{\partial x}(xy + \max(x,2)) \\
         &= \frac{1}{xy + \max(x,2)} \left[\frac{\partial(xy)}{\partial x} + \frac{\partial\max(x,2)}{\partial x} \right]\\
         &= \frac{1}{xy + \max(x,2)} \left[\left(y\frac{\partial(x)}{\partial x} + x\frac{\partial(y)}{\partial x}\right) + \left(\mathbf{1}(2>x)\frac{\partial 2}{\partial x} + \mathbf{1}(2<x)\frac{\partial x}{\partial x} \right)\right] \\
          &= \frac{1}{xy + \max(x,2)} \left[y + \mathbf{1}(2<x)\right]
\end{aligned}$$
 
 where the indicator function $\mathbf{1}(r)=1$ if $r$ evaluates to *true*, 0 otherwise.
"""

# ╔═╡ 73fea39a-3ba6-4a37-9014-261a95acc084
md"""
* What we just did here, i.e. unpacking the mathematical operation $\frac{\partial f}{\partial x}$ can be achieved by a computer using a *computational graph*. 
* Automatic Differentiation traverses the computational graph of an *expression* either forwards (in *forward accumulation* mode), or backwards (in *reverse accumulation* mode).
"""

# ╔═╡ a5e5f5bc-cc5e-4f70-91ac-43fb21f2cada
md"""
This can be illustrated in a **call graph** as below:
* circles denote operators
* arrows are input/output
* We want to unpack the expression by successively applying the chain rule:
    $$\frac{d f}{d x} = \frac{d f}{d c_4}\frac{d c_4}{d x} = \frac{d f}{d c_4}\left(\frac{d c_4}{d c_3}\frac{d c_3}{d x}\right) = \frac{d f}{d c_4}\left(\frac{d c_4}{d c_3}\left(\frac{d c_3}{d c_2}\frac{d c_2}{d x}\right)\right) = \dots$$


"""

# ╔═╡ 7ee3eb27-c1e1-477e-bdd0-894e4317c559
md"""
* Here is our operation $f(x,y) = \ln(xy + \max(x,2))$ described as a call graph: (will only show if you start julia in folder `week3` of the course website repo)
"""

# ╔═╡ 24266569-cd10-4765-95fd-61b06027dd0e
PlutoUI.LocalResource("./optimization/callgraph.png")

# ╔═╡ f8e89e44-d12c-43c3-b1ec-01c68f33c3b4
md"""
### Accumulating *forwards* along the call graph

* Let's illustrate how AD in forward mode works for $x=3,y=2$ and the example at hand. Remember that
    $$f(x,y) = \ln(xy + \max(x,2))$$
    and, hence 
    $$f(3,2) = \ln( 6 + 3 ) = \ln 9 \text{  and  }\frac{\partial f}{\partial x} = \frac{1}{6 + 3}(2 + 1) = \frac{1}{3}$$
* We start at the left side of this graph with the inputs. 
* The key is for each quantity to compute both the value **and** it's partial derivative wrt $x$ in this case.



"""

# ╔═╡ d9d7d94d-e457-4354-a1a3-4a230c9ddc29
PlutoUI.LocalResource("./optimization/callgraph1.png")

# ╔═╡ b0a8a72c-3eb1-431d-9b30-17115e60025a
PlutoUI.LocalResource("./optimization/callgraph2.png")

# ╔═╡ 443ec353-c574-4950-ad67-483791d8e934
PlutoUI.LocalResource("./optimization/callgraph3.png")

# ╔═╡ a7a07e38-6900-4fd1-8a87-0e16d92a5256
PlutoUI.LocalResource("./optimization/callgraph4.png")

# ╔═╡ 9315c9b1-87fa-4e91-a78d-c24a3007139b
PlutoUI.LocalResource("./optimization/callgraph5.png")

# ╔═╡ c6464aec-bdf5-49b7-a5d2-45c2f6471bc7
md"""
* Reverse mode works very similarly.
* So, we saw that AD yields both a function value ($c_4$) as well as a derivative ($\dot{c_4}$)
* They have the correct values.
* This procedure required a *single* pass forward over the computational graph.
"""

# ╔═╡ 347a3819-9300-49f5-97b4-d1847c5ee98c
md"""
* Notice that the **exact same amount of computation** needs to be performed by any program trying to evaluate merely the *function value* $f(3,2)$:
    1. multiply 2 numbers
    2. max of 2 numbers
    3. add 2 numbers
    4. natural logarithm of a number

QUESTION: **WHY HAVE WE NOT BEEN DOING THIS FOR EVER?!**
ANSWER: **Because it was tedious.**
"""

# ╔═╡ 6802424d-6072-4692-add2-d34abb3ce6b7
md"""
### Implementing AD

* What do you need to implement AD?

1. We need what is called *dual numbers*. This is similar to complex numbers, in that each number has 2 components: a standard *value*, and a *derivative*
    * In other words, if $x$ is a dual number, $x = a + b\epsilon$ with $a,b \in \mathbb{R}$.
    * For our example, we need to know how to do *addition*, *multiplication*, *log* and *max* for such a number type:
    $$\begin{aligned}
    (a+b\epsilon) + (c+d\epsilon) &= (a+c) + (b+d\epsilon) \\
    (a+b\epsilon) \times (c+d\epsilon) &= (ac) + (ad+bd\epsilon)
    \end{aligned}$$
2. You need a programming language where *analyzing expressions* is not too difficult to do. you need a language that can do *introspection*.
"""

# ╔═╡ 31b1bad8-5a3d-4d9e-93c8-45e854cf88f8
md"""
### Implementing Dual Numbers in Julia

This is what it takes to define a `Dual` number type in julia:

```julia
struct Dual 
    v
    ∂
end

Base.:+(a::Dual, b::Dual) = Dual(a.v + b.v, a.∂ + b.∂) 
Base.:*(a::Dual, b::Dual) = Dual(a.v * b.v, a.v*b.∂ + b.v*a.∂) 
Base.log(a::Dual) = Dual(log(a.v), a.∂/a.v)
function Base.max(a::Dual, b::Dual)
    v = max(a.v, b.v)
    ∂ = a.v > b.v ? a.∂ : a.v < b.v ? b.∂ : NaN 
    return Dual(v, ∂)
end
function Base.max(a::Dual, b::Int) 
    v = max(a.v, b)
    ∂ = a.v > b ? a.∂ : a.v < b ? 1 : NaN
    return Dual(v, ∂) 
end
```
"""

# ╔═╡ d9238a26-e792-44fc-be3d-7d8ec7e0117d
let
	x = ForwardDiff.Dual(3,1);
	y = ForwardDiff.Dual(2,0);
	log(x*y + max(x,2))
end

# ╔═╡ eb2d7221-25b4-4836-b818-3ed944570040
md"""
... or just:
"""

# ╔═╡ 66f0d9bb-7d04-4e82-b9dd-55510971691b
ForwardDiff.derivative((x) -> log(x*2 + max(x,2)), 3) # y = 2

# ╔═╡ 4c60c221-545c-4050-bfea-211048a36bce
md"""
Of course this also works for more than one dimensional functions:
"""

# ╔═╡ 2d1f128c-bcfa-4017-9690-01f3f75c3efa
ForwardDiff.gradient(rosen₁, [1.0,1.0])  # notice: EXACTLY zero.

# ╔═╡ b4ade3a3-668e-495b-9b7b-ad45fdf2655b
ForwardDiff.hessian(rosen₁, [1.0,1.0])  # again, no rounding error.

# ╔═╡ 9431caba-619d-4104-a267-914a9bcc78ef
md"""
## Introducing [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl)

* Multipurpose unconstrained optimization package 
  * provides 8 different algorithms with/without derivatives
  * univariate optimization without derivatives
  * It comes with the workhorse function `optimize`
"""

# ╔═╡ 58f32a65-1ef8-4d9a-a874-00f7df563b3c
md"""
let's opitmize the rosenbrock functoin *without* any gradient and hessian:
"""

# ╔═╡ 9f238c4a-c557-4c57-a24c-6d221d592a18
md"""
now with both hessian and gradient! we choose another algorithm:
"""

# ╔═╡ 278cc047-83ee-49b1-a0e3-d2d779c1bc17
md"""
function library
"""

# ╔═╡ 5f3ad56f-5f8f-4b51-b45c-46c37eaeced4
begin
		function g!(G, x)
           G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
           G[2] = 200.0 * (x[2] - x[1]^2)
	end
	function h!(H, x)
           H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
           H[1, 2] = -400.0 * x[1]
           H[2, 1] = -400.0 * x[1]
           H[2, 2] = 200.0
	end
end

# ╔═╡ f061e908-0687-4375-84e1-386a0dd48b39
o = optimize(rosen₁, g!, h!, zeros(2), Newton())

# ╔═╡ eb65a331-c977-4b0f-8add-873bd89095f4
Optim.minimizer(o)

# ╔═╡ d146a1e2-8067-4e25-b0cd-2a041162acb9
function minmax()
	v=collect(range(-2,stop = 2, length = 30))  # values
	mini = [x^2 + y^2 for x in v, y in v]
	maxi = -mini   # max is just negative min
	saddle = [x^2 + y^3 for x in v, y in v]
	Dict(:x => v,:min => mini, :max => maxi, :saddle => saddle)
end

# ╔═╡ 3722538e-76e9-4bab-bfa9-57eff72802b7
function mmplotter(s::Symbol;kws...)
	d = minmax()
surface(d[:x],d[:x],d[s],title="$s",fillalpha=0.8,leg=false,fillcolor=:heat; kws...)
end

# ╔═╡ b059cb44-349a-48b5-a96e-62c4835fde10
mmplotter(:max)

# ╔═╡ 5b925811-6255-4e2e-b691-40869d65d6df
mmplotter(:min)

# ╔═╡ a88b6949-4b4a-4f5a-a9a2-c6978cd0f758
mmplotter(:saddle,camera = (30,50))

# ╔═╡ f368672a-5c78-4d2a-aea9-f2a2c1ee0a54
info(text) = Markdown.MD(Markdown.Admonition("info", "Info", [text]));

# ╔═╡ 63703f51-bf0a-42c1-b981-3191d88b4901
warning(text) = Markdown.MD(Markdown.Admonition("warning", "Warning", [text]));

# ╔═╡ fcc24d08-bb9a-482f-987e-e64184c8d6f2
warning(md"Keep in mind that there may be other (better!) solutions outside of your interval of attention.")

# ╔═╡ d4c22f7b-31f5-4f41-8731-2f6189d231b4
function rosendata(f::Function)
	x = y = range(-2,stop = 2, length = 100)  # x and y axis
	rosenvals = [f(ix,iy) for ix in x, iy in y]  # f evaluations
	(x,y,rosenvals)
end

# ╔═╡ 76a613f2-482f-4a4d-8236-debee05bef1b
function rosenplotter(f::Function)
	x,y,vals = rosendata(f)  # get the data
	# plotting
	surface(x,y,vals, fillcolor = :thermal,colorbar=false, alpha = 0.9,xlab = "x",ylab = "y", zlab = "z")
end

# ╔═╡ 3cf9be4d-fa76-4264-b9b6-ff66bcf5db0e
rosenplotter(rosen₃)

# ╔═╡ dc21cc4b-aedd-42d7-b2a8-f36dfecee6f4
rosenplotter(rosenkw)

# ╔═╡ 7fcebc5a-a8c7-47d8-90b0-7ee8cd579585
rosenplotter( (x,y) -> rosenkw(x,y, a=1.5, b=2 ) ) # notice the `,` when calling

# ╔═╡ ba891e20-db23-4b03-9495-19c19df940d3
rosenplotter( (x,y) -> rosenkw(x,y, a=a, b=b ))

# ╔═╡ 12629919-26d3-4434-9c23-9778364fe71a
let
	x,y,z = rosendata(rosenkw)  # default a,b
	contour(x,y,z, fill = false, color = :deep,levels=[collect(0:0.1:175)...])
	scatter!([1.0],[1.0], m=:c, c=:red, label = "(1,1)")
end

# ╔═╡ b1c207b7-9d70-453c-b554-1c91f59ada0a
let
	x,y,z = rosendata(rosenkw)  # default a,b
	loglevels = exp.(range(log(0.05), stop = log(175.0), length = 100))
	contour(x,y,z, fill = false, color = :viridis,levels=loglevels)
	scatter!([1.0],[1.0], m=:c, c=:red, label = "(1,1)")
end

# ╔═╡ 33e3b11c-b1b4-4c64-b742-734ebd06926e
danger(text) = Markdown.MD(Markdown.Admonition("danger", "Danger", [text]));

# ╔═╡ ca7d694b-182a-443d-b47d-1bfe4ed8039f
danger(md"""
You should **not** normally attempt to write a numerical optimizer for yourself. Entire generations of Applied Mathematicians and other numerical pro's have worked on those topics before you, so you should use their work:

    1. Any optimizer you could come up with is probably going to perform below par, and be highly likely to contain mistakes.
    2. Don't reinvent the wheel.
That said, it's very important that we understand some basics about the main algorithms, because your task is **to choose from the wide array of available ones**.""")

# ╔═╡ 2e3243dc-f489-4117-82f8-7d05f5188429
bigbreak = html"<br><br><br><br><br>"

# ╔═╡ 5e09215e-1f9b-47a6-baf8-46f1f0dc1a20
bigbreak

# ╔═╡ 5a5bb3c5-f8da-4f7b-9b44-b54025d7e71c
midbreak = html"<br><br>"

# ╔═╡ 173b83be-dec2-487b-96ce-12cb5fba8be0
midbreak

# ╔═╡ c0edc9ae-ff2a-4224-820a-1a8844f41291
midbreak

# ╔═╡ 0b77e9c2-f360-498c-8a0a-157693866902
midbreak

# ╔═╡ dac4173c-9d3b-4573-b1ba-13c6b7cc5f30
midbreak

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Calculus = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[compat]
Calculus = "~0.5.1"
ForwardDiff = "~0.10.23"
LaTeXStrings = "~1.3.0"
Optim = "~1.5.0"
Plots = "~1.23.5"
PlutoUI = "~0.7.19"
SymEngine = "~0.8.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0bc60e3006ad95b4bb7497698dd7c6d649b9bc06"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "e527b258413e0c6d4f66ade574744c94edef81f8"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.40"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f885e7e7c124f8c92650d61b9477b9ac2ee607dd"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.1"

[[ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "9a1d594397670492219635b35a3d830b04730d62"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.1"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "3287dacf67c3652d3fed09f4c12c187ae4dbb89a"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.4.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "8b3c09b56acaf3c0e581c66638b85c8650ee9dca"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.8.1"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "6406b5112809c08b1baa5703ad274e1dded0652f"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.23"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fd75fa3a2080109a2c0ec9864a6e14c60cca3866"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.62.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MPC_jll]]
deps = ["Artifacts", "GMP_jll", "JLLWrappers", "Libdl", "MPFR_jll", "Pkg"]
git-tree-sha1 = "9618bed470dcb869f944f4fe4a9e76c4c8bf9a11"
uuid = "2ce0c516-f11f-5db3-98ad-e0e1048fbd70"
version = "1.2.1+0"

[[MPFR_jll]]
deps = ["Artifacts", "GMP_jll", "Libdl"]
uuid = "3a97d323-0669-5f0c-9066-3539efd106a3"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "35d435b512fbab1d1a29138b5229279925eba369"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.5.0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "7dc03c2b145168f5854085a16d054429d612b637"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.23.5"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "e071adf21e165ea0d904b595544a8e514c8bb42c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.19"

[[PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "e7bc80dc93f50857a5d1e3c8121495852f407e6a"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.4.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "eb35dcc66558b2dda84079b9a1be17557d32091a"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.12"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SymEngine]]
deps = ["Compat", "Libdl", "LinearAlgebra", "RecipesBase", "SpecialFunctions", "SymEngine_jll"]
git-tree-sha1 = "126c86efe59030cae872d92fabfaa62dac67381d"
uuid = "123dc426-2d89-5057-bbad-38513e3affd8"
version = "0.8.6"

[[SymEngine_jll]]
deps = ["GMP_jll", "Libdl", "MPC_jll", "MPFR_jll", "Pkg"]
git-tree-sha1 = "4dacada8e05ac49eb768219f8d02bc6b608627fb"
uuid = "3428059b-622b-5399-b16f-d347a77089a4"
version = "0.6.0+1"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─5c316980-d18d-4698-a841-e732f7632cec
# ╟─53ef0bfc-4239-11ec-0b4c-23f451fff4a6
# ╟─ca7d694b-182a-443d-b47d-1bfe4ed8039f
# ╟─9b3eee98-e481-4fb6-98c2-6ac408dcfe54
# ╠═6163277d-70d3-4a73-89df-65c329c2b818
# ╠═b1a45f30-beec-4089-904c-488b86b56a9e
# ╟─fcc24d08-bb9a-482f-987e-e64184c8d6f2
# ╠═843fef36-611c-4411-b31b-8a11e128881b
# ╟─3ba9cc34-0dcd-4c2e-b428-242e456bd436
# ╟─601e4aa9-e380-41a1-96a2-7089603889c3
# ╟─fefd6403-f46c-4eb1-b754-a85bcb75914c
# ╟─2ac3348d-196a-4507-b2f7-c575e42d7e7b
# ╟─09582278-6fed-4cac-9aaa-45cf0ac9fb6c
# ╟─58cb6931-91a5-4325-8be4-6675f7e142ed
# ╠═b059cb44-349a-48b5-a96e-62c4835fde10
# ╠═5b925811-6255-4e2e-b691-40869d65d6df
# ╠═a88b6949-4b4a-4f5a-a9a2-c6978cd0f758
# ╟─3af8c139-2e6c-4830-9b67-96f78356f521
# ╟─dd9bfbb1-aecf-458f-9a05-a93ff78fd741
# ╠═34b7e91e-d67f-4554-b985-b9100adda733
# ╠═2dbb5b13-790a-4ab7-95b1-b833c4cb027a
# ╟─f51233c4-ec66-4517-9109-5309601d1d87
# ╠═3729833f-80d4-4948-8d81-750008c8f16d
# ╟─7172d082-e6d2-419b-8bb6-75e30f1b4dfe
# ╠═e7841458-f641-48cf-8667-1e5b38cbd9f6
# ╠═abbc5a52-a02c-4f5b-bd1e-af5596455762
# ╠═95e688e2-9607-41a2-9098-626590bcf435
# ╟─8279fd8a-e447-49b6-b729-6e7b8883f5e4
# ╠═3cf9be4d-fa76-4264-b9b6-ff66bcf5db0e
# ╠═76a613f2-482f-4a4d-8236-debee05bef1b
# ╟─ed2ee298-ac4f-4ae3-a9e3-300040a706a8
# ╠═0bbaa5a8-8082-4697-ae98-92b2ae3769af
# ╠═dc21cc4b-aedd-42d7-b2a8-f36dfecee6f4
# ╟─dd0c1982-38f4-4752-916f-c05da365bade
# ╠═7fcebc5a-a8c7-47d8-90b0-7ee8cd579585
# ╟─202dc3b6-ddcb-463d-b8f2-a285a2ecb112
# ╟─91fd09a1-8b3a-4772-b6a5-7b149d91eb4d
# ╟─b49ca3b1-0d1b-4edb-8064-e8cd8d4db727
# ╠═ba891e20-db23-4b03-9495-19c19df940d3
# ╟─86f0e396-f81b-45be-94a7-90e40a8ba251
# ╠═12629919-26d3-4434-9c23-9778364fe71a
# ╟─9806ec5e-a884-41a1-980a-579915a33b8e
# ╠═b1c207b7-9d70-453c-b554-1c91f59ada0a
# ╟─8300dbb5-0eb6-4f84-80c6-24c4443b1f29
# ╠═3a40b68f-83cf-4db9-b301-492e5cedcd13
# ╠═b901c4aa-38f8-476a-8c9e-7eb523f59438
# ╟─d4af5141-422b-4941-8dc7-f2b4b09029c0
# ╠═3fd2f03a-fc52-4009-b284-0def00be601f
# ╠═27d955de-8d97-43e4-9176-aad5456eb797
# ╟─645ef857-aff9-4dee-bfd6-72fe9d542375
# ╟─06ca10a8-c922-4252-91d2-e025ab306f02
# ╟─ab589e93-a4ca-45be-882c-bc3da47e4d1c
# ╠═d4bb171d-3c1c-463a-9360-c78bdfc83363
# ╟─b600aafb-7d23-417a-a8c9-597d95182469
# ╟─bf8dfa21-29e4-4d6e-a876-ba1a6ca313b1
# ╠═068dd98e-8507-4380-a4b2-f6fee80adaaa
# ╟─4b3f4b1b-1b22-4e2e-be5b-d44d74d8da0e
# ╠═86440ba5-4b5f-440b-87e4-5446217dd073
# ╠═3e480576-ed7d-4f2d-bcd1-d7d1cbbeccf9
# ╟─bc52bf0c-6cd1-488d-a9c1-7a91a582dda9
# ╟─73fea39a-3ba6-4a37-9014-261a95acc084
# ╟─a5e5f5bc-cc5e-4f70-91ac-43fb21f2cada
# ╟─7ee3eb27-c1e1-477e-bdd0-894e4317c559
# ╠═24266569-cd10-4765-95fd-61b06027dd0e
# ╟─f8e89e44-d12c-43c3-b1ec-01c68f33c3b4
# ╟─d9d7d94d-e457-4354-a1a3-4a230c9ddc29
# ╟─173b83be-dec2-487b-96ce-12cb5fba8be0
# ╟─b0a8a72c-3eb1-431d-9b30-17115e60025a
# ╟─c0edc9ae-ff2a-4224-820a-1a8844f41291
# ╟─443ec353-c574-4950-ad67-483791d8e934
# ╟─0b77e9c2-f360-498c-8a0a-157693866902
# ╟─a7a07e38-6900-4fd1-8a87-0e16d92a5256
# ╟─dac4173c-9d3b-4573-b1ba-13c6b7cc5f30
# ╟─9315c9b1-87fa-4e91-a78d-c24a3007139b
# ╟─c6464aec-bdf5-49b7-a5d2-45c2f6471bc7
# ╟─347a3819-9300-49f5-97b4-d1847c5ee98c
# ╟─6802424d-6072-4692-add2-d34abb3ce6b7
# ╟─31b1bad8-5a3d-4d9e-93c8-45e854cf88f8
# ╠═d9238a26-e792-44fc-be3d-7d8ec7e0117d
# ╟─eb2d7221-25b4-4836-b818-3ed944570040
# ╠═66f0d9bb-7d04-4e82-b9dd-55510971691b
# ╟─4c60c221-545c-4050-bfea-211048a36bce
# ╠═2d1f128c-bcfa-4017-9690-01f3f75c3efa
# ╠═b4ade3a3-668e-495b-9b7b-ad45fdf2655b
# ╟─9431caba-619d-4104-a267-914a9bcc78ef
# ╟─58f32a65-1ef8-4d9a-a874-00f7df563b3c
# ╠═4d2a5726-2704-4b63-b334-df5175278b18
# ╟─9f238c4a-c557-4c57-a24c-6d221d592a18
# ╠═f061e908-0687-4375-84e1-386a0dd48b39
# ╠═eb65a331-c977-4b0f-8add-873bd89095f4
# ╟─5e09215e-1f9b-47a6-baf8-46f1f0dc1a20
# ╟─278cc047-83ee-49b1-a0e3-d2d779c1bc17
# ╟─5f3ad56f-5f8f-4b51-b45c-46c37eaeced4
# ╠═d146a1e2-8067-4e25-b0cd-2a041162acb9
# ╠═3722538e-76e9-4bab-bfa9-57eff72802b7
# ╠═f368672a-5c78-4d2a-aea9-f2a2c1ee0a54
# ╠═63703f51-bf0a-42c1-b981-3191d88b4901
# ╠═d4c22f7b-31f5-4f41-8731-2f6189d231b4
# ╠═33e3b11c-b1b4-4c64-b742-734ebd06926e
# ╟─2e3243dc-f489-4117-82f8-7d05f5188429
# ╟─5a5bb3c5-f8da-4f7b-9b44-b54025d7e71c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
