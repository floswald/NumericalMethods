### A Pluto.jl notebook ###
# v0.19.40

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
	using SymEngine
end

# ╔═╡ d4bb171d-3c1c-463a-9360-c78bdfc83363
begin
	using Calculus
	# also can compute gradients for multidim functions
	Calculus.gradient(x->x[1]^2 * exp(3x[2]),ones(2)), Calculus.hessian( x->x[1]^2 * exp(3x[2]),ones(2))
end

# ╔═╡ 86440ba5-4b5f-440b-87e4-5446217dd073
using ForwardDiff    # one particular AD package in julia

# ╔═╡ 7a7fc4fc-be68-40d6-868b-d141a7054319
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
		font-size: x-large
	}
</style>
"""

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
md"""
#
"""

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

# ╔═╡ 9a5cb736-237c-4b3d-9820-b05ec4c961d5
md"""
#
"""

# ╔═╡ 89de388b-4bd1-4814-8378-10bfd0ac3f3d
md"""
#
"""

# ╔═╡ 274f5fd9-904d-4f11-b4e1-93a37e206080
md"""
#
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

# ╔═╡ 1d698018-8b77-490d-ad3a-6c7001aa99ab
md"""
#
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
# 

Ok enough of that. Let's get a visual of the Rosenbrock function finally!
"""

# ╔═╡ ed2ee298-ac4f-4ae3-a9e3-300040a706a8
md"""
#

### Keyword Arguments

In fact, the numbers `1` and `5` in 

$$f(x,y) = (1-x)^2  + 5(y-x^2)^2$$

are just *parameters*, i.e. the function definition can be changed by varying those. Let's get a version of `rosen()` which allows this, then let's investigate the plot again:

"""

# ╔═╡ 0bbaa5a8-8082-4697-ae98-92b2ae3769af
rosenkw(x,y ; a = 1, b = 5) = (a - x)^2 + b*(y - x^2)^2  # notice the ; 

# ╔═╡ 5abc4cf1-7fe1-4d5e-9077-262984d07b4c
md"""
#
"""

# ╔═╡ dd0c1982-38f4-4752-916f-c05da365bade
md"""
* alright, not bad. but how can I change the a and b values now?
* One solution is to pass an *anonymous function* which will *enclose* the values for `a` and `b` (it is hence called a `closure`):
"""

# ╔═╡ f655db71-18c6-40db-83c8-0035e37e6eda
md"""
#
"""

# ╔═╡ 202dc3b6-ddcb-463d-b8f2-a285a2ecb112
md"""
This wouldn't be a proper pluto session if we wouldn't hook those values up to a slider, would it? Let's do it!
"""

# ╔═╡ 29d33b1f-8901-4fee-aa85-11adb6ebad1b
md"""
#
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

# ╔═╡ edd64823-b054-4974-b817-853319a62bcd
u(c; σ = 2) = ((c)^(1-σ)) / (1-σ)

# ╔═╡ 986fcae1-138c-42f6-810e-e3c193f669bb
u(2.2)

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
	c = 2.2
	∂u∂c = (u(c + h) - u(c)) / h  # definition from above!

	Dict(:finite_diff => ∂u∂c, :truth_wiktor => c^-2)
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

$$\left(\frac{\partial f(x,y)}{\partial x}, \frac{\partial f(x,y)}{\partial y}\right) = (0,0)$$
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

# ╔═╡ 068dd98e-8507-4380-a4b2-f6fee80adaaa
begin
	
	x = symbols("x");
	f = x^2 + x/2 - sin(x)/x; diff(f, x)
end

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


# 

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
#

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

# ╔═╡ bd866fc5-2bf7-49dc-971e-7cd16d11d68e
md"""
#
"""

# ╔═╡ 24266569-cd10-4765-95fd-61b06027dd0e
PlutoUI.LocalResource("./optimization/callgraph.png")

# ╔═╡ f8e89e44-d12c-43c3-b1ec-01c68f33c3b4
md"""

#


### Accumulating *forwards* along the call graph

* Let's illustrate how AD in forward mode works for $x=3,y=2$ and the example at hand. Remember that
    $$f(x,y) = \ln(xy + \max(x,2))$$
    and, hence 
    $$f(3,2) = \ln( 6 + 3 ) = \ln 9 \text{  and  }\frac{\partial f}{\partial x} = \frac{1}{6 + 3}(2 + 1) = \frac{1}{3}$$
* We start at the left side of this graph with the inputs. 
* The key is for each quantity to compute both the value **and** it's partial derivative wrt $x$ in this case.



"""

# ╔═╡ 64d23de8-d271-484e-b051-be0b0fb2be3f
md"""
#
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
function rosendata(f::Function;npoints = 30)
	x = y = range(-2,stop = 2, length = npoints)  # x and y axis
	rosenvals = [f(ix,iy) for ix in x, iy in y]  # f evaluations
	(x,y,rosenvals)
end

# ╔═╡ 76a613f2-482f-4a4d-8236-debee05bef1b
function rosenplotter(f::Function)
	x,y,vals = rosendata(f)  # get the data
	# plotting
	surface(x,y,vals, fillcolor = :thermal,colorbar=false, 
		alpha = 0.9,xlab = "x",ylab = "y", zlab = "z", zlim= (0,180))
end

# ╔═╡ 3cf9be4d-fa76-4264-b9b6-ff66bcf5db0e
rosenplotter(rosen₃)

# ╔═╡ dc21cc4b-aedd-42d7-b2a8-f36dfecee6f4
rosenplotter(rosenkw)

# ╔═╡ 7fcebc5a-a8c7-47d8-90b0-7ee8cd579585
rosenplotter( (x,y) -> rosenkw(x,y, a=1.2, b=2 ) ) # notice the `,` when calling

# ╔═╡ ba891e20-db23-4b03-9495-19c19df940d3
rosenplotter( (x,y) -> rosenkw(x,y, a=a, b=b ))

# ╔═╡ 12629919-26d3-4434-9c23-9778364fe71a
let
	x,y,z = rosendata(rosenkw,npoints = 100)  # default a,b
	contour(x,y,z, fill = false, color = :deep,levels=[collect(0:0.2:175)...])
	scatter!([1.0],[1.0], m=:c, c=:red, label = "(1,1)")
end

# ╔═╡ b1c207b7-9d70-453c-b554-1c91f59ada0a
let
	x,y,z = rosendata(rosenkw,npoints = 100)  # default a,b
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

# ╔═╡ 1e272a4c-cad6-423b-b5f3-f16b404e63a2
sb = md"""
#
"""

# ╔═╡ 1fdccba1-8ea3-41ca-9095-aa0e5eefeafc
sb

# ╔═╡ 173b83be-dec2-487b-96ce-12cb5fba8be0
sb

# ╔═╡ c0edc9ae-ff2a-4224-820a-1a8844f41291
sb

# ╔═╡ 0b77e9c2-f360-498c-8a0a-157693866902
sb

# ╔═╡ dac4173c-9d3b-4573-b1ba-13c6b7cc5f30
sb

# ╔═╡ 5e213422-7503-4ea7-ad63-99e271459cf1
sb

# ╔═╡ eacb00fe-7a99-40bc-8ff3-88822ebe94bb
sb

# ╔═╡ 8b5c1e3a-282e-4a27-9d41-f3b20a00b82f
sb

# ╔═╡ 73161163-b5f6-43b2-ab6c-ed81375e01ff
sb

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
ForwardDiff = "~0.10.36"
LaTeXStrings = "~1.3.1"
Optim = "~1.9.2"
Plots = "~1.40.1"
PlutoUI = "~0.7.55"
SymEngine = "~0.11.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "d40a651448e6e1724669bc5df438751d49c5089f"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "c278dfab760520b8bb7e9511b968bf4ba38b7acc"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.3"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "0fb305e0253fd4e833d486914367a2ee2c2e78d0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.1"

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "bbec08a37f8722786d87bedf84eae19c020c4efa"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.7.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1f03a9fa24271160ed7e73051fba3c1a759b53f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.4.0"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "9b1ca1aa6ce3f71b3d1840c538a8210a043625eb"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "75bd5b6fc5089df449b5d35fa501c846c9b6549b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.12.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "9c4708e3ed2b799e6124b5673a712dda0b596a9b"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "ac67408d9ddf207de5cfa9a97e114352430f01ed"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.16"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "73d1214fec245096717847c62d389a5d2ac86504"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.22.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "ff38ba61beff76b8f4acad8ab0c97ef73bb670cb"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.9+0"

[[deps.GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"
version = "6.2.1+6"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "3458564589be207fa6a77dbbf8b97674c9836aab"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "77f81da2964cc9fa7c0127f941e8bce37f7f1d70"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.2+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "abbbb9ec3afd783a7cbd82ef01dcd088ea051398"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.1"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60b1194df0a3298f460063de985eae7b01bc011a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.1+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MPC_jll]]
deps = ["Artifacts", "GMP_jll", "JLLWrappers", "Libdl", "MPFR_jll", "Pkg"]
git-tree-sha1 = "9618bed470dcb869f944f4fe4a9e76c4c8bf9a11"
uuid = "2ce0c516-f11f-5db3-98ad-e0e1048fbd70"
version = "1.2.1+0"

[[deps.MPFR_jll]]
deps = ["Artifacts", "GMP_jll", "Libdl"]
uuid = "3a97d323-0669-5f0c-9066-3539efd106a3"
version = "4.2.0+1"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "8b40681684df46785a0012d352982e22ac3be59e"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.25.2"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "806eea990fb41f9b36f1253e5697aa645bf6a9f8"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.4.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60e3045590bd104a16fefb12836c00c0ef8c7f8c"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "MathOptInterface", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "d024bfb56144d947d4fafcd9cb5cafbe3410b133"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.9.2"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "862942baf5663da528f66d24996eb6da85218e76"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "c4fa93d7d66acad8f6f4ff439576da9d2e890ee0"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.1"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "68723afdb616445c6caaef6255067a8339f91325"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.55"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.SymEngine]]
deps = ["Compat", "Libdl", "LinearAlgebra", "RecipesBase", "Serialization", "SpecialFunctions", "SymEngine_jll"]
git-tree-sha1 = "62926b1f0a2358480dde70bb6fa1d01ca58fd2c9"
uuid = "123dc426-2d89-5057-bbad-38513e3affd8"
version = "0.11.0"

    [deps.SymEngine.extensions]
    SymEngineSymbolicUtilsExt = "SymbolicUtils"

    [deps.SymEngine.weakdeps]
    SymbolicUtils = "d1185830-fcd6-423d-90d6-eec64667417b"

[[deps.SymEngine_jll]]
deps = ["Artifacts", "GMP_jll", "JLLWrappers", "Libdl", "MPC_jll", "MPFR_jll"]
git-tree-sha1 = "25e746f4a7bee5bbbf96eeea3bba10dea09c1333"
uuid = "3428059b-622b-5399-b16f-d347a77089a4"
version = "0.11.1+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "54194d92959d8ebaa8e26227dbe3cdefcdcd594f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.3"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "3c793be6df9dd77a0cf49d80984ef9ff996948fa"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.19.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "801cbe47eae69adc50f36c3caec4758d2650741b"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.2+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522b8414d40c4cbbab8dee346ac3a09f9768f25d"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.5+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "93284c28274d9e75218a416c65ec49d0e0fcdf3d"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.40+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─7a7fc4fc-be68-40d6-868b-d141a7054319
# ╟─5c316980-d18d-4698-a841-e732f7632cec
# ╟─53ef0bfc-4239-11ec-0b4c-23f451fff4a6
# ╟─ca7d694b-182a-443d-b47d-1bfe4ed8039f
# ╟─9b3eee98-e481-4fb6-98c2-6ac408dcfe54
# ╟─6163277d-70d3-4a73-89df-65c329c2b818
# ╟─b1a45f30-beec-4089-904c-488b86b56a9e
# ╟─fcc24d08-bb9a-482f-987e-e64184c8d6f2
# ╠═843fef36-611c-4411-b31b-8a11e128881b
# ╟─3ba9cc34-0dcd-4c2e-b428-242e456bd436
# ╟─601e4aa9-e380-41a1-96a2-7089603889c3
# ╟─fefd6403-f46c-4eb1-b754-a85bcb75914c
# ╟─2ac3348d-196a-4507-b2f7-c575e42d7e7b
# ╟─09582278-6fed-4cac-9aaa-45cf0ac9fb6c
# ╟─58cb6931-91a5-4325-8be4-6675f7e142ed
# ╟─9a5cb736-237c-4b3d-9820-b05ec4c961d5
# ╠═b059cb44-349a-48b5-a96e-62c4835fde10
# ╟─89de388b-4bd1-4814-8378-10bfd0ac3f3d
# ╠═5b925811-6255-4e2e-b691-40869d65d6df
# ╟─274f5fd9-904d-4f11-b4e1-93a37e206080
# ╠═a88b6949-4b4a-4f5a-a9a2-c6978cd0f758
# ╟─3af8c139-2e6c-4830-9b67-96f78356f521
# ╟─dd9bfbb1-aecf-458f-9a05-a93ff78fd741
# ╠═34b7e91e-d67f-4554-b985-b9100adda733
# ╠═2dbb5b13-790a-4ab7-95b1-b833c4cb027a
# ╟─f51233c4-ec66-4517-9109-5309601d1d87
# ╟─1d698018-8b77-490d-ad3a-6c7001aa99ab
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
# ╟─5abc4cf1-7fe1-4d5e-9077-262984d07b4c
# ╠═dc21cc4b-aedd-42d7-b2a8-f36dfecee6f4
# ╟─dd0c1982-38f4-4752-916f-c05da365bade
# ╟─f655db71-18c6-40db-83c8-0035e37e6eda
# ╠═7fcebc5a-a8c7-47d8-90b0-7ee8cd579585
# ╟─202dc3b6-ddcb-463d-b8f2-a285a2ecb112
# ╟─29d33b1f-8901-4fee-aa85-11adb6ebad1b
# ╟─91fd09a1-8b3a-4772-b6a5-7b149d91eb4d
# ╟─b49ca3b1-0d1b-4edb-8064-e8cd8d4db727
# ╠═ba891e20-db23-4b03-9495-19c19df940d3
# ╟─86f0e396-f81b-45be-94a7-90e40a8ba251
# ╠═12629919-26d3-4434-9c23-9778364fe71a
# ╟─9806ec5e-a884-41a1-980a-579915a33b8e
# ╠═b1c207b7-9d70-453c-b554-1c91f59ada0a
# ╟─8300dbb5-0eb6-4f84-80c6-24c4443b1f29
# ╠═edd64823-b054-4974-b817-853319a62bcd
# ╠═986fcae1-138c-42f6-810e-e3c193f669bb
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
# ╟─1fdccba1-8ea3-41ca-9095-aa0e5eefeafc
# ╠═86440ba5-4b5f-440b-87e4-5446217dd073
# ╠═3e480576-ed7d-4f2d-bcd1-d7d1cbbeccf9
# ╟─bc52bf0c-6cd1-488d-a9c1-7a91a582dda9
# ╟─73fea39a-3ba6-4a37-9014-261a95acc084
# ╟─a5e5f5bc-cc5e-4f70-91ac-43fb21f2cada
# ╟─7ee3eb27-c1e1-477e-bdd0-894e4317c559
# ╟─bd866fc5-2bf7-49dc-971e-7cd16d11d68e
# ╟─24266569-cd10-4765-95fd-61b06027dd0e
# ╟─f8e89e44-d12c-43c3-b1ec-01c68f33c3b4
# ╟─64d23de8-d271-484e-b051-be0b0fb2be3f
# ╟─d9d7d94d-e457-4354-a1a3-4a230c9ddc29
# ╟─173b83be-dec2-487b-96ce-12cb5fba8be0
# ╟─b0a8a72c-3eb1-431d-9b30-17115e60025a
# ╟─c0edc9ae-ff2a-4224-820a-1a8844f41291
# ╟─443ec353-c574-4950-ad67-483791d8e934
# ╟─0b77e9c2-f360-498c-8a0a-157693866902
# ╟─a7a07e38-6900-4fd1-8a87-0e16d92a5256
# ╟─dac4173c-9d3b-4573-b1ba-13c6b7cc5f30
# ╟─9315c9b1-87fa-4e91-a78d-c24a3007139b
# ╟─5e213422-7503-4ea7-ad63-99e271459cf1
# ╟─c6464aec-bdf5-49b7-a5d2-45c2f6471bc7
# ╟─347a3819-9300-49f5-97b4-d1847c5ee98c
# ╟─eacb00fe-7a99-40bc-8ff3-88822ebe94bb
# ╟─6802424d-6072-4692-add2-d34abb3ce6b7
# ╟─8b5c1e3a-282e-4a27-9d41-f3b20a00b82f
# ╟─31b1bad8-5a3d-4d9e-93c8-45e854cf88f8
# ╠═d9238a26-e792-44fc-be3d-7d8ec7e0117d
# ╟─eb2d7221-25b4-4836-b818-3ed944570040
# ╠═66f0d9bb-7d04-4e82-b9dd-55510971691b
# ╟─73161163-b5f6-43b2-ab6c-ed81375e01ff
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
# ╟─1e272a4c-cad6-423b-b5f3-f16b404e63a2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
