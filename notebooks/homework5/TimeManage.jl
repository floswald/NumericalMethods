### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ af5b2742-8b1b-11eb-3bd7-df721374aa40
using Plots

# ╔═╡ dd3a69da-8aea-11eb-14c6-9b0d053d0f15
md"
# Optimal Time Management

>Florian Oswald, SciencesPo 2021

>First part is based on problem 2.13 of _Approximate Dynamic Programming_ by Warren B. Powell

You suddenly realize towards the end of the semester that your have three courses that have assigned a term project instead of a final exam. You quickly estimate how much each one will take to get the maximum of 100 points. You then guess that if you invest $t$ hours in a project, which you estimated would need $T$ to get 100 points, your score will be

$$R = 100 \sqrt{t/T}$$

In other words, there are declining marginal returns to putting more work into a project.

You decide that you cannot spend morea than a total of 30 hours on those projects, and you want to choose a value of $t$ for each project that is a multiple of 5. Also, you feel you need to spend at least 5 hours on any project (you cannot totally ignore any one project). Here is your estimate of time it takes to get full points:

Project | Completion time $T$ to get 100 points
:--------: | :------------:
1 | 20
2 | 15
3 | 10

You decide to solve the problem as a dynamic program.
"


# ╔═╡ bfa89ede-8aeb-11eb-0b37-c50dc5fe742b
md"
# Question 1

* What is the state variable, the choice variable and decision epoch for this problem? Keep in mind that both variables could be vectors.
"

# ╔═╡ 55885146-8b00-11eb-0a64-2d7202c60f10
md"
# Question 2

* What is the law of motion of your state vector, i.e. write down the transition function.
"

# ╔═╡ dfa9f2ca-8aeb-11eb-2797-f716cc8fa3b5
md"
# Question 3

* What is your reward function?
"

# ╔═╡ e6e62a7a-8aeb-11eb-0ec7-ed7893ab04a7
md"
# Question 4

* Write out the problem as an optimization problem.
"

# ╔═╡ a2553488-8b0f-11eb-191f-a1afb963066f
md"
# Question 5

* Solve your optimization problem either on a piece of paper or by using your computer.
"

# ╔═╡ f28bfc8a-8aeb-11eb-3996-a17dad8a13a0
md"
# Question 6

* Write out the Bellman Equation for this problem. Start with the final period, which is necessarily going to be period 6 (since 6*5 = 30 hours).
"

# ╔═╡ fdfac466-8aeb-11eb-066c-5308ca95d208
md"
# Question 7

* Solve the Bellman Equation to find the optimal time investment strategy.
"

# ╔═╡ 4a1db708-8b1a-11eb-0bf1-a57233f4cfb8
md"
# Modifying the Setup: True Dynamics

Let's modify the problem slightly. Suppose our table now looks like

Project | Completion time $T$ to get 100 points | Fun Factor Function
:--------: | :------------: | :-----:
1 | 20  | $f(x_1,T_1,\theta_1) = \theta_1(T_1 - x_1)^{3}$
2 | 15  | $f(x_2,T_2,\theta_2) = \theta_2(T_2 - x_2)^{3}$
3 | 10  | $f(x_3,T_3,\theta_3) = \theta_3(T_3 - x_3)^{3}$

So there is now additional element: Different projects are differently _enjoyable_ to work on. Here again $x_i$ is the _time already spent_ on project $i$. _fun_ acrues to the decision maker _while working on the project_. It's the **period payoff**. 

Here is a plot of how _fun_ behaves with time spent on each project. You can see that even though some projects are a lot of fun to work on, all end up with zero fun if you work on them until you reach 100 points. Some projects are so boring that you get negative fun from them. This is a purely artificial example of course 😄.

Let's set as default values for $\theta = [0.01,0.01,-0.02]$
"

# ╔═╡ 2b354bea-8b1e-11eb-0d8e-25fa332b330e
md"

# Question 8

* Re-formulate the Bellman Equation. Remember the unit vector definition $e_2 = [0,1,0]$. For our purposes, we move in steps of 5, so let's redefine that as

$$\epsilon_2 \equiv 5 e_2 = [0,5,0]$$ where the index 2 is of course just an example. I propose to , in each period just choose the _index_ of the project to work on, i.e. choose $i \in\{1,2,3\}$. You can then reformulate the law of motion as

$$x(t+1) = x(t) + \epsilon_i$$ which will add 5 to the chosen project's counter.


"

# ╔═╡ 5b040ac6-8b25-11eb-0cf0-9bc220cef4e0
md"
# Question 9

* solve this on your computer! 
* We are looking for the _policy function_ $i^*(x(t))$ which tells us which project to work on, given state $x(t)$!
* Make a plot that shows the optimal choice of project at each period and the level of associated utility!
"

# ╔═╡ Cell order:
# ╟─dd3a69da-8aea-11eb-14c6-9b0d053d0f15
# ╟─bfa89ede-8aeb-11eb-0b37-c50dc5fe742b
# ╟─55885146-8b00-11eb-0a64-2d7202c60f10
# ╟─dfa9f2ca-8aeb-11eb-2797-f716cc8fa3b5
# ╟─e6e62a7a-8aeb-11eb-0ec7-ed7893ab04a7
# ╟─a2553488-8b0f-11eb-191f-a1afb963066f
# ╟─f28bfc8a-8aeb-11eb-3996-a17dad8a13a0
# ╟─fdfac466-8aeb-11eb-066c-5308ca95d208
# ╟─4a1db708-8b1a-11eb-0bf1-a57233f4cfb8
# ╠═af5b2742-8b1b-11eb-3bd7-df721374aa40
# ╟─2b354bea-8b1e-11eb-0d8e-25fa332b330e
# ╟─5b040ac6-8b25-11eb-0cf0-9bc220cef4e0
