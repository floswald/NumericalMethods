
~~~
<h1>Lecture 6: Function Approximation</h1>
~~~

My neighbor was doing constructions works at home, so 👷👷👷 were pretty busy on the drill ⚒️ during the lecture. The video is painful to watch - sorry!

## Julia-stuff

* We looked at [https://github.com/JuliaLang/Example.jl](https://github.com/JuliaLang/Example.jl) to introduce package development.
* We introduced two ways to develop packages: 
    * plain vanilla in your terminal, editing code in an editor, and repeatedly doing `include("src/Example.jl")` to update the module
    * using [Revise.jl](https://github.com/timholy/Revise.jl) from wihtin VScode to *automatically* update the code we modify in the running julia session.
* We introduced _test driven development_ and the julia built-in testing framework.

## Function Approximation

You can find the notebooks for this lecture [here](https://github.com/floswald/NumericalMethods/tree/master/lecture_notebooks/week6)


* We introduced some basic concepts:
    * Basis Functions
    * Types of interpolations/approximations
* We introduced _orthogonal polynomials_ like the Chebyshev polynomials and their basis functions
* We looked at _splines_
* We introduced finally some julia packages to do interpolation.

## Using JuMP.jl in Research

* I showed you code from one of my papers that uses JuMP.jl to compute the solution to an equation system.
* I really want to show you the interactive dashboard that I use to calibrate the model but my computer let me down. We'll try again! 🙂