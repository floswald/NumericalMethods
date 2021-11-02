
~~~
<h1>Lecture 6: Function Approximation</h1>
~~~


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

