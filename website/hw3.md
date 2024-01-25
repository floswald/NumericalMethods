~~~
<h1>Homework 3 : Optimizing a Likelihood Function</h1>
~~~


This homework asks you to think about a relatively simple case of nonlinear optimization: the well known Probit model. Of course there are plenty of _canned_ solutions to this problem, however, here we want to hone our skills a bit when it comes to actually implementing a nonlinear optimization problem. We will see that 

1. providing gradient and/or hessian information to an algorithm changes the speed and quality of convergence
2. Different algorithms reach slightly different optima
3. there are several ways to obtain standard errors in a likelihood estimation setting.

> Get the [notebook here](https://github.com/floswald/NumericalMethods/blob/master/notebooks/homework3/Probit.jl)

### Submission

* As usual, teams of at least 2.
* submit static html file.
* dropbox link via slack.