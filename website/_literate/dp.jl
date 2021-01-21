# # Numerical Dynamic Programming
 
# Florian Oswald, Sciences Po, 2019




# ## Intro

# * Numerical Dynamic Programming (DP) is widely used to solve dynamic models.
# * You are familiar with the technique from your core macro course.
# * We will illustrate some ways to solve dynamic programs.
#     1. Models with one discrete or continuous choice variable
#     1. Models with several choice variables
#     1. Models with a discrete-continuous choice combination
# * We will go through:
#     1. Value Function Iteration (VFI)
#     1. Policy function iteration (PFI)
#     1. Projection Methods
#     1. Endogenous Grid Method (EGM)
#     1. Discrete Choice Endogenous Grid Method (DCEGM)


# ## Dynamic Programming Theory

# * Payoffs over time are 
# 	$$U=\sum_{t=1}^{\infty}\beta^{t}u\left(s_{t},c_{t}\right) $$
# 	where $\beta<1$ is a discount factor, $s_{t}$ is the state, $c_{t}$ is the control.

# * The state (vector) evolves as $s_{t+1}=h(s_{t},c_{t})$.
# * All past decisions are contained in $s_{t}$.

# ### Assumptions

# * Let $c_{t}\in C(s_{t}),s_{t}\in S$ and assume $u$ is bounded in $(c,s)\in C\times S$.
# * Stationarity: neither payoff $u$ nor transition $h$ depend on time.
# * Write the problem as 
# 	$$ v(s)=\max_{s'\in\Gamma(s)}u(s,s')+\beta v(s') $$
# * $\Gamma(s)$ is the constraint set (or feasible set) for $s'$ when the current state is $s$

# ### Existence

# **Theorem.** Assume that $u(s,s')$ is real-valued, continuous, and bounded, that $\beta\in(0,1)$, and that the constraint set $\Gamma(s)$ is nonempty, compact, and continuous. Then there exists a unique function $v(s)$ that solves the above functional equation.

# **Proof.** [@stokeylucas] <cite data-cite=stokeylucas></cite> theoreom 4.6.

# # Solution Methods

# ## Value Function Iteration (VFI)

# * Find the fix point of the functional equation by iterating on it until the distance between consecutive iterations becomes small.
# * Motivated by the Bellman Operator, and it's characterization in the Continuous Mapping Theorem.

# ## Discrete DP VFI

# * Represents and solves the functional problem in $\mathbb{R}$ on a finite set of grid points only.
# * Widely used method.
#    * Simple (+)
#    * Robust (+)
#    * Slow (-)
#    * Imprecise (-)
# * Precision depends on number of discretization points used. 
# * High-dimensional problems are difficult to tackle with this method because of the curse of dimensionality.



# ### Deterministic growth model with Discrete VFI

# * We have this theoretical model:

# \begin{eqnarray}
#    V(k) &=& \max_{0<k'<f(k)} u(f(k) - k') + \beta V(k')\\
#   f(k)  &=& k^\alpha
# \end{eqnarray}

# * and we employ the followign numerical approximation:
# 	$$ V(k_i) = \max_{i'=1,2,\dots,n} u(f(k_i) - k_{i'}) + \beta V(i') $$

# * The iteration is then on successive iterates of $V$: The LHS gets updated in each iteration!

# $$
# \begin{aligned}
# 	V^{r+1}(k_i) &= \max_{i'=1,2,\dots,n} u(f(k_i) - k_{i'}) + \beta V^{r}(i') \\
# 	V^{r+2}(k_i) &= \max_{i'=1,2,\dots,n} u(f(k_i) - k_{i'}) + \beta V^{r+1}(i') \\
# 	... & 
# \end{aligned}
# $$

# * And it stops at iteration $r$ if $d(V^{r},V^{r-1}) < \text{tol}$
# * You choose a measure of *distance*, $d(\cdot,\cdot)$, and a level of tolerance.
# * $V^{r}$ is usually an *array*. So $d$ will be some kind of *norm*.
# * maximal absolute distance
# * mean squared distance

# ### Exercise 1: Implement discrete VFI

# ## Checklist

# 1. Set parameter values
# 1. define a grid for state variable $k \in [0,2]$
# 1. initialize value function $V$
# 1. start iteration, repeatedly computing a new version of $V$.
# 1. stop if $d(V^{r},V^{r-1}) < \text{tol}$.
# 1. plot value and policy function 
# 1. report the maximum error of both wrt to analytic solution


alpha     = 0.65
beta      = 0.95
grid_max  = 2  # upper bound of capital grid
n         = 150  # number of grid points
N_iter    = 3000  # number of iterations
kgrid     = 1e-2:(grid_max-1e-2)/(n-1):grid_max  # equispaced grid
f(x) = x^alpha  # defines the production function f(k)
tol = 1e-9

