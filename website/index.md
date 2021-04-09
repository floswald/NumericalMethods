
# Computational Economics for PhDs


* **Teacher:** Florian Oswald, [florian.oswald@sciencespo.fr](mailto:florian.oswald@sciencespo.fr)
* **Class Times:** Fridays 10:15-12:15 starting 29 Jan 2021
* **Class Location:** Zoom
* **Slack**: There will be a slack channel for all communication

## Course Description

This is a course for PhD students at the [Department of Economics at Sciences Po](http://econ.sciences-po.fr) in Computational Economics. 

## Course Overview

In this course you will learn about some commonly used methods in Computational Economics. These methods are being used in all fields of Economics. The course has a clear focus on applying what you learn. We will cover the theoretical concepts that underlie each topic, but you should expect a fair amount of *hands on* action required on your behalf. In the words of the great [Che-Lin Su](https://bfi.uchicago.edu/people/che-lin-su):

> Doing Computation is the only way to learn Computation. Doing Computation is the only way to learn Computation. Doing Computation is the only way to learn Computation.

True to that motto, there will be homeworks for you to try out what you learned in class. There will also be a term paper.

## Prerequisites

1. You need a laptop. 
1. You should be familiar with the material from *Introduction to Programming*  taught by Clement Mazet in M1. Check out the materials [here](https://cms27.github.io/teaching/)
1. You must sign up for a free account at github.com. Choose a reasonable user name and upload a profile picture.
1. **Before** you come the first class, please do this:
    1. Download the latest [stable `julia` release](https://julialang.org/downloads/) for your OS.
    1. Download the [`VSCode Editor`](https://code.visualstudio.com)

### Getting Programming Skills

1. Check out [Clement Mazet's materials](https://cms27.github.io/teaching/). You must know this level.
1. We will be using [Julia](http://julialang.org/) for this course. 
    - [Noteworthy Differences from Other Languages](https://docs.julialang.org/en/v1/manual/noteworthy-differences/)
    - [MATLAB, Python, Julia Syntax Comparison](http://cheatsheets.quantecon.org/)
1. Clement in his course will introduce you to things like the [Unix Shell](https://en.wikipedia.org/wiki/Unix_shell) and the verion control system Git. Both of those are very useful - for this course, and for the rest of your life as a scientist. 
1. [What is Version Control? watch this 5 minute video.](http://git-scm.com/video/what-is-version-control) and go back to Clement's stuff if unclear.



## Term Project

This year your term project will be to replicate a paper published in an economics journal. Ideally this would be related to your field of interest. The requirements for choice of paper to replicate are:

1. Published version and replication kit is available online.
2. The paper to replicate must not use julia.
3. You must use julia for your replication.
    * Ideally your choice will involve at least some level of computational interest (i.e. more than an IV regression)
    * However, you can replicate a paper with an IV regression, but you have to go all the way to get the exact same results as in the paper. I.e. if the author typed the stata command `ivreg2 lw s expr tenure rns smsa _I* (iq=med kww age), cluster(year)` you will have to write (or find) julia code which will match all output from this, including standard errors. I do not recommend to do this.
4. You need to set up a public github repository where you will build a documentation website of your implementation. You'll learn how to do this in the course.
5. I encourage you to let the world know about your replication effort via social media and/or email to the authors directly. This is independent of whether you were able or not to replicate the results. Replication is not about finding errors in other peoples' work. If you are able to replicate some result in julia, this may be very interesting for others.

> Please use the starter kit available at [https://github.com/floswald/Replicate.jl](https://github.com/floswald/Replicate.jl) to get going!

### Resources

* [here is a great list by the AEA](https://www.aeaweb.org/rfe/showCat.php?cat_id=9)
* [ECTA code and data](https://www.econometricsociety.org/publications/econometrica/journal-materials/supplemental-materials)
* [RevEconDynamics codes](https://ideas.repec.org/s/red/ccodes.html)
* Each issue of RevEconDynamics , e.g. [https://www.economicdynamics.org/volume-39-2021/](https://www.economicdynamics.org/volume-39-2021/)
* [The AEA Data Editor's website](https://aeadataeditor.github.io/talks/)
* [The Restud Data Editor](https://restud.github.io/data-editor/replicate/#replicate-a-paper) and their [zenodo repo of replication kits](https://zenodo.org/communities/restud-replication/?page=1&size=20)
* [The Social Science Data Editor's joint website](https://social-science-data-editors.github.io/guidance/)

## Grade

Your grade will be 60% homeworks, 40% term project.

  

## Textbooks

There are some excellent references for computational methods out there. This course will use material from 

* **Fackler and Miranda** (2002), Applied Computational Economics and Finance, MIT Press
* **Kenneth Judd** (1998), Numerical Methods in Economics, MIT Press
* **Nocedal, Jorge, and Stephen J. Wright** (2006): Numerical Optimization, Springer-Verlag
* [**Kochenderfer and Wheeler**](https://mitpress.mit.edu/books/algorithms-optimization) (2019), Algorithms for Optimization, MIT Press
* [**A Gentle Introduction to Effective Computing in Quantitative Research**](https://mitpress.mit.edu/books/gentle-introduction-effective-computing-quantitative-research) - What Every Research Assistant Should Know, Harry J. Paarsch and Konstantin Golyaev


## Course Schedule

### 1. Programming languages and why `julia`

1. Talk through homework requirements
1. Talk through term project requirements
1. Show where material is and do first set of slides.
    


___

### 2. `julia` setup and Getting Started

* Setup environment
* Tools and Editors
* Examples
* Types
* Essentials
* Speed
* Data and Statistical Packages

___


### 3. Integration and Function Approximation

1. Numerical Integration
    * Monte-Carlo integration
    * Gaussian Quadrature
    * Multidimensional Quadrature
        * Quadrature with correlated shocks
2. Function Approximation
    * Polynomial Interpolation
        * Basis functions and Coefficients
    * Regression as Approximation
    * Colocation Methods
    * Multidimensional Approximation
        * The Smolyak Grid

___

### 4. Optimisation 1

1. Intro
2. Conditions for Optima
3. Derivatives and Gradients
4. Numerical Differentiation
5. JuliaOpt

___

### 5. Optimisation 2 

1. Bracketing
2. Local Descent
3. First/Second Order and Direct Methods
4. Constraints
5. JuMP.jl

___


### 6. Numerical Dynamic Programming

* Review of DP theory
* Different Solution methods for different cases
    * Discretization
    * Parametric approximation methods [basically Function Approximation](#Function-Approximation)
    * The Endogenous Grid Method
    * Finite time vs inifinite horizon models
+ Solving the Growth Model in 7 Different ways

___


### 7. Constrained Optimisation Applications as MPECs

* What is an MPEC?
* How can we cast constrained problems as MPECs?

Applications:

1. MPEC on John Rust's Bus Engine Replacement
1. The Berry-Levinsohn-Pakes (BLP) paper as constrainted optimization problems

___

   
### 8. HPC `julia`

* Brief intro to parallel computing concepts
* Parallel computing with julia
* GPU computing with julia


____

### 9. Rust Bus Model and Dynamic Discrete Choice


____

### 10. Intro to Machine Learning with julia


___

### 11. The julia ML stack


___

### 12. julia ML applications


___



## Statements on Plagiarism

We will try to honour [Science Po's anti-plagiarism policy](http://www.sciencespo.fr/welcome/en/content/plagiarism):

>Plagiarism occurs when a student submits work that does not allow one to distinguish the student's own thoughts from those of other authors: it can be characterised by the absence of citation of a group of consecutive words (five or more), by reformulation or translation, or by copying directly." (article on intellectual honesty)


### Related to Plagiarism

Reuse and building upon ideas or code are major parts of modern software development. As an economist writing code, you will (hopefully) never write anything from scratch. This class is structured such that all solutions are public. You are encouraged to learn from the work of your peers. As I said above, I won't hunt down people who are simply copying-and-pasting solutions, because without challenging themselves, they  are simply wasting their time and money taking this class.

Please respect the terms of use and/or license of any code you find, and if you reimplement or duplicate an algorithm or code from elsewhere, credit the original source with an inline comment.

## License

The copyright notice to be included in any copies and other derivative work of this material is:

```
Copyright 2021 Florian Oswald, Sciences Po Paris, florian.oswald@gmail.com
```

Thank you.

![](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png) This is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/)
