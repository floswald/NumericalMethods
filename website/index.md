
# Computational Economics for PhDs


* **Teacher:** Florian Oswald, [florian.oswald@sciencespo.fr](mailto:florian.oswald@sciencespo.fr)
* **Class Times:** Tuesdays 17:00
* **Class Location:** Paris 😄
* **Slack**: There will be a slack channel for all communication

@def mintoclevel=2 
@def maxtoclevel=3

~~~
<br>
~~~

---

**Table of Contents:**

\toc

---

## Course Overview

This is a course for PhD students at SciencesPo Paris in Computational Economics. You will learn about some commonly used methods in Computational Economics. These methods are being used in *all* fields of Economics. The course has a clear focus on applying what you learn. We will cover the theoretical concepts that underlie each topic, but you should expect a fair amount of *hands on* action required on your behalf. In the words of the great [Che-Lin Su](https://scholar.google.com/citations?user=6AZSMiwAAAAJ&hl=en):

> Doing Computation is the only way to learn Computation. Doing Computation is the only way to learn Computation. Doing Computation is the only way to learn Computation.

True to that motto, there will be homeworks for you to try out what you learned in class. There will also be a term paper.

## Prerequisites

1. You need a laptop. 
2. You should be familiar with the material from [Introduction to Programming](https://floswald.github.io/ScPoProgramming).
3. You must sign up for a free account at github.com. Choose a reasonable user name and upload a profile picture.
4. **Before** you come the first class, please do this:
    1. Download the latest [stable `julia` release](https://julialang.org/downloads/) for your OS.
    2. Download the [`VSCode Editor`](https://code.visualstudio.com)

### Getting Programming Skills

1. Check out what is being taught in the [Introduction to Programming](https://floswald.github.io/ScPoProgramming) course. You must know this level.
1. We will be using [Julia](http://julialang.org/) for this course. 
    - [Noteworthy Differences from Other Languages](https://docs.julialang.org/en/v1/manual/noteworthy-differences/)
    - [MATLAB, Python, Julia Syntax Comparison](http://cheatsheets.quantecon.org/)


## Term Project

There are two options:

1. Replicate a published paper.
2. Develop the computational aspects of your own work.

### Replication

The requirements for choice of paper to replicate are:

1. It's an economics paper.
1. Published version and replication kit is available online.
2. The paper to replicate must not use julia.
3. You must use julia for your replication.
    * Ideally your choice will involve at least some level of computational interest (i.e. more than an IV regression)
    * However, you can replicate a paper with an IV regression, but you have to go all the way to get the exact same results as in the paper. I.e. if the author typed the stata command `ivreg2 lw s expr tenure rns smsa _I* (iq=med kww age), cluster(year)` you will have to write (or find) julia code which will match all output from this, including standard errors.
4. You need to set up a public github repository where you will build a documentation website of your implementation. You'll learn how to do this in the course.
5. I encourage you to let the world know about your replication effort via social media and/or email to the authors directly. This is independent of whether you were able or not to replicate the results. Replication is not about finding errors in other peoples' work. If you are able to replicate some result in julia, this may be very interesting for others.

> Please use the starter kit available at [https://github.com/floswald/Replicate.jl](https://github.com/floswald/Replicate.jl) to get going!

#### Replication Resources

* [here is a great list by the AEA](https://www.aeaweb.org/rfe/showCat.php?cat_id=9)
* [ECTA code and data](https://www.econometricsociety.org/publications/econometrica/journal-materials/supplemental-materials)
* [RevEconDynamics codes](https://ideas.repec.org/s/red/ccodes.html)
* Each issue of RevEconDynamics , e.g. [https://www.economicdynamics.org/volume-39-2021/](https://www.economicdynamics.org/volume-39-2021/)
* [The AEA Data Editor's website](https://aeadataeditor.github.io/talks/)
* [The Restud Data Editor](https://restud.github.io/data-editor/replicate/#replicate-a-paper) and their [zenodo repo of replication kits](https://zenodo.org/communities/restud-replication/?page=1&size=20)
* [The Social Science Data Editor's joint website](https://social-science-data-editors.github.io/guidance/)


### Develop Your Own Work

You can develop your own work as well. Requirements:

1. setup a github repository which contains the code (your decision whether public or private, in any case you have to share it with me)
1. produce a short document (max 10 pages, ideally much less) which describes
    1. the aim of the project
    1. the computational problem
    1. your computational strategy to solve that problem
1. The main focus for me will lie on 
    1. How easy is it to use your code?
    1. How easy is it to understand your code (code readability and provided documentation)?
    1. Did you provide unit tests? Can I be convinced that your code does what it is supposed to do?

## Grade

Your grade will be 60% homeworks, 40% term project.

  

## Textbooks

There are some excellent references for computational methods out there. This course will use material from 

### The Classics

* **Fackler and Miranda** (2002), Applied Computational Economics and Finance, MIT Press
* **Kenneth Judd** (1998), Numerical Methods in Economics, MIT Press
* **Nocedal, Jorge, and Stephen J. Wright** (2006): Numerical Optimization, Springer-Verlag

### Newcomers

* [**Julia for Data Analysis**](https://www.manning.com/books/julia-for-data-analysis) (2023), Bogumił Kamiński, Manning Publications.
* [**Algorithms for Optimization**](https://mitpress.mit.edu/books/algorithms-optimization) (2019), Mykel J. Kochenderfer and Tim A. Wheeler, Algorithms for Optimization, MIT Press.
* [**A Gentle Introduction to Effective Computing in Quantitative Research**](https://mitpress.mit.edu/books/gentle-introduction-effective-computing-quantitative-research) - What Every Research Assistant Should Know, Harry J. Paarsch and Konstantin Golyaev
* [**Statistics with Julia**](https://statisticswithjulia.org) (2021), Yoni Nazarathy and Hayden Klok, Springer.
* [**Quantitative Economics with Julia**](https://julia.quantecon.org/intro.html) by Perla, Sargent and Stachurski is a wonderful resource and we use it a lot in this course. 



## License

The copyright notice to be included in any copies and other derivative work of this material is:

```
Copyright 2023 Florian Oswald, Sciences Po Paris, florian.oswald@gmail.com
```

Thank you.

![](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png) This is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/)
