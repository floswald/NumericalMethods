
~~~
<h1>Getting Started with Julia</h1>
~~~

In this lecture we will start talking `julia`! We will go through a series of notebooks in tutorial style. Here is again a table with pre-rendered notebooks for you to look at:

Topic | Notebook
:-----: | :--------:
Defining Data Types | [click for notebook](https://floswald.github.io/julia-bootcamp/04-defining-types.html)
Intro do Differential Equations | [click for notebook](https://floswald.github.io/julia-bootcamp/08-popgrowth.html)

**Table of Contents**

\toc


## Arrays

## VScode and Julia

I asked you to install the [julia-vscode](https://www.julia-vscode.org) extention for the vscode editor [initially](../#prerequisites). Today we'll have a first look at this environment. Some things we'll go over are:

1. How to start a julia REPL
1. How to run code line by line
1. How to run a file
1. Package environments
1. More advanced features (already couple of months back, but still) are explained in this [video by one of the package authors](https://youtu.be/IdhnP00Y1Ks).

## Julia Modules, Packages, `Pkg.jl` and Environments

1. What is a `module`?
    * A separate name space.
    * We can group together functions from different files using the `include`.
    * Notice there is no relationship between function names and file names, like for instance in matlab or python.
    * you decide what is `export`ed from the package, if anything, with the `export` keyword.

```julia
module MyExampleModule
using Statistics   # if any imports
include("files1.jl")
include("functions2.jl")
include("data.jl")
export xyz_fun   # if any exports
end # module
```

2. What's a **package**?
    * Basically, a `module`. 
    * Why do we need packages?
    * Here an [Example](https://github.com/JuliaLang/Example.jl)
2. The problem with user-contributed packages and open source software is that there is no central authority imposing a strict versioning. 
    * [Why not? 👉 Because thou shall not force volunteers to do stuff ([Benabou and Tirole](https://academic.oup.com/restud/article-abstract/70/3/489/1571401)).] 
    * For example, my package could depend on a feature of your package. 
    * You decide to remove that feature.
    * My package breaks.
    * Now what?
3. What do other open source languages do here?
    * `R`: nothing. You install whichever version you want and hope it all works. Latest versions is a good idea.
    * `python`: versioning problem comes even before packages! Python 2.7 incompativel with python 3.6. So we have *virtual environments* for different python versions
        * It can get complicated [really easily](https://github.com/econ-ark/KrusellSmith/issues/3).
4. What is an _environment_
    * It's like a box inside which a certain set of versions can be assumed.
    * It's great for reproducibility: you know exactly which set of versions produced which set of results.
    * A docker container is an example of an environment in a remote data center, where you specify what kind of machine you want and what software should be available.
5. julia: [`Pkg.jl` package manager](https://julialang.github.io/Pkg.jl/v1/). 
    * Environments are specified in a text file `Project.toml` at the root of a given package or folder.
    * It specifies the required packages to run the current *project* (or whatever code you have in the current folder)
    * You can see an example [here](https://github.com/floswald/NumericalMethods/tree/master/lecture_notebooks/week2/Project.toml)
    * A second file called `Manifest.toml` specifies the *exact* list of dependencies that are implied by the packages you rely on in `Project.toml`. 
    * Including both `Project.toml` and `Manifest.toml` in a folder provides an **exact** snapshot of the code needed to reproduce the results.
    * It's a major step ahead to improve reproducibility of results.
6. Demonstrate how to activate a package in REPL.
7. Show some `]` package manager commands and `help`


## Notebook Formats

You will notice that there are two different notebook formats in our course: the new Pluto.jl format, and the more traditional jupyter notebook. Here's a quick video of how to launch a jupyter notebook server on your computer to run those notebooks - on the example of our course content.

{{youtube jupyter-notebooks}}