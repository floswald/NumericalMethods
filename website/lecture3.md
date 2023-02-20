~~~
<h1>Lecture 3 : Optimization 1</h1>
~~~


## Some Julia Package Stuff

> Let's create a julia package!
   
```julia
]
(@v1.8) pkg> add PkgTemplates

julia> using PkgTemplates

help?> Template

julia> t = Template(user = "floswald", interactive = true)
Template keywords to customize:
[press: d=done, a=all, n=none]
   [ ] authors
   [ ] dir
   [ ] host
   [ ] julia
 > [ ] plugins

julia> t("MyNewPackage")
[ Info: Running prehooks
[ Info: Running hooks
  Activating project at `~/.julia/dev/MyNewPackage`
    Updating registry at `~/.julia/registries/General`
    Updating git-repo `https://github.com/JuliaRegistries/General.git`
┌ Warning: The active manifest file at `/Users/74097/.julia/dev/MyNewPackage/Manifest.toml` has an old format that is being maintained.
│ To update to the new format, which is supported by Julia versions ≥ 1.6.2, run `Pkg.upgrade_manifest()` which will upgrade the format without re-resolving.
│ To then record the julia version re-resolve with `Pkg.resolve()` and if there are resolve conflicts consider `Pkg.update()`.
└ @ Pkg.Types /Users/julia/.julia/scratchspaces/a66863c6-20e8-4ff4-8a62-49f30b1f605e/agent-cache/default-macmini-x64-5.0/build/default-macmini-x64-5-0/julialang/julia-release-1-dot-8/usr/share/julia/stdlib/v1.8/Pkg/src/manifest.jl:287
  No Changes to `~/.julia/dev/MyNewPackage/Project.toml`
  No Changes to `~/.julia/dev/MyNewPackage/Manifest.toml`
Precompiling project...
  1 dependency successfully precompiled in 1 seconds
[ Info: We haven't cleaned this depot up for a bit, running Pkg.gc()...
      Active manifest files: 113 found
      Active artifact files: 303 found
      Active scratchspaces: 48 found
     Deleted 40 package installations (186.329 MiB)
  Activating project at `~/.julia/environments/v1.8`
[ Info: Running posthooks
[ Info: New package is at /Users/74097/.julia/dev/MyNewPackage
```

* Great! Now let's open VSCode in that location and make some changes.

```julia
module MyNewPackage

mutable struct MPoint
    x::Number
    y::Number
end

import Base.:+

+(a::MPoint,b::MPoint) = MPoint(a.x + b.x, a.y + b.y)

end   # module
```

* Now, we could just `execute active file in REPL` in VSCode, or indeed, type in the REPL `include("src/MyNewPackage.jl")`:

```julia
julia> include("src/MyNewPackage.jl")
Main.MyNewPackage
```

* Now we can test the code in the REPL. 
* Notice, **importantly**, each time you want to see the effects of changing your code, you have to **replace the module** via `include("src/MyNewPackage.jl")`.

```julia
julia> include("src/MyNewPackage.jl")
WARNING: replacing module MyNewPackage.
```

* Let's add a test for our `+` method in the `test` folder:

```julia
@testset "MyNewPackage.jl" begin
    a = MyNewPackage.MPoint(3,5)
    b = MyNewPackage.MPoint(1,2)
    @test a + b isa MyNewPackage.MPoint
    p = a + b
    @test p.x == a.x + b.x
    @test p.y == a.y + b.y
end
```

* Run the tests from REPL in Pkg mode: `]; test`
* This *works*, but it involves one extra step that we need to do manually. Small steps add up! So let's try to improve on that.
* Let's tell the package to add `Revise.jl`: `]; add Revise`. 
* Shut down and restart VScode in same folder.
* **Before** anything else, type `using Revise`. Then type `using MyNewPackage`.
* Let's add a new function now. 

```julia
-(a::MPoint,b::MPoint) = MPoint(a.x - b.x, a.y - b.y)
```

* *Don't* replace the module via `include("src/MyNewPackage.jl")`. Instead, just save the file and go back to REPL.
  
```julia
julia> a = MyNewPackage.MPoint(3,4)
MyNewPackage.MPoint(3, 4)

julia> b = MyNewPackage.MPoint(99,100)
MyNewPackage.MPoint(99, 100)

julia> a - b
MyNewPackage.MPoint(-96, -96)
```

* 🎉

### Debugging A Package

* Debugging simple scripts or packages is the same workflow. 
* Let's add another function. An _economic model_ of sorts:

```julia
function econ_model(; startval = 1.0)
    # make an Mpoint
    x = MPoint(startval, startval-0.5)
    # ... and evaluate a utility function
    MPoint(log(x.x),log(x.y))
end
```

* Make sure to try out that it works.

```julia
julia> MyNewPackage.econ_model()
MyNewPackage.MPoint(0.0, -0.6931471805599453)
```

* Ok great. Now what about that? Try it out!

```julia
julia> MyNewPackage.econ_model(startval = 0.3)
```

* Good. Let's pretend we don't know what's going on and we need to investigate this function.

1. Add `println` statements.
2. Add `@debug` statements. then attaching a logger with 

```julia
using Logging
debug_logger = ConsoleLogger(stdout, Logging.Debug)
global_logger(debug_logger)  # turns on logging of @debug messages
```

3. Use an actual debugger to step through our code.
   1. `VSCode` exports by default the `@enter` macro. type: `@enter MyNewPackage.econ_model(startval = -0.3)`
   2. click on teh play symbol. program hits an error. 
   3. set a break point just before
   4. click on `replay`.




## Some Julia-Bootcamp stuff

Topic | Notebook
:-----: | :--------:
Intro to Macros | [click for notebook](https://floswald.github.io/julia-bootcamp/10-intro-to-macros.html)
Intro to Differential Equations | [click for notebook](https://floswald.github.io/julia-bootcamp/08-popgrowth.html)
Plotting | [click for notebook](https://floswald.github.io/julia-bootcamp/06-plotting.html)
Interactive | [click for notebook](https://floswald.github.io/julia-bootcamp/07-interactive.html)



## Optimization, Finally!

Topic | Notebook
:-----: | :--------:
Review of Optimization Algorithms | [right-click and save notebook](https://github.com/floswald/NumericalMethods/blob/master/notebooks/week3/optimization1.jl)
