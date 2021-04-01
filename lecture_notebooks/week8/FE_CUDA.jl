# FixedEffectModels Benchmark with CUDA

using LinearAlgebra, Statistics
using BenchmarkTools
using DataFrames
using FixedEffectModels
using RCall
using CUDA

"""
    FEbenchmark(;N=10_000_000,K= 100)

A function to create a 2-way fixed effect dataset. `N` observations, `K` 
and `N/K` categories for Fixed effects respectively. We generate 7 regressors as well as a weight vector.
"""
function FEbenchmark(;N=10_000_000,K= 100)
    id1_int = Int.(rand(1:(N/K), N))
    id2_int = Int.(rand(1:K, N))
    w = cos.(id1_int)
    
    x1 = 5 * cos.(id1_int) + 5 * sin.(id2_int) + randn(N)
    x2 =  cos.(id1_int) + sin.(id2_int) + randn(N)
    x3 =  cos.(id1_int) + sin.(id2_int) + randn(N)
    x4 =  cos.(id1_int) + sin.(id2_int) + randn(N)
    x5 =  cos.(id1_int) + sin.(id2_int) + randn(N)
    x6 =  cos.(id1_int) + sin.(id2_int) + randn(N)
    x7 =  cos.(id1_int) + sin.(id2_int) + randn(N)
    y= 3 .* x1 .+ 5 .* x2 .+ x3 .+ x4 .+ x5 .+ x6 .+ x7 .+ cos.(id1_int) .+ cos.(id2_int).^2 .+ randn(N)
    df = DataFrame(id1 = categorical(id1_int),id1_int = id1_int, 
                   id2 = categorical(id2_int), id2_int = id2_int,
                   x1 = x1, 
                   x2 = x2,
                   x3 = x3,
                   x4 = x4,
                   x5 = x5,
                   x6 = x6,
                   x7 = x7,        
                   w = w, y = y)
    df
end

# create our dataset
df = FEbenchmark()

# run once to compile
println("result = reg(df, @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + fe(id1) + fe(id2)))")
result = reg(df, @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + fe(id1) + fe(id2)))

# let's take the biggest model as our benchmark timing
println("FixedEffects.jl single CPU core time")
@time result = reg(df, @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + fe(id1) + fe(id2)));


println("\n\n\n\n We'll run the same data through R packages lfe and fixest now.\n\n\n")

rstring = R"""
# factorize data
# notice that we have to pass the integer data in id1_int.
r_d = $df  # this is the *same* memory in my computer! no copy taken.
r_d[,"id1"] = factor(r_d[,"id1_int"])
r_d[,"id2"] = factor(r_d[,"id2_int"])

library(lfe)
library(Matrix)
lfe_time = system.time(lfe <- felm(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7| id1 + id2, data=r_d))
print("1. Same result?")
print("")
print(summary(lfe))

print("")
print("2. lfe timing?")
print("")
print(paste0("R lfe time: ",lfe_time[1]))

# also fixest
# how many threads?
print("")
print("3. fixest timing?")
print("")
library(fixest)
threads = getFixest_nthreads()
print(paste0("fixest running on ",threads," threads"))
fixest_time = system.time(fe <- feols(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 | id1 + id2, data=r_d))
print(paste0("R fixest time: ",fixest_time[1]))

list(lfe = lfe_time[1], fixest = fixest_time[1])
#
## OLS.
#ols <- lm(y ~x1 + x2 + x3 + x4 + x5 + x6 + x7 + id1 + id2 -1, data=r_d)
#summary(ols)
"""

println("\n\n\n\n Now let's try on the GPU!\n\n\n")

CUDA.versioninfo()

# run once to compile
@show begin
    result = reg(df, 
    @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + fe(id1) + fe(id2)),
    method = :gpu,
    double_precision = false);
end

println("\n FixedEffects.jl GPU timing:\n")

@time result = reg(df, 
    @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + fe(id1) + fe(id2)),
    method = :gpu,
    double_precision = false);
