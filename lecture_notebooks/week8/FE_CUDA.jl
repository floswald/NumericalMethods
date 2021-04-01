# FixedEffectModels Benchmark with CUDA

using LinearAlgebra, Statistics
using BenchmarkTools
using DataFrames
using RegressionTables, FixedEffectModels


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


# create our dataset
df = FEbenchmark()

# run a bunch of different models
# the comments show timings of a previous version of FixedEffectModels (faster!)
@time reg(df, @formula(y ~ x1 + x2))
# 0.582029 seconds (852 allocations: 535.311 MiB, 18.28% gc time)
@time reg(df, @formula(y ~ x1 + x2), Vcov.cluster(:id2))
# 0.809652 seconds (1.55 k allocations: 649.821 MiB, 14.40% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
# 1.655732 seconds (1.21 k allocations: 734.353 MiB, 16.88% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)), Vcov.cluster(:id1))
#  2.113248 seconds (499.66 k allocations: 811.751 MiB, 15.08% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
# 3.553678 seconds (1.36 k allocations: 1005.101 MiB, 10.55% gc time))

# let's take the biggest model as our benchmark timing
@btime result = reg($df, @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + fe(id1) + fe(id2)))

# using R lfe package
using RCall
rstring = R"""
# factorize data
# notice that we have to pass the integer data in id1_int.
r_d = $df
r_d[,"id1"] = factor(r_d[,"id1_int"])
r_d[,"id2"] = factor(r_d[,"id2_int"])


library(lfe)
lfe_time = system.time(lfe <- felm(y ~x1 + x2 + x3 + x4 + x5 + x6 + x7| id1 + id2, data=r_d))
print(paste0("R lfe time: ",lfe_time[1]))
print(summary(lfe))

# also fixest
# how many threads?
threads = getFixest_nthreads()
print(paste0("fixest running on ",threads," threads"))
fixest_time = system.time(fe <- fixest::feols(y ~x1 + x2 + x3 + x4 + x5 + x6 + x7 | id1 + id2, data=r_d))
print(paste0("R fixest time: ",fixest_time[1]))

list(lfe = lfe_time[1], fixest = fixest_time[1])
#
## OLS.
#ols <- lm(y ~x1 + x2 + x3 + x4 + x5 + x6 + x7 + id1 + id2 -1, data=r_d)
#summary(ols)
"""


using CUDA

# run once to compile
result = reg(df, 
    @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + fe(id1) + fe(id2)),
    method = :GPU,
    double_precision = false)

@btime result = reg(df, 
    @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + fe(id1) + fe(id2)),
    method = :GPU,
    double_precision = false)
