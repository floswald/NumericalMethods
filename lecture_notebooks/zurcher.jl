
module Rust

    using LinearAlgebra  # for norm()
    using Plots

    function make_trans(θ,n)
        transition = zeros(n,n);
        # ensure params sum to 1.0
        pr = θ ./ sum(θ)

        for i=0:n-4
            transition[i+1,(i+1):(i+4)] = pr[1:4]
        end
        transition[n-2,(n-2):n] = [pr[1], pr[2] , 1-(pr[1]+pr[2])];
        transition[n-1,(n-1):n] = [pr[1], 1-pr[1]];
        transition[n,n] = 1.0
        return transition
    end

    Base.@kwdef struct Zurcher
        # parameters
        n::Int = 175
        RC::Float64 = 11.7257
        c::Float64 = 2.45569
        θ::Vector{Float64} = [0.0937,0.4475,0.4459,0.0127]
        β::Float64 = 0.9

        # numerical settings
        niter::Int = 1000
        tol::Float64 = 1e-6

        # state space
        mileage::Vector{Float64} = collect(0.0:n-1)
        transition :: Matrix{Float64} = make_trans(θ,n)
    end

    function bellman(z::Zurcher, ev0::Vector)
        maintainance = -0.001 .* z.mileage .* z.c  # StepRange of length n
        v0 = maintainance .+ z.β .* ev0  # vector of length n
        v1 = -z.RC + maintainance[1] + z.β * ev0[1]  # a scalar. if you replace, you start at zero miles
        M  = maximum(vcat(v1,v0))  # largest value in both vs
        logsum = M .+ log.(exp.(v0 .- M) .+ exp(v1 - M))
        ev1 = z.transition * logsum # matrix multiplication
        # ev1 = logsum # matrix multiplication
        ev1
    end

    function ccp(z::Zurcher, ev::Vector)
        maintainance = -0.001 .* z.mileage .* z.c  # StepRange of length n
        v0 = maintainance .+ z.β .* ev  # vector of length n
        v1 = -z.RC + maintainance[1] + z.β * ev[1] 
        1.0 ./ (exp.(v0 .- v1) .+ 1)
    end

    function vfi(z::Zurcher)
        ev0 = zeros(z.n)  # starting value
        for it in 1:z.niter

            ev1 = bellman(z,ev0)
            err = norm(abs.(ev0 .- ev1))

            if err < z.tol
                break   # break for loop
            elseif it==z.niter
                error("no convergence after $it iterations")
            end
            # update
            ev0[:] .= ev1   # [:] do not reallocate a new object
        end
        return ev0
    end

    function plotit()
        z = Zurcher()
        sol = vfi(z)
        pr  = ccp(z,sol)
        plot(plot(z.mileage, sol, title = "Value Function"), 
             plot(z.mileage, pr,  title = "Probability of Replacement"),
             xlab = "Miles", leg = false)
    end

    function simit(; T = 500)
        z = Zurcher(n = 1750)  # need go higher with miles
        sol = vfi(z)
        pr  = ccp(z,sol)

        P = cumsum(z.θ ./ sum(z.θ))

        miles = Int[]
        push!(miles, 1)  # start on first grid point of mileage
        a = Int[]  # 0/1 keep or replace
        push!(a,0)  # keep in first period

        for it in 2:T
            action = rand() < pr[miles[end]] ? 1 : 0
            push!(a, action)
            # update miles
            if action == 1
                push!(miles, 1)  # go back to first state
            else
                next_miles = findfirst(rand() .< P)  # index of first `true`
                push!(miles, miles[end] + next_miles)
            end
        end

        plot(1:T, miles, xlab = "Period", ylab = "miles",
            title = "Simulating Harold Zurcher's Decisions",
            leg = false, lw = 2)
    end
end


