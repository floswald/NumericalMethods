### A Pluto.jl notebook ###
# v0.12.12

using Markdown
using InteractiveUtils

# ╔═╡ 2e356456-568a-11eb-13ca-a363a86b5b7a
begin
	using Dates  # for date functionality

	mutable struct Spell 
		start       :: Date
		stop        :: Date
		duration    :: Week
		firm        :: Int   # ∈ 1,2,...,L+1
		wage        :: Float64
		change_firm :: Bool   # switch firm after this spell?
		function Spell(t0::Date,fid::Int)  # this is the `inner constructor` method
			this = new()
			this.start = t0
			this.stop = t0
			this.duration = Week(0)
			this.firm = fid
			this.wage = 0.0
			this.change_firm = false
			return this 
		end
	end
end

# ╔═╡ 4dc17b06-5690-11eb-2c7f-4b631bef34de
sp = Spell(Date("2015-03-21"), 34)

# ╔═╡ 67e6fd94-5690-11eb-30ba-a9076b9ef44d
md"
ok, great. now we need a way to set some infos on this type. In particular, we want to record the wage the worker got, and how long the spell lasted. Here is function to call at the end of a spell:
"

# ╔═╡ a871e540-5690-11eb-3d43-155b9cdbb824
function finish!(s::Spell,w::Float64,d::Week)
    @assert d >= Week(0)
    s.stop = s.start + d
    s.duration = d
    s.wage = w
end		

# ╔═╡ 0b169d58-5691-11eb-1267-1514234dffe4
md"
let's say that particular spell lasted for 14 weeks and was characterised by a wage of 100.1 Euros
"

# ╔═╡ cc8a1272-5690-11eb-14dc-0992724903eb
finish!(sp, 100.1, Week(14))

# ╔═╡ 00b1d5be-5691-11eb-296a-ad58977711cb
sp

# ╔═╡ Cell order:
# ╠═2e356456-568a-11eb-13ca-a363a86b5b7a
# ╠═4dc17b06-5690-11eb-2c7f-4b631bef34de
# ╠═67e6fd94-5690-11eb-30ba-a9076b9ef44d
# ╠═a871e540-5690-11eb-3d43-155b9cdbb824
# ╠═0b169d58-5691-11eb-1267-1514234dffe4
# ╠═cc8a1272-5690-11eb-14dc-0992724903eb
# ╠═00b1d5be-5691-11eb-296a-ad58977711cb
