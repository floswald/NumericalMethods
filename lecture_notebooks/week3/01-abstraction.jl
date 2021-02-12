### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 0f8cd702-6d12-11eb-2fb4-53838bcc279f
using Images,QuartzImageIO

# ╔═╡ cee9426c-55b5-11eb-2e5d-73534f65e6f6
md"
# Intro to abstractions

This is following in the footsteps of Prof Alan Edelman's example.

**Question**: What's an _array_ again, and what can we do with it?
"

# ╔═╡ 4cbc4f40-55b6-11eb-1297-abf3f0e48d44
element = load(download("https://www.skiarlberg.at/uploads/media/1440x730/01/341-Freeriden%20St.%20Anton%20am%20Arlberg%202.jpg?v=1-0?v=202011041516"))

# ╔═╡ 559abad4-55b6-11eb-02de-f30062b2dede
a = fill(element,2,3)

# ╔═╡ bdafcfba-55b6-11eb-099a-a149ffb1d97e
tracking_elements = []

# ╔═╡ c9af6744-55b6-11eb-2ccc-4df7a533b569
push!(tracking_elements, element)

# ╔═╡ 5d2be854-55b6-11eb-33b4-0dfdc5dda1fa
types = DataType[]

# ╔═╡ 786977c6-55b6-11eb-19c6-01a9c02902a3
push!(types,eltype(a))

# ╔═╡ 719a66a8-55b6-11eb-25ce-d7c7dd0f9ff7


# ╔═╡ Cell order:
# ╟─cee9426c-55b5-11eb-2e5d-73534f65e6f6
# ╠═0f8cd702-6d12-11eb-2fb4-53838bcc279f
# ╠═4cbc4f40-55b6-11eb-1297-abf3f0e48d44
# ╠═559abad4-55b6-11eb-02de-f30062b2dede
# ╠═bdafcfba-55b6-11eb-099a-a149ffb1d97e
# ╠═c9af6744-55b6-11eb-2ccc-4df7a533b569
# ╠═5d2be854-55b6-11eb-33b4-0dfdc5dda1fa
# ╠═786977c6-55b6-11eb-19c6-01a9c02902a3
# ╠═719a66a8-55b6-11eb-25ce-d7c7dd0f9ff7
