### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ f77eb3dc-7294-11eb-0f0a-93873d07bcee
using Images, QuartzImageIO

# ╔═╡ cee9426c-55b5-11eb-2e5d-73534f65e6f6
md"
# Intro to abstractions

This is following in the footsteps of Prof Alan Edelman's example.

**Question**: What's an _array_ again, and what can we do with it?
"

# ╔═╡ fef90054-7294-11eb-2926-55d966c3fecc


# ╔═╡ 4cbc4f40-55b6-11eb-1297-abf3f0e48d44
element = load(download("https://www.skiarlberg.at/uploads/media/1440x730/01/341-Freeriden%20St.%20Anton%20am%20Arlberg%202.jpg?v=1-0?v=202011041516"))

# ╔═╡ 719a66a8-55b6-11eb-25ce-d7c7dd0f9ff7
a = fill(element, 2, 3)

# ╔═╡ 5f07e22e-7295-11eb-30f8-e525fdab8e0f
a'

# ╔═╡ 6b9d93dc-7295-11eb-1936-291d159f35db
inv(a[1,1])

# ╔═╡ 9f8caba2-7294-11eb-1bdc-1928a5570fc5
tracking_element = []

# ╔═╡ a8c7526c-7294-11eb-0f24-4d3c4586a1aa
push!(tracking_element, element)

# ╔═╡ b7eebfdc-7294-11eb-1716-7dd39b6cfd1f
types = DataType[]

# ╔═╡ bde36000-7294-11eb-30a2-5f589d63f473
push!(types, eltype(a))

# ╔═╡ 2f25591a-7295-11eb-008c-e54d6f423e89
element[1,1]

# ╔═╡ Cell order:
# ╟─cee9426c-55b5-11eb-2e5d-73534f65e6f6
# ╠═f77eb3dc-7294-11eb-0f0a-93873d07bcee
# ╠═fef90054-7294-11eb-2926-55d966c3fecc
# ╠═4cbc4f40-55b6-11eb-1297-abf3f0e48d44
# ╠═719a66a8-55b6-11eb-25ce-d7c7dd0f9ff7
# ╠═5f07e22e-7295-11eb-30f8-e525fdab8e0f
# ╠═6b9d93dc-7295-11eb-1936-291d159f35db
# ╠═9f8caba2-7294-11eb-1bdc-1928a5570fc5
# ╠═a8c7526c-7294-11eb-0f24-4d3c4586a1aa
# ╠═b7eebfdc-7294-11eb-1716-7dd39b6cfd1f
# ╠═bde36000-7294-11eb-30a2-5f589d63f473
# ╠═2f25591a-7295-11eb-008c-e54d6f423e89
