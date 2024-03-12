### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 501fe9a7-d1a0-43b3-be2a-474ed3ac57f6
md"""
# Solving Systems of Equations

* In economics are models are often defined by a set of equations which most hold at a solution.
* For example, the Euler Equation in consumption-savings problems, or any other optimality condition.
* In this notebook we look at approaches to solve those.
"""

# ╔═╡ 7f6f5e08-6cc9-495f-9bb3-dd2a98db6a93


# ╔═╡ d365b3c0-4470-11ec-03de-b72cbdf34085

begin
danger(head,text) = Markdown.MD(Markdown.Admonition("danger", head, [text]));
danger(text) = Markdown.MD(Markdown.Admonition("danger", "Attention", [text]));
info(text) = Markdown.MD(Markdown.Admonition("info", "Info", [text]));
tip(text) = Markdown.MD(Markdown.Admonition("tip", "Tip", [text]));
end

# ╔═╡ Cell order:
# ╟─501fe9a7-d1a0-43b3-be2a-474ed3ac57f6
# ╠═7f6f5e08-6cc9-495f-9bb3-dd2a98db6a93
# ╠═d365b3c0-4470-11ec-03de-b72cbdf34085
