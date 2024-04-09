### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 048bed3a-e7aa-4866-ac4f-d8990bc1b844
using LinearAlgebra

# ╔═╡ 2101bfaf-33bd-42a5-a836-c0c253b253e8
using Plots

# ╔═╡ 76706c29-fcec-4475-a766-fa32b0434a5c
using Optim

# ╔═╡ 5a3b056e-27fd-4828-a573-9a1e95a546eb
using PlutoUI,  PlutoTeachingTools

# ╔═╡ dc3a7c1b-512c-49c4-a96d-838556e1d9c2
md"""# To get started:
1. Install julia and pluto.jl following [https://plutojl.org](https://plutojl.org)
2. Get the notebook from [https://github.com/nathanaelbosch/probnumspringschool2024-tutorial](https://github.com/nathanaelbosch/probnumspringschool2024-tutorial)
"""

# ╔═╡ e58eea83-277b-40c7-ac9d-5f3300e59ffe
md"""
# Practical Session: Probabilistic Numerics for ODEs
Author: [Nathanael Bosch](https://nathanaelbosch.github.io)
"""

# ╔═╡ 799233c0-97b9-4620-8356-16c41239a92b
md"""
## **Overview**

In this tutorial you will **implement your own ODE filter, completely from scratch!** 
The tutorial is structured as follows:
"""

# ╔═╡ 9593dfa0-821b-45b2-ba23-03ba29f3eb79
md"""
# 0. How to code in Julia and use Pluto.jl

You don't need any prior Julia experience to work with this notebook! We will write very simple code, and we will only use quite basic things that we can quickly cover here.
"""

# ╔═╡ 369bea9e-994f-4168-b6bf-ea16c60734bb
md"### Basic Julia"

# ╔═╡ 0ea5e615-e988-4f6d-bf72-66a9fed1b809
md"We can define variables as usual"

# ╔═╡ 5b24cacf-921b-4b73-b6f5-355fdf9f8adb
variablename = 1000

# ╔═╡ 11204fa2-4594-4244-a9ac-b600450014e6
md"In comparison to Jupyter notebooks, Pluto.jl prints the output _above_ the code cell."

# ╔═╡ e97cf1b4-b254-4c12-b014-ef6f80362a5f
md"Math is mostly aready built-in and we don't need any `numpy` equivalent to work with arrays:"

# ╔═╡ 3848eed0-8aca-4aaa-9d82-011f0977bc29
vector = [1, 2]

# ╔═╡ cec7a2ee-d308-49aa-93dc-0f5584c4c874
matrix = [1 2; 3 4]

# ╔═╡ a0653a41-87de-4c97-9812-f1112fbfb195
size(vector)

# ╔═╡ eba65e7a-5b38-49bc-b81b-e878cc0d33bd
size(matrix)

# ╔═╡ da6dcaf7-f8eb-4dc9-a1d4-79b1d86007e9
md"`*` does a matrix vector multiplication:"

# ╔═╡ 992e7173-f70e-41a1-9588-fb67b56a84fe
matrix * vector

# ╔═╡ 5544ded8-904e-4e88-8038-bc1e26035279
md"But we can also broadcast operations to get a numpy-like behaviour of `*`:"

# ╔═╡ 72430f2d-55d2-4c65-bc99-f808fb153451
2 * vector

# ╔═╡ e41d5a46-48ef-4891-87f9-b6909ebbdbe1
vector' * matrix

# ╔═╡ 0c5021ff-7bef-4863-90c5-55a507451299
md"`'` transposes (or actually computes the adjoint, but with real numbers it's the same):"

# ╔═╡ 4bf13155-904d-4f5e-842e-8fa7f18bb807
matrix', vector'

# ╔═╡ 0add1185-f307-4dfe-8114-4280d5622fde
md"We can copy vectors with `copy`:"

# ╔═╡ e607ba98-bc21-4b04-ab55-d41ecff19182
vector2 = copy(vector)

# ╔═╡ 95eb44df-af71-488b-9d97-cafe1294a02a
md"and add elements to a vector with `push!`:"

# ╔═╡ d7e85fa4-ad26-4c19-b86d-8c8346d5727b
push!(vector2, 10)

# ╔═╡ 6a087440-6077-4773-a01f-d39dfbc34874
md"Note that this mutates `vector2`."

# ╔═╡ 2a211405-4d7d-478d-bd2e-b53719817df7
md"Defining functions:"

# ╔═╡ e11629dc-1bf3-4dbe-977b-b8ca567dc6df
function myfunction(arg1, arg2=2; kwarg1, kwarg2=4)
	
	return arg1+arg2+kwarg1+kwarg2

end

# ╔═╡ 51dd874e-8a3e-4eed-97f2-148ad5a82efe
myfunction(1, 1; kwarg1=1)

# ╔═╡ 25023dba-be97-4cfb-a880-2be8c6e3330d
md"`LinearAlgebra.jl` exports some handy linear algebra things, like identity matrices or symmetric matrices, Kronecker products, and more:"

# ╔═╡ 572111f8-92b1-45c6-868c-8abaa163a92a
I(3)

# ╔═╡ a73a21d0-6c09-4815-b92d-319b38be9292
Matrix(I(3))

# ╔═╡ fdc4f920-a35e-4fed-8012-bd7e1bf624ac
Symmetric([1 2; 2 1])

# ╔═╡ f9773354-6b28-4fd8-9a20-c31803785354
md"### Pluto.jl specific things"

# ╔═╡ df25ff79-d301-4ff7-97c2-eeff64eb6873
md"The cool part: Pluto is completely reactive! Try changing the definition of `myfunction` above and see how the cell below reacts immediately."

# ╔═╡ 534ee2de-afe5-4a03-b914-9619a3594e47
md"The somewhat inconvenent part: Pluto cells always contain a single expression, so you can't just write multiple lines into a cell. So the following doesn't work:"

# ╔═╡ 233cc2c2-cf5d-472f-a690-76cd7b5c4e12
begin
	variable1 = 1
	variable2 = 2
end

# ╔═╡ 365dfbf2-9288-49ab-b702-da010b078d47
md"But Pluto tells you how to fix this: Either split the cell into two cells (after which each cell contains only a single expression), or wrap the code into a `begin ... end` block to effectively turn it into one expression. Try it out!"

# ╔═╡ e00e8289-999d-4209-bef0-2e9318148fe2
md"And finally, since Pluto notebooks are completely reactive you can't re-define variables that you already used:"

# ╔═╡ 8dd03abf-05ba-47ed-93f3-c3b5e0dee217
# ╠═╡ disabled = true
#=╠═╡
variablename = 10
  ╠═╡ =#

# ╔═╡ f8fab074-a2be-4218-8e0f-f85535c1764e
md"So we will need to make sure that we're always using a new variable name."

# ╔═╡ a1bc57e5-d3a7-4ee0-ac6a-204ea4ddb2ff
md"### Plotting in Julia with Plots.jl"

# ╔═╡ d8551bc9-6139-42f5-ab2c-95ad47163a50
begin
	x = range(1, 10)
	y = rand(10, 2)

	plot(x, y, xlabel="x", ylabel="y(x)", label=["y1" "y2"])
end

# ╔═╡ f35114c9-a5a2-4754-a4f1-b25f32842c73
md"Now let's get started with the actual notebook!"

# ╔═╡ bdb5dddf-1435-4537-aa08-ba5dd35b301d
begin
problem_setting_link = "#" * (PlutoRunner.currently_running_cell_id[] |> string)
md"""
# 1. Problem Setting: Ordinary Differential Equations (ODEs)

We are interested to solve ODEs of the form

$$\begin{aligned}
\dot{y}(t) &= f(y(t), t), \qquad t \in [t_0, t_\text{max}], \\ 
y(t_0) &= y_0,
\end{aligned}$$

where $f: \mathbb{R}^n \times [t_0, t_\text{max}] \to \mathbb{R}^n$ is the vector field, $[t_0, t_\text{max}]$ is the time interval, and $y_0 \in \mathbb{R}^n$ is the initial condition.  
**Our goal is to find $y : [t_0, t_\text{max}] \to \mathbb{R}^n$ which satisfies the ODE and the initial condition. This is what we call the _solution_ of the ODE.**
"""
end

# ╔═╡ f2fa37c4-488f-4dd8-8987-98f12436550a
md"""
### The concrete ODE: An epidemeological dynamical system

The concrete dynamical system that we will consider in this tutorial is an epidemeological [SIRD model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIRD_model), the _Susceptibe-Infectious-Recovered-Deceased model_, which describes the evolution of an infectious disease in a population.
It partitions the population into a discrete set of compartments and describes the transitions between these compartments with a set of ordinary differential equations:

$$\begin{aligned}
\frac{dS(t)}{dt} &= -\beta S(t) I(t) \\
\frac{dI(t)}{dt} &= \beta S(t) I(t) - \gamma I(t) - \eta I(t) \\
\frac{dR(t)}{dt} &= \gamma I(t) \\
\frac{dD(t)}{dt} &= \eta I(t)
\end{aligned}$$

where $S$, $I$, $R$, and $D$ are the fractions of the population that are susceptible, infected, recovered, and deceased, respectively, 
and $\beta$, $\gamma$, and $\eta$ are the infection, recovery, and death rates, respectively.
In this example, we consider initial conditions $S_0 = 0.99$, $I_0 = 0.01$, $R_0 = 0$, and $D_0 = 0$, parameters $\beta = 0.5$, $\gamma = 0.06$, and $\eta = 0.002$, and a time interval $[t_0, t_\text{max}] = [0, 100]$.

"""

# ╔═╡ 7b0e453c-e746-4995-84bf-282a65ee8d78
md"""
**Here is this SIRD problem *in code*:**
"""

# ╔═╡ 436ee1d8-d9c1-4b49-a171-a806954cea80
function f(y, p, t)
    S, I, R, D = y
    beta, gamma, eta = p
    
	dy = [
		-beta * S * I,
    	beta * S * I - gamma * I - eta * I,
        gamma * I,
        eta * I,
	]
	
	return dy
end

# ╔═╡ 3348963c-4aa9-4d29-86ff-ebeb0351e3be
y0 = [0.99, 0.01, 0.0, 0.0]

# ╔═╡ 2d4fb782-4cd4-4ab2-a28a-b6b0aff40622
tspan = (0.0, 100.0)

# ╔═╡ 8ccb9b17-7a9a-4562-a217-1d0c462d62a0
p = (beta, gamma, eta) = (0.5, 0.06, 0.002)

# ╔═╡ 4cf6b346-bdbe-4095-830a-080e64d9e307
labels = ["S(t)" "I(t)" "R(t)" "D(t)"] # used later for plotting

# ╔═╡ bec9cd97-182b-411d-ba08-3072dccf0c4f
forward_euler_link = "#" * (PlutoRunner.currently_running_cell_id[] |> string); md"""
# 2. Solve the ODE with forward Euler

To warm up and get more familiar with ODEs, let's first solve the ODE with a simple ODE solver, namely [forward Euler](https://en.wikipedia.org/wiki/Euler_method).
It works as follows:
1. Decide on a discrete time grid $t_0, t_1, \dots, t_K$. Typically: $t_k = t_0 + k \Delta t$ for some step size $\Delta t > 0$.
2. Initialize the solution $y(t_0)=y_0$.
3. Iteratively compute the solution at the next time step $y(t_{k+1})$ from the solution at the current time step $y(t_k)$:

$$y(t_{k+1}) = y(t_k) + \Delta t_k f(y(t_k), t_k).$$

!!! warning "Task"
    Implement forward Euler in the function `forward_euler` below.
"""

# ╔═╡ c7c73c79-757b-4cb0-bca7-ff92a27d3aa3
function forward_euler(f, y0, ts; p=p)
    ys = [y0] # output: A list of y values

	# TODO
	yk = y0
	for k in range(1, length(ts)-1)
		dt = ts[k+1] - ts[k]
		yk = yk + dt * f(yk, p, ts[k])

		push!(ys, yk)
	end
	
    return ys
end

# ╔═╡ 612a5c94-0969-4f3c-8c90-fa14e0624eb2
let
	ts = [0.0, 0.1, 0.2]
	out = forward_euler(f, y0, ts)
	if ismissing(out) || out == [y0]
		still_missing(md"Implement the function above and return the numerical solution.")
	elseif length(out) != length(ts)
		keep_working(md"The output should be a vector of the same length as the input.")
	else
		correct(md"Seems reasonable! Let's try it out.")
	end
end

# ╔═╡ 15dcaa60-bb28-489e-9c8d-7edb1f62d904
dt_small = 0.001

# ╔═╡ b014b770-422c-45b7-bfea-6f0c8cb744e7
times = range(tspan[1], tspan[2], step=dt_small)

# ╔═╡ e316d446-f64f-4edc-80b0-5be5ee656ece
ys = forward_euler(f, y0, times)

# ╔═╡ a0440b5d-3b33-49b6-b132-dc1e8f02e66c
if ys != [y0] #assuming you have implemented something above, let's plot
	plot(times, stack(ys)', label=labels, xlabel="t", ylabel="y(t)")
	# `stack(ys)'` turns the vector into a N x d matrix which `plot` handles well
end

# ╔═╡ 8527e16e-8641-499a-bdf6-f3d87b0241b0
md"Nice! Let's get a better feel for forward Euler by playing around with the step size:"

# ╔═╡ 519d00c3-ed43-40e1-a360-e95214552ade
md"dt = $(@bind dt_slider Slider(10 .^ (-2:1//10:1), show_value=true))"

# ╔═╡ e8a99120-0e7f-43b4-bf9c-4e20b36ba291
begin
	times_coarse = tspan[1]:dt_slider:tspan[2]	
	ys_coarse = forward_euler(f, y0, times_coarse)
	@assert length(ys_coarse) > 1 "Implementation missing above"
	plot(xlabel="t", ylabel="y(t)")
	plot(times, stack(ys)', label=map(l->l*" true", labels), color="black", linestyle=:dash)
	plot!(times_coarse, stack(ys_coarse)', label=map(l -> l * " Euler", labels))
end

# ╔═╡ 1a54e048-2153-4755-9a96-9b53ca10363a
md"""
If we make the steps large enough we can definitely see the numerical error. 

Note that the black dashed "accurate" solution also has error of course: it too has been computed by a numerical algorithm. 
For this notebook we will consider this not-exact-but-quite-accurate solution as the reference solution, which we compare all other coarser solutions to. 
The following function will come in handy later. 
"""

# ╔═╡ 9d5d91b8-cd68-442d-82ec-b8571e8ab50c
function reference_solution(times)
	dense_grid = vcat(tspan[1]:0.001:tspan[2], times) |> unique |> sort
	ref_ys_all = forward_euler(f, y0, dense_grid)
    ref_ys_attimes = ref_ys_all[[t in times for t in dense_grid]]
	return ref_ys_attimes
end

# ╔═╡ dca16cfc-1914-4eea-9356-a910b01fa299
md"Let's visualize the numerical errors."

# ╔═╡ 6b4b0659-026f-46d3-be2f-2617570506a5
let
	@assert length(ys_coarse) > 1 "Implementation missing above"
	ys_ref = reference_solution(times_coarse)
	errs = ys_ref .- ys_coarse
	plot(times_coarse, stack(errs)', label=labels, xlabel="t", ylabel="error(t)")
end

# ╔═╡ 70d5b269-852e-4957-ad6e-5d00c29f3f7a
md"""
There we have it: _there are errors, but they are not quantified by the algorithm!_ Which is unsurprising of course, it's forward Euler after all.

So, let's do better and quantify numerical error. **Let's do _Probabilistic Numerics_!**
"""

# ╔═╡ f4c6d0b5-fc3a-4b47-9063-cdfe302e1148
pn_link = "#" * (PlutoRunner.currently_running_cell_id[] |> string); md"""
# 3. Towards Probabilistic Numerical ODE Solvers

Let's forget most of the above and start over. 
We want to find the ODE solution $y : [t_0, t_\text{max}] \to \mathbb{R}^n$ which satisfies the ODE $y'(t) = f(y(t), t)$ and the initial condition $y(t_0) = y_0$.
But we also know that, even though there is a unique function $y$ with this property, we can never compute it exactly; we can only compute it _approximately_.
So, let's instead find a _probability distribution_ over the ODE solution:

$$p \left( y(t) \mid y(0) = y_0, \{ y'(t_i) = f(y(t_i), t_i) \}_{i=1}^N \right),$$

where $t_0, t_1, \dots, t_N$ is a discrete time grid. 

As you probably know from the lectures, we consider this to be a [Bayesian state estimation problem](https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation), which can be solved (approximately) with [extended Kalman filtering](https://en.wikipedia.org/wiki/Extended_Kalman_filter).
The result is a _probabilistic numerical ODE solution_, and we call the algorithm an _ODE filter_.

**Let's build an ODE filter, step by step!**
"""

# ╔═╡ 0a98a145-e3e9-4b31-ab57-5a7a928115f4
prior_link = "#" * (PlutoRunner.currently_running_cell_id[] |> string); md"""
## 3.1. The Gauss--Markov Prior: 2-times integrated Wiener processes

To do Bayesian inference, we need a prior distribution $p(y(t))$ over the ODE solution $y(t)$.
A natural and common choice are $q$-times integrated Wiener processes (with $q \geq 1$), which are Gauss-Markov processes with transition densities of the form

$$p( Y(t+h) \mid Y(t) ) = \mathcal{N} \left( Y(t + h); Y(t) A(h), Q(h) \right),$$

where $A(h)$ and $Q(h)$ are known matrices that depend on the time step $h$ (they will be given in code below).
The capital $Y(t)$ is a state vector of dimension $d \cdot (q+1)$ (where $d$ is the dimension of the ODE) which tracks not just the ODE solution $y(t)$, but also its first $q$ derivatives:

$$Y(t) = \begin{bmatrix} y(t) \\ y'(t) \\ \vdots \\ y^{(q)}(t) \end{bmatrix}.$$

We can access them via projection matrices $E_0, E_1, \dots, E_q$: $E_i Y(t) = y^{(i)}(t)$.
"""

# ╔═╡ b4a488cc-3d97-48ea-bc77-be8a2177682a
md"""
*In code*, these are the transition matrices of the one-dimensional two-times integrated Wiener process prior:
"""

# ╔═╡ 99e1701c-94ba-4864-b30d-2067db333429
q = 2

# ╔═╡ a2443f6c-8dbb-47c9-a083-866cf75b54e1
A1(h) = [
	1 h h^2/2;
    0 1 h;
    0 0 1;
]

# ╔═╡ c06f27bd-8c14-4d0b-aaee-63f951e0c7b3
A1(0.1)

# ╔═╡ 28c262d4-899f-467d-8bfa-7c7d2b2f79cc
Q1(h) = [
	h^5 / 20    h^4 / 8    h^3 / 6;
    h^4 / 8     h^3 / 3    h^2 / 2;
    h^3 / 6     h^2 / 2    h
]

# ╔═╡ 7df7f8dd-3189-40cc-9320-b78646360c4f
Q1(0.1)

# ╔═╡ 27415e7d-8a6b-448a-95ea-b663f260a288
md"""
Since the ODE is $d$-dimensional we also need a $d$-dimensional prior. We can simply use the same prior for each dimension, and then stack them together:
"""

# ╔═╡ d5e8620b-9604-4f12-8bc6-2434bffec0c0
d = length(y0)

# ╔═╡ 36a2dfe7-0488-4b81-a308-97f0557773cd
A(h) = kron(A1(h), I(d))

# ╔═╡ fe06c623-2c85-4f0a-b8b3-c131c6f48477
Q(h) = kron(Q1(h), I(d))

# ╔═╡ afd8a4ab-1b54-4d44-9255-5d51c3e72815
md"""
Let's also implement the projection matrices $E_0, E_1, E_2$:
"""

# ╔═╡ bb8302f5-4a5e-47f3-ab21-32e7f057f8c5
E0 = kron([1 0 0], I(d))

# ╔═╡ 1128efbe-bcd4-4ae7-b003-950528d272a7
E1 = kron([0 1 0], I(d))

# ╔═╡ ac85113c-3125-480f-ad98-80100a8d0ff9
E2 = kron([0 0 1], I(d))

# ╔═╡ 3abe5e98-a2de-4d57-b2f8-b3ce07d887de
md"""
We also have a standard Gaussian initial distribution 

$$\begin{aligned}
p(Y(t_0)) = \mathcal{N}(Y(t_0); m_0, C_0),
\end{aligned}$$

and we assume $m_0 = 0$ and $C_0 = I$.

*Note:* This initial distribution does not relate to the initial condition $y(t_0) = y_0$ of the ODE! It is just a prior. We will condition on the ODE initial condition later.
"""

# ╔═╡ 1e9bb574-16ac-4ad9-aa7a-2c2cd5ae6b1b
m0 = zeros(d * (q + 1))

# ╔═╡ 8e6cec5d-6c5f-4011-a79f-964b21368972
C0 = Matrix(1.0 * I(d * (q+1)))

# ╔═╡ 73817e2a-ad4b-49c8-852c-3f75d76dbdcb
sampling_link = "#" * (PlutoRunner.currently_running_cell_id[] |> string); md"""
## 3.2. Visualize the prior: Sampling

To get a feeling for the prior $Y(t)$, let's sample from it and plot the samples.  
Recall: The prior is given by

$$\begin{aligned}
p(Y(t_0)) &= \mathcal{N}(Y(t_0); m_0, C_0) \\
p(Y(t+h) \mid Y(t)) &= \mathcal{N} \left( Y(t + h); A(h)Y(t), Q(h) \right),
\end{aligned}$$

with $m_0, C_0, A(h), Q(h)$ as given above.

!!! warning "Task"
    Implement the sampling function `sample_prior` below.
"""

# ╔═╡ d403bef1-bce5-4cfc-8ac0-52996010ad2f


# ╔═╡ fbdf49b8-0d36-4d9d-8e13-e8fed2f86171
function sample_prior(m0, C0, ts::AbstractVector{Float64}, A, Q)::Vector{Vector{Float64}}
	Yt = m0 + cholesky(C0).L * randn(d*(q+1))
    sample = [Yt]
	
	for i in 2:length(ts)
		h = ts[i] - ts[i-1]
		Yt = A(h) * Yt + cholesky(Q(h)).L * randn(d*(q+1))

		push!(sample, Yt)
	end
	
	return sample
end

# ╔═╡ 0331404c-b18d-4ea7-b948-9d7138c6d2f8
let
	ts = [0.0, 1.0, 2.0]
	Ys = sample_prior(m0, C0, ts, A, Q)
	if ismissing(Ys) || length(Ys) == 1
		still_missing(md"Implement the function above and return a sample.")
	elseif length(Ys) != length(ts)
		keep_working(md"The output should be a vector of the same length as the input.")
	else
		correct(md"Seems reasonable! Let's try it out.")
	end
end

# ╔═╡ 1168a27e-8c2f-49fb-9857-3dba65a3bf03
let
    ts = range(tspan[1], tspan[2], length=100)
	Ys = sample_prior(m0, C0, ts, A, Q)
	@assert length(Ys) > 1 "Implementation missing above"
	ys = map(Y -> E0 * Y, Ys)
	plot(layout=(4,1), xlabel="t", ylabel=labels)
	plot!(ts, stack(ys)', label="", color=[1 2 3 4])
end

# ╔═╡ 11356108-c1dd-49c5-81ab-706936c0adf4
md"""
Neat! To get a better feeling for the prior _distribution_, let's plot many samples:
"""

# ╔═╡ ffc68588-f673-4382-94a1-8e4b212124a0
let
    ts = range(tspan[1], tspan[2], length=100)
	plot(layout=(4,1), xlabel="t", ylabel=labels)
	for _ in 1:10
		Ys = sample_prior(m0, C0, ts, A, Q)
		@assert length(Ys) > 1 "Implementation missing above"
		ys = map(Y -> E0 * Y, Ys)
		plot!(ts, stack(ys)', label="", color=[1 2 3 4])
	end
	plot!()
end

# ╔═╡ 9fa7ca21-c6cb-4de0-b3e4-c2e466fe0fb6
md"""
**This is what the 2-times integrated Wiener process prior looks like!**

Above we visalized the zeroth derivative of the state vector $Y(t)$, which is to our prior for the ODE solution $y(t)$.
The lines look quite different from the ODE solution that we're looking for; but after all it's just a prior. 
We will relate the prior to the ODE solution later.
"""

# ╔═╡ c2428a5f-4553-4e07-bc8c-0c9671bea8b0
md"""
To really see why it's called a "2-times integrated Wiener process" we can plot the second derivative of the sample:
"""

# ╔═╡ 5be999d6-70ec-40d7-9d33-37ca8bea155e
let
    ts = range(tspan[1], tspan[2], length=100)
	plot(layout=(4,1), xlabel="t", ylabel=labels)
	for _ in 1:10
		Ys = sample_prior(m0, C0, ts, A, Q)
		@assert length(Ys) > 1 "Implementation missing above"
		ys = map(Y -> E2 * Y, Ys)
		plot!(ts, stack(ys)', label="", color=[1 2 3 4])
	end
	plot!()
end

# ╔═╡ cc1aa6b3-2c74-4e1e-90bd-f4c023d0a0c3
md"""
## 3.3. Towards inference: Marginalizing and conditioning Gaussians

We have a prior $p(Y(t))$. But we want a posterior $p(Y(t) \mid \text{data})$. 
To get there, we need to _marginalize_ and _condition_ Gaussians.
"""

# ╔═╡ cc259a06-e028-4a42-a07c-acdd0cbabab4
marginalization_link = "#" * (PlutoRunner.currently_running_cell_id[] |> string); md"""
### Gaussian marginalization

Let $x$ be a Gaussian-distributed random variable with 
$p(x) = \mathcal{N}(x; m, P)$, 
and let $y$ be conditionally Gaussian-distributed given $x$, with
$p(y \mid x) = \mathcal{N}(y; A x + b, C)$.  
Then $y$ is also Gaussian-distributed:

$$p(y) = \mathcal{N}(y; A m + b, A P A^\top + C).$$

In Kalman filtering, this is also known as the "predict" step. 

!!! warning "Task"
    Implement this in the function `marginalize_gaussian` below.
"""

# ╔═╡ 4d7e17e4-4610-4ad1-8e2c-fad89dfa0e3c
function marginalize_gaussian(m, P, A, b, C)

	ymean = A*m + b
	ycov = A*P*A' + C
	
    return ymean, ycov
end

# ╔═╡ 6ed274e4-dbc0-4629-a0d3-3d838876cb35
let
	out = marginalize_gaussian([1], [1;;], [1;;], [1], [1;;])
	if ismissing(out)
		still_missing(md"Implement the function above.")
	elseif length(out) != 2
		keep_working(md"The output should be a tuple `(mean, cov)`.")
	elseif !(out[1] isa Vector && out[2] isa Matrix)
		keep_working(md"The output should be a tuple `(mean, cov)` where `mean` isa vector and `cov` isa matrix.")
	elseif !(out[1] ≈ [2] && out[2] ≈ [2;;])
		keep_working(md"The actual numbers you computed do not seem correct.")
	else		
		correct()
	end
end

# ╔═╡ 52f23a1c-516c-4c0c-bc37-1af122a3900e
md"""
Let's iteratively apply this to the prior to compute Gaussian marginals $p(Y(t_i)) = \mathcal{N}(Y(t_i); m_i, P_i)$ for all $i$.

In Kalman filtering, this is just a Kalman filter without any data.

!!! warning "Task"
    Implement this in the function `compute_marginals` below.
"""

# ╔═╡ 13b9921d-6f6b-462e-90d4-725892d40707
function compute_marginals(m0, C0, ts, A, Q)
    ms = [m0]
	Cs = [C0]

	# TODO: Fill `ms` and `Cs` with the marginal means and covariances
	m = m0
	C = C0	
	for i in 2:length(ts)
		h = ts[i] - ts[i-1]
		m, C = marginalize_gaussian(m, C, A(h), zeros(length(m)), Q(h))

		push!(ms, m)
		push!(Cs, C)
	end
    
    return ms, Cs
end

# ╔═╡ 576e9138-8e8b-4751-b615-08c6a73e8dfb
let
	out = compute_marginals([1], [1;;], [1, 2, 3], h->[1;;], h->[1;;])
	if ismissing(out) || out == ([[1]], [[1;;]])
		still_missing(md"Implement the function above.")
	elseif length(out) != 2
		keep_working(md"The output should be a tuple `(means, covs)`.")
	elseif !(length(out[1]) == length(out[2]) == 3)
		keep_working(md"The output should be a tuple `(means, covs)` where `means` and `covs` are vectors of the same length as `ts`.")
	else		
		correct(md"Seems reasonable! Let's try it out")
	end
end

# ╔═╡ 34a79302-be2a-4d6b-a509-753fc45abbf1
ms, Cs = compute_marginals(m0, C0, times, A, Q)

# ╔═╡ 55a6a562-1f81-442a-a049-fdea157d023b
md"""
Here is some helper code to plot the marginals:
"""

# ╔═╡ 46f5dc2f-dc36-44c5-b0b3-a8bb063f0283
begin
	function plot_marginals!(ts, ms, Cs; derivative=0, kwargs...)
		E = [E0, E1, E2][derivative+1]
	
		means = map(m -> E * m, ms)
		stddevs = map(C -> sqrt.(diag(E * C * E')), Cs)
			
		plot!(ts, stack(means)'; ribbon=1.96stack(stddevs)', 
			  color=[1 2 3 4], label="", kwargs...)
	end
	
	function plot_marginals(ts, ms, Cs; derivative=0, kwargs...)
		plot(layout=(4,1), xlabel="t", ylabel=labels)
		plot_marginals!(ts, ms, Cs; derivative=0, kwargs...)
	end
end

# ╔═╡ 69660525-2246-4a7a-9e3c-c99d1058f7a8
if length(ms) > 1 # don't execute while not implemented yet
	plot_marginals(times, ms, Cs)
end

# ╔═╡ 9be29ad7-e1b1-4b4e-9b8a-5434b0ee21a8
md"""
!!! check
    Check: Zero-mean, ever-increasing variance, shaped like a trumpet?
    If yes then your marginalization code is probably correct!

So now that we know how to interact with the prior, we will work towards _informing_ the prior about the ODE data.
This is done by _conditioning_ the prior.
"""

# ╔═╡ 783e96cb-ce53-493f-9dcc-51ab620fca7f
affine_conditioning_link = "#" * (PlutoRunner.currently_running_cell_id[] |> string); md"""
### Affine Gaussian Conditioning

Let $x$ be a Gaussian-distributed random variable with
$p(x) = \mathcal{N}(x; m^-, P^-)$,
and let $y$ be conditionally Gaussian-distributed given $x$, with
$p(y \mid x) = \mathcal{N}(y; A x + b, C)$.
Then, $x$ is also conditionally Gaussian-distributed given $y$:

$$\begin{aligned}
p(x \mid y=y) &= \mathcal{N}(x; m, P) \\
m &= m^- + P^- A^\top (A P^- A^\top + C)^{-1} (y - (A m^- + b)) \\
P &= P^- - P^- A^\top (A P^- A^\top + C)^{-1} A P^-.
\end{aligned}$$

In Kalman filtering, this is also known as the "update" step.

!!! warning "Task"
    Implement this in the function `condition_gaussian` below.
"""

# ╔═╡ f80a5591-1387-40ee-8dad-c99c7df01fa0
function condition_gaussian(m, P, A, b, C, y)

	S = inv(A * P * A' + C)
	ymean = m + P * A' * S * (y - (A * m + b))
	ycov = P - P * A' * S * A * P
	
    return ymean, ycov
end

# ╔═╡ 226b69db-cb33-477b-acbe-7e29a7776806
let
	out = condition_gaussian([1], [1;;], [1;;], [1], [1;;], [1])
	if ismissing(out) || ismissing(out[1])
		still_missing(md"Implement the function above.")
	elseif length(out) != 2
		keep_working(md"The output should be a tuple `(mean, cov)`.")
	elseif !(out[1] isa Vector && out[2] isa Matrix)
		keep_working(md"The output should be a tuple `(mean, cov)` where `mean` isa vector and `cov` isa matrix.")
	elseif !(out[1] ≈ [0.5] && out[2] ≈ [0.5;;])
		keep_working(md"The actual numbers you computed do not seem correct.")
	else		
		correct()
	end
end

# ╔═╡ fc15ff6a-0090-4c58-bc06-f994b4a42b2c
begin
	h1 = hint(md"Recall that $E_0 Y(t) = y(t)$.")
	h2 = hint(md"Conditioning on an equation is equivalent to conditioning on a Dirac observation, which again is equivalent to conditioning on a Gaussian observation with zero variance (i.e. $C = 0$).")
	md"""
Let's try it out! 

!!! warning "Task"
    Update the initial distribution $p(Y(t_0))$ on the initial condition: $y(t_0) = y_0$.

$(h1)
$(h2)
"""
end

# ╔═╡ d7b02583-b3e4-44db-96e8-0d8373290c54
m, C = condition_gaussian(m0, C0, E0, zeros(d), zeros(d, d), y0)

# ╔═╡ e6a691db-1ed0-4665-8f93-1593f170ee89
let
	result = if ismissing(m) || ismissing(C)
		still_missing()
	elseif E0 * m == y0 && iszero(E0 * C * E0')
		correct()
	elseif (E0 * m == y0) || iszero(E0 * C * E0')
		almost(md"One of the mean or covariance is correct, but the other isn't yet.")
	else
		keep_working()
	end
	h = hint(md"After conditioning we should satisfy the initial condition exactly: $\mathbb{E}[E_0 Y(t_0)] = y_0$ and $\mathbb{V}[E_0 Y(t_0)] = 0$!")
	md"$result $h"	
end

# ╔═╡ e5da35ae-192a-4a38-ade1-6fe13f53ba07
md"""
Let's look how this affects the prior marginals:
"""

# ╔═╡ 906aee43-d874-409b-b52a-35f40701dd58
if !all(ismissing.((m, C))) 
	ts_pm = range(0, 0.1, length=100)
	ms_pm, Cs_pm = compute_marginals(m, C, ts_pm, A, Q)
	plot_marginals(ts_pm, ms_pm, Cs_pm)
	scatter!(zeros(d, 1), reshape(y0, 1, d), color=[1 2 3 4], label="")
end

# ╔═╡ dbda9377-2e05-44b4-bd50-ec90f2ebcb0d
correct(md"The mean should still be constant, but not zero, and the variance should still increase over time.In particular, the initial condition should now be satisfied. If this is the case then your code is probably good!")

# ╔═╡ e62c343e-616e-434b-af3d-c2d730c5bc5a
let
	h = hint(md"Recall that $E_1 Y(t) = \dot{y}(t)$.")

	md"""From the ODE problem we also get derivative information at the initial time $t_0$: 
$\dot{y}(t_0) = f(y_0, t_0)$. 
This is also useful information, so let's condition on this, too!

!!! warning "Task"
    Condition on the initial derivative: $\dot{y}(t_0) = f(y_0, t_0)$.

$h
"""
	#(ms_pm |> eachrow |> unique |> length) == 1
end

# ╔═╡ 3c511004-5e21-4975-88e1-a1e0b885225f
mcond1, Ccond1 = condition_gaussian(m, C, E1, zeros(d), zeros(d, d), f(y0, p, tspan[1]))

# ╔═╡ 432f9128-f899-438e-98ff-a9406b2745d2
(ismissing(mcond1) && ismissing(Ccond1)) ? missing : let
	ts = range(0, 0.1, length=100)
	ms, Cs = compute_marginals(mcond1, Ccond1, ts, A, Q)
	
	plot_marginals(ts, ms, Cs) # adjust variable names as necessary
	scatter!(zeros(d, 1), reshape(y0, 1, d), color=[1 2 3 4], label="")
end

# ╔═╡ 36fe7a32-1f31-471b-8c26-1fac7aef860d
correct(md"The means should now not be constant anymore.")

# ╔═╡ aee34cb8-f6c5-42fd-901e-a1f52c3567b1
md"""
The last missing piece is to condition _on the ODE itself_. 
And to be able to do that, we need to condition on non-linear observations.
"""

# ╔═╡ c4e594b7-9d4d-4b4f-add3-707dc3848158
linearization_link = "#" * (PlutoRunner.currently_running_cell_id[] |> string); md"""
### Approximate conditioning via linearization

Let $x$ be a Gaussian-distributed random variable with
$p(x) = \mathcal{N}(x; m^-, C^-)$,
and let $y$ be conditionally Gaussian-distributed given $x$, with
$p(y \mid x) = \mathcal{N}(y; g(x), R)$.
*Then, $x$ is **not** conditionally Gaussian-distributed given $y$.*
But to still have efficient, albeit approximate, inference, we can linearize the non-linear observation model $g(x)$ and then still do Gaussian inference.

*Taylor-approximation:* We linearize the model by doing a Taylor approximation:

$$g(x) \approx g(\xi) + G(\xi)^\top (x - \xi),$$

where $\xi$ is some linearization point $\xi$ and $G(\xi)$ is the Jacobian of $g$ at $\xi$.

Then, given the approximate observation model $p(y \mid x) \approx \mathcal{N}(y; g(\xi) + G(\xi)^\top (x - \xi), R)$, $x$ is again conditionally Gaussian-distributed given $y$, with the same update formula as above.

Typically we choose $\xi = m^-$, the mean of the prior.

In Kalman filtering, conditioning with a linearized observation model where the linearization point corresponds to the prior mean is known as to an "extended Kalman update" step.
"""

# ╔═╡ f45937d9-d127-42ce-b89e-9cd446d5bda7
md"""
!!! warning "Task"
    Implement the `linearize` function, which takes in a function `g` and a linearization point `xi`, and returns the parameters of a function $g_\xi(x) = A x + b$ that approximates $g(x)$ around $\xi$.
	You can use `ForwardDiff.jacobian` for this.
"""

# ╔═╡ c8674da0-c7fa-4c5c-9e7e-c976a66d9c34
import ForwardDiff

# ╔═╡ 62bad736-b3a0-4bfa-a55b-242bb7bc2d81
function linearize(g, xi)
	G = ForwardDiff.jacobian(g, xi)
	b = g(xi) - G * xi
	return G, b
end

# ╔═╡ fe0f9312-83ca-4847-9384-5f51998769f8
md"""
Try it out on a simple function: $g(x) = (x+1)^2$
"""

# ╔═╡ d339404a-2ae5-463b-b214-001aafd9a0ef
begin
	g(x) = (x .+ 1) .^ 2
	plot(g, range(-2, 2, length=100), label="h(x)")
end

# ╔═╡ 4b53a0d2-9ee4-4c30-8dde-3ba7415ff16a
md"Linearize at $\xi = 0.0$"

# ╔═╡ 9f2540af-aab9-4faa-823a-84fc00c239cb
xi = [0.0]

# ╔═╡ 81e0051f-70f0-4688-88e1-7ecb9c536cc9
let
	out = linearize(g, xi)
	result = if ismissing(out) || ismissing(out[1])
		still_missing()
	elseif !(out[1] isa Matrix && out[2] isa Vector)
		keep_working(md"The output should be a matrix and a vector.")
	elseif out[1][1] == 2 * (xi[1] + 1) && out[2][1] == g(xi[1]) - out[1][1] * xi[1]
		correct()
	else
		keep_working()
	end	
end

# ╔═╡ 7204380c-5302-4763-b46b-da4e98f17006
G, b = linearize(g, xi)

# ╔═╡ c47f009d-fad7-4507-939b-403cfe5c42df
md"""
We can also visualize what's we're doing here:
"""

# ╔═╡ 2fa961ad-8c37-4c85-8111-842ae292d9cb
md"xi = $(@bind xi_slider Slider(range(-2, 2, length=21), show_value=true, default=0))"

# ╔═╡ bd685522-ab71-4d41-bd93-d980695fd99f
all(ismissing.((G, b))) ? missing : let
	xs = range(-2, 2, length=100)
	plot(g, xs, label="h(x)")
	
	G, b = linearize(g, [xi_slider])
	glin(x) = (G * x + b)[1]
	
	plot!(glin, xs, label="h_lin(x)")
	scatter!([xi_slider], [g(xi_slider)], marker=:x, markersize=15, markeredgewidth=3, label="ξ")
end

# ╔═╡ 332ad601-66d0-45c4-a3c6-38d3285d8726
md"""
Let's now perform approximate conditioning on a non-linear observation!

!!! warning "Task"
    Assume $x \sim \mathcal{N}(0, 1)$, a conditional $y \mid x \sim \mathcal{N} (y; g(x), 1e-3)$, and data $y = 0$.
    Compute an approximate $p(x \mid y) \approx \mathcal{N} (m, P)$, by conditioning on the linearized (or actually "affine-ized"?) observation model and data.
"""

# ╔═╡ aaaba38a-0c06-41e1-8f0e-1ce39af20377
m_before, C_before = [0.0], [1.0]

# ╔═╡ a6f27549-1584-4fee-a503-ad0bef8dc7a7
begin
	__A, __b = linearize(g, m_before)
	__C = 1e-3*I(1)
	m_conditioned, C_conditioned = condition_gaussian(m_before, C_before, __A, __b, __C, [0])
end

# ╔═╡ 7a475a2b-0e15-41d2-8fdf-4bfead53646c
let
	result = if ismissing(m_conditioned) || ismissing(C_conditioned)
		still_missing()
	elseif g(m_conditioned) > g(m_before) 
		keep_working(md"Somehow the update went into the wrong direction. Maybe a sign error?")
	elseif norm(C_conditioned) > norm(C_before) 
		keep_working(md"The covariance increased. We should get _more_ certain, not _less_!")
	else
		correct(md"Seems reasonable!")
	end
end

# ╔═╡ d83e49e6-3514-4de7-a96d-73310abf46e6
information_operator_link = "#" * (PlutoRunner.currently_running_cell_id[] |> string); md"""
## 3.4. The ODE information operator

Let's get back to ODEs! 
From the lectures you probably already know how to construct the _information operator_, but let's briefly recall it nevertheless:
We have

$$\dot{y}(t) = f(y(t), t).$$

In the language of the chosen two-times integrated Wiener process prior, this is equivalent to

$$E_1 Y(t) = f( E_0 Y(t), t),$$

or equivalently

$$0 = E_1 Y(t) - f( E_0 Y(t), t).$$

**This is essentially a nonlinear observation model!**
Indeed, let
$h_i(Y(t_i)) := E_0 Y(t_i) - f(Y(t_i), t_i),$
then $h_i$ is a nonlinear function of the state $Y(t_i)$, and $h_i(Y(t_i)) = 0$ is the observation $i$.
"""

# ╔═╡ 04c3d31e-b5c2-4e99-820d-0895ebdde680
md"""
!!! warning "Task"
    Implement the ODE information operator `h` at time $t_0$.

*Note:* The vector-field $f$ is independent of the time $t$; at this point the dependence on $t$ is just there to make the code more general.
"""

# ╔═╡ 167a7221-452a-4b83-a9fb-950b6feba5bf
h(Y) = E1*Y - f(E0*Y, p, nothing)

# ╔═╡ 8a0b82ac-ec71-492e-91ca-6d4778bc8eeb
md"""
With `h` defined we should be able to just condition (approximately) on this nonlinear observation model.
So, let's "condition on the ODE"!


!!! warning "Task"
    Condition the initial distribution `m0, C0` on the ODE information operator `h` at time `t0`. Save the result into `m0_conditioned`, `C0_conditioned`.
"""

# ╔═╡ 9ba6b83b-4c1f-482f-b98f-11239ee7a48c
m0_conditioned, C0_conditioned = let
	A, b = linearize(h, m0)
	condition_gaussian(m0, C0, A, b, zeros(d, d), zeros(d))
end

# ╔═╡ d271a1ff-5432-4503-9c02-9954d40989db
let
	result = if ismissing(m0_conditioned) || ismissing(C0_conditioned)
		still_missing()
	end
end

# ╔═╡ 4f062a0e-dd9a-4994-83e5-921cc248f210
md"""
...this is still zero mean?? Did anything go wrong? 
No! That's because the zero function $y(t) = 0$ actually solves the ODE, too.
That is, $f(0, t) = 0$ holds. 
Therefore the mean did not change - it was already good.

If we condition on the initial value first, and then on the ODE, we get a different result:

!!! warning "Task"
    Condition the initial distribution `m0, C0` on the initial value `y0`, and then on the ODE information operator `h` at time `t0`.
"""

# ╔═╡ 0eb1590f-b7a0-4e00-aa4b-6de05607baf5
m0_conditioned_0, C0_conditioned_0 = condition_gaussian(m0, C0, E0, zeros(d), zeros(d, d), y0)

# ╔═╡ 3d198b59-0654-4ade-a82c-c047f96a5788
m0_conditioned_1, C0_conditioned_1 = let
	A, b = linearize(h, m0_conditioned_0)
	condition_gaussian(m0_conditioned_0, C0_conditioned_0, A, b, zeros(d, d), zeros(d))
end

# ╔═╡ 674b8281-3df7-43b2-b412-65c94e8d2aa7
let
	result = if any(ismissing.((m0_conditioned_0, C0_conditioned_0, m0_conditioned_1, C0_conditioned_1)))
		still_missing()
	elseif E0*m0_conditioned_0 != y0
		keep_working(md"`m0_conditioned_0` should contain `y0`")
	elseif !iszero(E0*C0_conditioned_0*E0')
		keep_working(md"`C0_conditioned_0` should be zero")
	elseif E1*m0_conditioned_1 != f(y0, p, nothing)
		keep_working(md"`m0_conditioned_1` should contain `f(y0)`")
	elseif !iszero(E1*C0_conditioned_1*E1')
		keep_working(md"`C0_conditioned_0` should be zero")
	else
		correct(md"Seems good! Let's look at what we just did to double-check.")
	end
	
	result
end

# ╔═╡ dc490607-2ca4-435b-884d-57e808b04d5e
md"Before conditioning we estimated $y(t_0)$ and $\dot{y}(t_0)$ to be"

# ╔═╡ bb268636-14b1-4235-825e-eb2c409e1838
(y0=E0*m0, dy0=E1*m0)

# ╔═╡ 01978e85-7633-42ff-accd-8313a80ec632
md"After conditioning on the initial value we have"

# ╔═╡ ba3ac875-3c83-4ee5-a88b-5f44b2e32d94
(y0=E0*m0_conditioned_0, dy0=E1*m0_conditioned_0)

# ╔═╡ a118e1ac-e499-4d57-bcd7-88c10c919b2b
md"After also conditioning on the ODE we have"

# ╔═╡ 95e097b2-2ba9-4b33-a7c2-9f94d70dc655
(y0=E0*m0_conditioned_1, dy0=E1*m0_conditioned_1)

# ╔═╡ b6b7facb-9aa3-4423-a82e-56264f4cb6ad
md"The initial mean should now match the initial condition and the initial derivative exactly!"

# ╔═╡ 4912bea0-4af4-49f9-9ec5-a356a7d94ef8
md"""
At this point we have all the building blocks to implement the full ODE filtering algorithm:
- we have a 2-times integrated Wiener process prior
- we know how to _predict_ future values under the prior (via Gaussian marginalization)
- we know how to _update_ the distribution, both on linear observations (for the initial value) and (approximately) on non-linear observations (for the ODE)

**Let's put it all together!**
"""

# ╔═╡ e0a58a9e-e7a7-4546-99be-7f55cce2bbdf
ode_filtering_link = "#" * (PlutoRunner.currently_running_cell_id[] |> string); md"""
# 4. The ODE Filter

We use everything that we did above and put it together, to build the following algorithm:
    
**Algorithm**

    Input:
    - an initial value problem, consisting of 
      * a vector-field f and 
      * initial value y0,
    - a prior, consisting of 
      * an initial distribution (m0, C0), 
      * transition matrices A(h), Q(h), and 
      * projections to the zeroth and first derivative E0, E1,
    - a discrete-time grid ts=${t_i}_{i=1}^N$ chosen by the user, typically as $t_k=t_0+k*dt$ for some time step dt>0.
  
    Then:
    1. Condition the prior on the initial value y0; Condition the prior on the initial derivative $y'(t_0) = f(y0, t_0)$;
    2. For each time point $t_i$, $i \in \{1, \dots, N\}$:
       - Extrapolate from $p(Y(t_{i-1})) = N(m_{i-1}, C_{i-1})$ to $p(Y(t_i)) = N(m_i^P, C_i^P)$ using the prior, via Gaussian marginalization
       - Condition $Y(t_i)$ on the ODE information operator $h_i$ at time $t_i$, by
          1. linearizing the ODE information operator around the prior mean
          2. conditioning on the linearized ODE information operator
    
    Return: the computed marginals $p(Y(t_i)) \sim \mathbb{N} (m_i, C_i)$, $i \in \{1, \dots, N\}$.
    
    
!!! warning "Task"
    Implement the algorithm described above in the `ode_filter` function.
"""

# ╔═╡ 04ac021b-1bcb-485d-aef7-a666f3c54b36
function ode_filter(f, y0, ts, m0, C0, A, Q, E0, E1; p=p)	
	D = d*(q+1)
	
    # Output:
    ms = typeof(m0)[]
	Cs = typeof(C0)[]

	m, C = m0, C0

	# First condition on initial observations
	m, C = condition_gaussian(m, C, E0, zeros(d), zeros(d,d), y0)
	m, C = condition_gaussian(m, C, E1, zeros(d), zeros(d,d), f(y0, p, tspan[1]))
		
	push!(ms, m)
	push!(Cs, C)
    
	# Then iteratively compute the Gaussian state estimates for each time step
	for i in 2:length(ts)
		dt = ts[i] - ts[i-1]
		
		m, C = marginalize_gaussian(m, C, A(dt), zeros(D), Q(dt))

		h(Y) = E1 * Y - f(E0 * Y, p, ts[i])
		G, b = linearize(h, m)
		m, C = condition_gaussian(m, C, G, b, zeros(d, d), zeros(d))
		
		push!(ms, m)
		push!(Cs, C)
	end
    
    return ms, Cs
end


# ╔═╡ 61ba4734-a538-4e1d-b770-3b50c839a2be
md"dt = $(@bind dt_slider_2 Slider(10 .^ (-2:1//10:1/2), show_value=true))"

# ╔═╡ f152c95a-6b69-4821-b522-8cc034d95200
let
	dt = 1
	ts_sol = range(tspan[1], tspan[2], length=Int(round((tspan[2] - tspan[1]) / dt_slider_2)))
	ms_sol, Cs_sol = ode_filter(f, y0, ts_sol, m0, C0, A, Q, E0, E1)
	
	result = if any(ismissing.((ms_sol, Cs_sol))) || all(length.((ms_sol, Cs_sol)) .== 0)
		still_missing()
	elseif !(length(ms_sol) == length(Cs_sol) == length(ts_sol))
		keep_working(md"The outputs should be vectors of the same length as the input time grid.")
	elseif !(eltype(ms_sol) <: Vector && eltype(Cs_sol) <: Matrix)
		keep_working(md"The returned mean should be a vector of vectors and the returned covariance a vector of covariances.")	
	else
		correct(md"Seems reasonable! Let's test it.")
	end
	
	result
end

# ╔═╡ 779cdcfa-cadc-4986-bf8c-6e4d81e2c27d
begin
	ts_sol = range(tspan[1], tspan[2], length=Int(round((tspan[2] - tspan[1]) / dt_slider_2)))
	ms_sol, Cs_sol = ode_filter(f, y0, ts_sol, m0, C0, A, Q, E0, E1)
	@assert length(ms_sol) > 1 "Implementation missing above"
	
	# The filter returns marginals, so we plot them with `plot_marginals`
	plot_marginals(ts_sol, ms_sol, Cs_sol)
	plot!(times, stack(ys)', labels="", color="black", linestyle=:dash)
end

# ╔═╡ ea5ed6a6-bc35-402d-b56d-4179d6a9d923
md"""
**Check:** 
The ODE filter posterior mean (colored lines) and the reference solution (black dashed lines) should (visually) coincide,
and there should be a visible credible interval around it.

Looks great! The `ode_filter` is able to solve the ODE, and it also returns uncertainties!

Let's visualize uncertainties a bit more by looking at the space of errors directly, i.e. $y(t) - \hat{y}(t)$:
"""

# ╔═╡ 5ff96ed0-fbcf-4947-a878-cae942dc5550
function plot_pn_errors(ts, ms, Cs, ys_acc)

	ys = map(m -> E0 * m, ms)
	ys_ref = reference_solution(ts)
	errs = ys_ref .- ys

	stddevs = map(C -> sqrt.(diag(E0 * C * E0')), Cs)

	errs = stack(errs)'
	stddevs = stack(stddevs)'
	
	plot(layout=(4,1), xlabel="t", ylabel=labels)
	plot!(ts, zero(errs), ribbon=3stddevs, label="", color=[1 2 3 4])
	plot!(ts, errs, color="black", linestyle=:dash, label="")
end

# ╔═╡ e804d104-3811-46f2-b014-d5931ab61f23
let
	@assert length(ms_sol) > 1 "Implementation missing above"
	plot_pn_errors(ts_sol, ms_sol, Cs_sol, ys)
end

# ╔═╡ 4f2e6d01-8673-4b4e-a4c5-fe13c744e0f9
md"""    
**Check:** 
The colored and black-dashed lines should both be around zero, and the y-axis should be in the order of 1.


This plot reveals an important issue:
*The actual error seems to be much smaller than the posterior uncertainty.*
To be a meaningful error estimate, the posterior uncertainty should be *calibrated*, which (among other things also) means that it should have a similar scale as the true error.
How do we do this? By estimating the hyperparameter `sigma_sq` that we didn't really talk about yet!
"""

# ╔═╡ 284cfdd1-6d25-47fb-ba5d-e0a563197ce3
calibration_link = "#" * (PlutoRunner.currently_running_cell_id[] |> string); md"""
# 5. Calibrating uncertainties

*There is a hyperparameter $\sigma^2$ in our choice of 2-times integrated Wiener process prior that we (purposefully) didn't talk about yet.*

As our prior, we have

$$\begin{aligned}
Y(t_0) &\sim \mathcal{N}(0, \sigma^2 I), \\
Y(t+h) \mid Y(t) &\sim \mathcal{N}(Y(t); A(h) Y(t), \sigma^2 Q(h)),
\end{aligned}$$

with some scalar hyperparameter $\sigma^2$, often also called the *diffusion* parameter.

So basically, all the covariances that appear in the prior are scaled by some a-priori unknown parameter $\sigma^2$.  
*It is not really surprising that the posterior uncertainty is not calibrated, if there is some (so far) arbitrarily chosen hyperparameter floating around in our model!*
"""

# ╔═╡ 9a815e39-65b8-4bd1-bb98-6e241f11f24d
md"""
## How to estimate the diffusion $\sigma^2$ 

We can estimate the diffusion $\sigma^2$ by maximizing the likelihood of the observations, 
where, strictly speaking, "observations" here means the ODE information "$0$" observed with the obseravtion model "$h_i$". 
We will not go into detail _why_ this is the correct thing to do 
(for that have a look [^4] or [^7],
but **this is how it works:**

$$\hat{\sigma}_\text{MLE}^2 = \frac{1}{Nd} \sum_{i=1}^N (H m_i^P + b)^\top (H C_i^P H^\top)^{-1} (H m_i^P + b),$$

are the mean and covariance of the distribution _before_ the conditioning at time $t_i$ (in filtering called the "prediction" mean and covariance, thus the "P").
Note that the quantities inside the sum are computed during the Gaussian conditioning anyways, so a good place to compute these terms is inside the Gaussian conditioning function `condition_gaussian`.
Then, once we have $\hat{\sigma}_\text{MLE}^2$, just re-scale the covariances:

$$C_i^\text{cal} =  \hat{\sigma}_\text{MLE}^2 \cdot C_i.$$

That's it!

(If you're iterested in the details and the derivations and proofs, have a look at Section 4 of [^4] or Section 3 of [^7]!)
"""

# ╔═╡ 12ac610e-0709-43b6-8ee2-3c0befaccc48
md"""
!!! warning "Task"
    Implement a new function `condition_gaussian_cal` that computes not just the conditioned mean and covariance, but also the term inside the sum above.
"""

# ╔═╡ 5f93f93a-f6df-4247-ab71-c8430e1b12ac
function condition_gaussian_cal(m, P, A, b, C, y)

	yhat = A * m + b
	Sinv = inv(A * P * A' + C)
	
	ymean = m + P * A' * Sinv * (y - yhat)
	ycov = P - P * A' * Sinv * A * P

	sigma_sq_inc = (y-yhat)' * Sinv * (y-yhat)
	
    return ymean, ycov, sigma_sq_inc
end

# ╔═╡ b882cc9a-e52c-400b-84d3-aa77c83b559e
let
	out = condition_gaussian_cal([1], [1;;], [1;;], [1], [1;;], [1])
	if any(ismissing.(out))
		still_missing(md"Implement the function above.")
	elseif length(out) != 3
		keep_working(md"The output should be a tuple `(mean, cov, σ²_increment)`.")
	elseif !(out[1] isa Vector && out[2] isa Matrix && out[3] isa Number)
		keep_working(md"The output should be a tuple `(mean, cov, σ²_increment)` where `mean` isa vector and `cov` isa matrix, and `σ²_increment` isa number.")
	elseif !(out[1] ≈ [0.5] && out[2] ≈ [0.5;;] && out[3] ≈ 0.5)
		keep_working(md"The actual numbers you computed do not seem correct.")
	else
		correct()
	end
end

# ╔═╡ 1f9b1229-0b5d-4bb8-8edb-dca3f7fce44e
md"""
Let's implement a _calibrated_ ODE filter now!

!!! warning "Task"
    Implement the calibrated ODE filter in the `ode_filter_cal` function.    
"""

# ╔═╡ a98eb4ad-4606-4f38-874d-c559ad7dfef1
function ode_filter_cal(f, y0, ts, m0, C0, A, Q, E0, E1; p=p)
	D = length(m0)
	
	# Some magic that might be handy later
	m0 = eltype(p).(m0)
	C0 = eltype(p).(C0)
	
    # Output:
    ms = typeof(m0)[]
	Cs = typeof(C0)[]

	m, C = m0, C0

    # Output:
    ms = typeof(m0)[]
	Cs = typeof(C0)[]

	m, C = m0, C0

	# First condition on initial observations
	m, C = condition_gaussian(m, C, E0, zeros(d), zeros(d,d), y0)
	m, C, sigma_sq_acc = condition_gaussian_cal(m, C, E1, zeros(d), zeros(d,d), f(y0, p, tspan[1]))
		
	push!(ms, m)
	push!(Cs, C)
    
	# Then iteratively compute the Gaussian state estimates for each time step
	for i in 2:length(ts)
		dt = ts[i] - ts[i-1]
		
		m, C = marginalize_gaussian(m, C, A(dt), zeros(D), Q(dt))

		h(Y) = E1 * Y - f(E0 * Y, p, ts[i])
		G, b = linearize(h, m)
		m, C, sigma_sq_inc = condition_gaussian_cal(m, C, G, b, zeros(d, d), zeros(d))
		sigma_sq_acc += sigma_sq_inc
		
		push!(ms, m)
		push!(Cs, C)
	end
    
	# And don't forget to re-scale the covariances with the MLE
	Cs .*= sigma_sq_acc / d / length(ts)

    return ms, Cs
end

# ╔═╡ a0e66a7c-3c44-4ecb-9718-3334ed16acef
let
	dt = 1e-2
	ts_sol = range(tspan[1], tspan[2], length=Int(round((tspan[2] - tspan[1]) / dt)))
	ms_sol, Cs_sol = ode_filter_cal(f, y0, ts_sol, m0, C0, A, Q, E0, E1)
	
	result = if any(ismissing.((ms_sol, Cs_sol))) || length(ms_sol) == 0
		still_missing()
	elseif !(length(ms_sol) == length(Cs_sol) == length(ts_sol))
		keep_working(md"The outputs should be vectors of the same length as the input time grid.")
	elseif !(eltype(ms_sol) <: Vector && eltype(Cs_sol) <: Matrix)
		keep_working(md"The returned mean should be a vector of vectors and the returned covariance a vector of covariances.")	
	else
		correct(md"Seems reasonable! Let's test it.")
	end
	
	result
end

# ╔═╡ d2caedba-3b90-4a2d-9167-985675dd77da
md"dt = $(@bind dt_slider_3 Slider(10 .^ (-2:1//10:1/2), show_value=true))"

# ╔═╡ e177c47e-2d16-4f71-946f-09e9d7a3d8b8
begin
	ts_sol_2 = range(tspan[1], tspan[2], length=Int(round((tspan[2] - tspan[1]) / dt_slider_3)))

	ms_sol_cal, Cs_sol_cal = ode_filter_cal(f, y0, ts_sol_2, m0, C0, A, Q, E0, E1)
	@assert length(ms_sol_cal) > 1 "Implementation missing above"
	
	plot_marginals(ts_sol_2, ms_sol_cal, Cs_sol_cal)
	plot!(times, stack(ys)', labels="", color="black", linestyle=:dash)
end

# ╔═╡ e61b237d-e9ce-4ca2-8cdb-1e1fbab1a637
begin
	@assert length(ms_sol_cal) > 1 "Implementation missing above"
	plot_pn_errors(ts_sol_2, ms_sol_cal, Cs_sol_cal, ys)
end

# ╔═╡ 67c90ed2-7035-4152-8b49-491c5c55bf72
md"""    
**Check:** 
The colored lines should be exactly zero, but this time both the colored uncertainty estimates and the numerical error (the black lines) should be visibly non-zero.
"""
#**It works! We again get relatively small errors, but this time the errors live on the same scale as the error estimate!**
#"""

# ╔═╡ 0f0ba357-bd35-4bc3-93e7-c0e7d69768e5
parameter_inference_link = "#" * (PlutoRunner.currently_running_cell_id[] |> string); md"""
# 6. Parameter estimation with ODE filters

!!! info
	At this point we're done with the main part: We implemented an ODE filter, completely from scratch. Nice!
	This final section shows how to _use_ the ODE filter in a problem setting that is not purely about _simulating_ and ODE, but about _learning_ the ODE from data.

In many problems of interest we don't actually want to solve a known ODE.
Instead, we have some observed data and a hypothesis for how the data was generated, i.e. a model, and we want to learn the parameters of the model that fit the data best.
We call this _parameter estimation_, or sometimes also an _inverse problem_.

In the case of ODEs, we can formulate the parameter estimation problem as follows:
- We have an **initial value problem**
  $$\dot{y}(t) = f(y(t), t, \theta), \quad t \in [t_0, t_\text{max}], \qquad y(t_0) = y_0(\theta),$$
  where $\theta$ is a vector of (unknown) parameters.
- We have **observations**
  $$u_i = H y(t_i) + \epsilon_i, \qquad i \in \{1, \dots, M\},$$
  where $H$ is a linear observation operator, $\epsilon_i \sim \mathcal{N} (0, \nu^2I)$ is iid. Gaussian noise, and $t_i$ are the observation times.
- **We want to estimate the parameters $\theta$:** For this tutorial we will just compute _maximum-likelihood estimates_
  $$\hat{\theta} = \arg\max_\theta \mathcal{L}(\theta) = \arg\max_\theta \log p(u_1, \dots, u_N \mid \theta).$$
  <font color='gray'>
  The method can also be extended to maximum a-posteriori estimates or to full Bayesian inference; in those cases just add a prior and then either optimize or run MCMC, or other (possibly approximate) inference methods. 
  But in both cases you need to compute the marginal likelihood - this is the main content of this section.
  </font>

How do we do this in the context of ODE filters? This paper[^10] explains how!
In a nutshell, we use the ODE filter _posterior_ as a _prior_ for a standard Gaussian process regression problem, and then compute the maximum likelihood estimate of the parameters $\theta$ by optimizing the log-likelihood.
This is what we'll do - in a slightly simplified setting.
"""

# ╔═╡ 9e2df89e-7faf-48a1-90c3-55e44b2d02fc
md"""
## The concrete parameter inference problem 

We still consider the SIRD model as [above](#sird), but with a reduced time-span $t \in [0, 15]$.  
Let $\beta = 0.5$, $\gamma = 0.06$, $\eta = 0.002$ (as above) be the unknown ground truth parameters.  
As _data_, we have just a single observation $u = H y(t_\text{max})$ at the final time point, 
with $H = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$, that is, we observe only the last two dimensions of $y$ corresponding to the number of recovered and dead individuals.
Our goal is then to estimate the parameters $\theta = (\beta, \gamma, \eta)$ from this single observation.

So on a high level, the question is: *What are the contact rate, the recovery rate, and the death rate, that would explain the observed number of recovered and dead individuals?*

Visually, the problem looks like this:
"""

# ╔═╡ ad5033a9-166b-44f6-99eb-03b64b38adbb
H = [0 0 1 0
     0 0 0 1]

# ╔═╡ cd63a01a-d24e-4786-822f-64356c994f28
p_true = (0.5, 0.06, 0.002)

# ╔═╡ 0b820d93-da41-43d3-9f57-b6d5cd1a0756
tspan2 = (0, 15)

# ╔═╡ 9ed6966f-2934-4f92-a44e-da5a5150481d
ts2 = range(tspan2..., length=1000)

# ╔═╡ 5b194301-c8b8-4024-9ac1-76e9593db32f
ys_true = let
	ms, _ = ode_filter(f, y0, ts2, m0, C0, A, Q, E0, E1, p=p_true)
	map(m->E0*m, ms)
end

# ╔═╡ 83299d45-bb36-4f72-9a02-4682bdf91020
data = H*ys_true[end]

# ╔═╡ ce4b0038-cd33-43a7-b3a8-e53c8f23eb5b
let
	plot(layout=(4,1), xlabel=["" "" "" "t"], ylabel=labels)
	plot!(ts2, stack(ys_true)', color="black", linestyle=:dash, label="")
	pinf_colors = findfirst.(eachrow(Bool.(H)))'
	scatter!([tspan2[end]], replace(H' * data, 0.0 => missing)', 
			 label="", color=[1 2 3 4])
end

# ╔═╡ af110785-8412-43d2-8d7a-f45c021ff7e4
md"""
Our goal is to recover the true solution (the black dashed line) from the observation (the colored cross).

To do this via maximum-likelihood estimation, we need two things:
1. We need a marginal likelihood function $p(u \mid \theta)$, and then
2. We maximize this marginal likelihood (or actually minimize the negative marginal log-likelihood) with some optimizer.
So let's get to it!
"""

# ╔═╡ d8024d4a-42c0-4c2f-b8ee-5aeb19302100
md"""
## The marginal likelihood
When we run an ODE filter, we compute a posterior
$$p(y(t_i) \mid \theta) = \mathcal{N} \left( y(t_i); E_0 m_i, E_0 P_i E_0^\top \right), \qquad i \in \{1, \dots, N\}.$$
In the parameter inference problem, we assume a data likelihood
$$p(u \mid y(t_\text{max})) = \mathcal{N}(u; H y(t_\text{max}), \sigma^2 I).$$
So, using the (approximate) posterior from the ODE filter above, we can compute the (approximate) marginal likelihood:

$$p(u \mid \theta) 
= \int p(u \mid y(t_\text{max})) p(y(t_\text{max}) \mid \theta) \mathrm{d}y(t_\text{max}) 
= \mathcal{N} \left(u; H E_0 m_N, H E_0 P_N E_0^\top H^\top + \sigma^2 I \right),$$

since both distributions are Gaussian.  

**That's it! This is the one formula that we need to implement to compute the marginal likelihood and then do parameter inference with ODE filters.**

!!! warning "Task"
	Implement a negative marginal log-likelihood function `nll` that computes the negative log-likelihood of the observation $u$ given the parameters $\theta$ and observation noise $\sigma^2 = 10^{-10}$.
    

$(hint(md"Don't forget to project the ODE filter posterior to the solution space with `E0`, and the solution to the data space with `H`."))

$(hint(md"You can use `Distributions.MvNormal` and `Distributions.logpdf` to compute the log-likelihood of a multivariate Gaussian. This might need applying `LinearAlgebra.Symmetric` to the covariance."))
"""

# ╔═╡ f13ec44f-307a-490f-a5e8-b642e48042ee
import Distributions

# ╔═╡ dc0a8b43-00f9-4731-8b3e-289065e700f1
function nll(p)
    
    # TO IMPLEMENT
	ms, Cs = ode_filter_cal(f, y0, ts2, m0, C0, A, Q, E0, E1; p=p)
	final_y_mean = E0 * ms[end]
	final_y_cov = E0 * Cs[end] * E0'

	obs_mean = H * final_y_mean
	obs_cov = H * final_y_cov * H' + 1e-10*I

    return -Distributions.logpdf(Distributions.MvNormal(obs_mean, Symmetric(obs_cov)), data)
end

# ╔═╡ e3df901b-bf88-4031-a25c-e01739284085
p0 = [0.3, 0.03, 0.003]

# ╔═╡ 244b8480-d74a-4734-a500-2d439af7546d
nll(p_true), nll(p0)

# ╔═╡ 656d511d-1f37-4288-a479-90112f84cc19
md"""    
**Check:** `nll(p_true)` should be a lower value than `nll(p0)`. Since `p_true` was used to generate the data, its likelihood should be much higher than the likelihood of another arbitrarily chosen parameter!
    
If both values are the same, you probably did not pass the parameter `p` on to the ODE vector field `f` in your `ode_filter_cal` implementation.
"""

# ╔═╡ 8f3d7317-f8af-4aa5-b9ab-67456d22156e
md"""
## Minimizing the negative log-likelihood

There is only one thing left to do: Compute the maximum-likelihood estimate by minimizing `nll`!

!!! warning "Task"
	Starting from `p0`, minimize the negative marginal log-likelihood `nll` with `Optim.optimize`.
	The "Nelder-Mead" method should work reasonably well.
"""

# ╔═╡ 3a663281-c595-4233-8304-fb3daa1e3667
res = optimize(nll, p0, NelderMead())

# ╔═╡ f975608b-7271-4139-874e-08506e8d6375
p_opt = res.minimizer

# ╔═╡ 1f4cbb92-1478-4999-9d47-e56678e8eb7b
p_opt, nll(p_opt)

# ╔═╡ 81f18197-39f0-4c25-9e82-6d0071b99aee
md"""    
**Check:** `nll(p_opt)` should be much lower than `nll(p0)`, otherwise the optimization went completely wrong.
    
Also: It is quite probable you did not actually recover the true parameters - _this is intended_. We will have a look at the result of the optimization next.

To properly evaluate the quality of the result, let's make another plot:
"""

# ╔═╡ 01055f56-a49a-470f-a021-e67e7f9efb87
let
	@assert !ismissing(p_opt) "Implementation missing above"
	ms0, Cs0 = ode_filter_cal(f, y0, ts2, m0, C0, A, Q, E0, E1, p=p0)
	ms_opt, Cs_opt = ode_filter_cal(f, y0, ts2, m0, C0, A, Q, E0, E1, p=p_opt)

	plot_marginals(ts2, ms0, Cs0; linestyle=:dash)
	plot_marginals!(ts2, ms_opt, Cs_opt)
	plot!(ts2, stack(ys_true)', labels="", color="black", linestyle=:dash)
	scatter!([tspan2[end]], replace(H' * data, 0.0 => missing)', 
		     flabel="", color=[1 2 3 4], label="")
end

# ╔═╡ f01e08dc-a621-4a9a-b355-b78555d63216
md"""
It worked! At least, kind of? We did recover parameters that explain the data well. 
But, the inferred parameters are actually quite different from the true parameters and we see a mismatch between the true and the inferred solution, in particular for the number of susceptible and infected individuals.
That's because the very limited data we have is not actually sufficient to identify the parameters uniquely - 
_which is why it would indeed make sense to be more Bayesian about this and compute an actual posterior over the parameters_. 
So if you have some time left, you might want to try that out!

Also: If we had a different setup where S and I are observed too, the inferred solution would have been very close to the true solution (you can verify this for yourself by changing `H` above!).

**This concludes the tutorial!** Thanks a lot for reading and coding, and for participating in the workshop, I hope you enjoyed it!
"""

# ╔═╡ d37ade4a-cb4e-42f4-b3ed-997cd9e40d53
begin
more_link = "#" * (PlutoRunner.currently_running_cell_id[] |> string) 
Markdown.parse(raw"""
# Things that we had to leave out

We built a functioning ODE filter from scratch, but there are a few things that we had to leave out for the sake of simplicity.
The actual algorithms that appear in most of the literature are a bit more involved, and there are many bells and whistles that can be added to make these methods more flexible, more robust and more performant, _which matter a lot when you actually want to use these methods in practice_.
Things we didn't cover include:

- **Smoothing:**   
  The algorithm we implemented above is a _filter_. 
  The "posterior" we computed is essentially
  
  $$p \left( y(t_i) \mid y(0) = y_0, \{ y'(t_i) = f(y(t_i), t_i) \}_{j=1}^{i-1} \right), \qquad i \in \{1, \dots, N\},$$

  that is, the posterior of the solution at time $t_i$ only considers the ODE information up to time $t_{i-1}$.
  But what we actually described [above](#pn_link) is a posterior that considers the ODE information _everywhere_.
  This is what a _smoother_ does.
  Typically, if you are looking at solution strategies, this might be the quantity you are interested in; and if you want _interpolation_ (i.e. computing the solution at arbitrary times), you will also need to smooth.  
  To learn more about filtering and smoothing in general have a look at this book[^1];
  and in the ODE context, the "Probabilistic Numerics" book[^11] is a great starting point!
- **Other priors:**   
  In this tutorial we only considered a two-times integrated Wiener process prior, but depending on your problem you might want higher orders.
  Generally speaking, $q$-times integrated Wiener processes are the most popular choice, and all the [software](#software_link) packages linked below implement them;
  but you could also use other Gauss--Markov priors, e.g. $q$-times integrated Ornstein--Uhlenbeck processes, or even more general Markov processes.  
  This paper [^5] mentions the various priors; or check out the [ProbNumDiffEq.jl documentation](https://docs.sciml.ai/ProbNumDiffEq/stable/priors/) to just use them.
- **Square-root implementation & preconditioning**  
  It turns out that the standard extended Kalman filtering implementation that we did here can become numerically unstable for high-dimensional problems and/or very small step sizes [^6] - _this is a problem that you will actually run into in practice!_
  The solution is to use a square-root implementation, together with preconditioning (essentially a change of coordinates in the state space).  
  All the software packages linked [below](#software_link) implement this and provide numerically stable solvers.
- **Approximate linearization:**  
  The ODE filter we implemented is essentially a _extended Kalman filter_, which is a Gaussian filter combined with a first-order Taylor linearization of the vector field.
  There is more you can do:
  We can do a zeroth-order Taylor linearization of the vector field to gain some speed (actually a lot of speed; but at the cost of stability and coarser uncertainty quantification) [^9];
  or we could also do _statistical linearization_ (related to the _unscented_ Kalman filter that you might have heard about)[^2][^4].
  If you want to just use this functionality, have a look at [probdiffeq](https://pnkraemer.github.io/probdiffeq/) which provides a range of linearization strategies, implemented very efficiently.
- **Adaptive time-stepping:**  
  In practice, we often want to discretize the ODE _adaptively_, to have more steps in regions where the solution is changing rapidly, and fewer steps in regions where the extrapolation is very accurate.
  To learn how to do this with ODE filters, have a look at [^3] or [^7].
  Or again: Just use any one of the [software](#software_link) packages linked below.
- **Better initialization with taylor-mode autodiff:**  
  Above, we initialized the ODE filter with the solution and derivative at the first time point - which left the second derivative, _which we do also model_, uncertain.
  This is not ideal, but we can do better:
  The ODE actually contains information about _all_ derivatives of the solution (just apply the chain rule to verify this),
  and it turns out that we can efficiently compute these with a computer by using Taylor-mode automatic differentiation!
  For more information have a look at[^6],
  or [the documentation of `jax.experimental.jet`](https://jax.readthedocs.io/en/latest/jax.experimental.jet.html),
  or at [TaylorIntegration.jl](https://perezhz.github.io/TaylorIntegration.jl/stable/).  
  Or, just use it with any of the [software](#software_link) packages linked below.
- **Flexible information operators:**  
  There are many ODE-related problems that do not 100% exactly correspond to the problem setup from [above](#problem-setup), for example higher-order ODEs, dynamical systems with conserved quantities (e.g. often energy, mass, or momentum), or even differential-algebraic equations.
  _You can solve these with (O)DE filters too!_ 
  See this paper[^8] for an explanation of how to do this;
  and again, just use any of the [software](#software_link) packages linked below to do this in practice.
""")
end

# ╔═╡ c21f17bf-e4e2-4b73-b6d8-96e53dd576f6
software_link = "#" * (PlutoRunner.currently_running_cell_id[] |> string); md"""
# Software packages
"""

# ╔═╡ 0ad15284-e678-4341-8278-18e423b77034
Markdown.parse("""
1. [**The ODE**]($problem_setting_link)  
   The guiding example for this tutorial will be an epidemeological dynamical system: a so-called SIRD model.
   It is a simple model that describes how a disease evolves over time in a population.
2. [**Solve an ODE with forward Euler**]($forward_euler_link)
   The "hello world" of solving ODEs. 
   Forward Euler is a very simple, non-probabilistic ODE solver. 
   We will use it to perform first simulations of the SIRD model, and we will visualize the solution and the numerical errors - _which are not quantified by the algorithm_.
3. _Towards ODE filters:_ [**Gauss-Markov priors and Gaussian inference**]($prior_link)
   ODE filters essentially perform nonlinear Gauss-Markov regression.
   So, in this next step we explore these Gauss-Markov priors and get more familiar with Gaussian distributions.
   We  
   + [sample from a Gauss--Markov prior]($sampling_link)
   + implement Gaussian [marginalization]($marginalization_link) and [conditioning]($affine_conditioning_link)
   + condition on non-linear observations by [linearizing]($linearization_link) the model
   These are the main algorithmic building blocks that we need for _extended Kalman filtering_; next, we turn it into an _ODE filter_.
3. [**The ODE information operator**]($information_operator_link)
   In this section we define they key component of the ODE filter: the _information operator_.
   This is the part that turns an extended Kalman filter into an ODE solver! 
5. Putting things together: [**Your first ODE filter**]($ode_filtering_link)
   We assemble all of the above to build an ODE filter, and we use it to solve the SIRD model.
   We will visualize the posterior and its uncertainties, and again look at the numerical error - _and this time also at error estimates_! 
5. [**Calibrating uncertainties**]($calibration_link)
   It turns out that there is still one thing left to do: We need to calibrate the error estimates.
   So, we implement a quasi-maximum likelihood estimation for the one free _diffusion_ hyperparameter, and get a _calibrated_ ODE filter.
   We solve the ODE again, visualize it, and get meaningful posteriors!

**At this point you will have implemented your own ODE filter, completely from scratch!**  
There is one more section where we use these filters, not to solve an ODE but inside of an inference problem:   

6. [**Parameter inference**]($parameter_inference_link)
   Sometimes you don't want to infer the solution from the ODE, but infer the ODE itself from some data. This is what we'll consider in this _parameter inference_ part.
   We will use our ODE filter implementation to do some maximum-likelihood estimation of the SIRD parameters (the contact rate, recovery rate, and death rate).


After all of this, if you still have time and are curious for more, there are two directions you could go:
there is still quite some functionality that one would want to have when using ODE filters, so you could have a look at [this list]($more_link) and explore these.
Or you could simply check out our [software packages]($software_link)! They provide feature-rich ODE filters which are implemented very efficiently, and there are many tutorials to guide you through both basic and more avanced use-cases.

**Let's get started!**
""")


# ╔═╡ c09f1297-b309-4043-8c3f-5726bdaf203b
Markdown.parse("""
## [ProbNumDiffEq.jl](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable/): *Probabilistic Numerical Differential Equation solvers in Julia*

ProbNumDiffEq.jl provides probabilistic numerical ODE solvers to the Julia [DifferentialEquations.jl](https://diffeq.sciml.ai) ecosystem.
They are implemented as easy-to-use drop-in replacements to their classic counterparts, aim to be as fast as possible, and provide most of the bells and whistles mentioned [above]($more_link).
Have a look at the [documentation](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable/) for more information; or just try them out with the following code snippet:

**Example:** Solving the SIRD model as defined in this tutorial with ProbNumDiffEq.jl

```julia
using ProbNumDiffEq, Plots

# Define the ODE problem
function f(du, u, p, t)
    S, I, R, D = u
    β, γ, μ = p
    du[1] = -β * S * I
    du[2] = β * S * I - γ * I - μ * I
    du[3] = γ * I
    du[4] = μ * I
end 
u0 = [0.99, 0.01, 0.0, 0.0]
tspan = (0.0, 100.0)
p = [0.5, 0.06, 0.002]
prob = ODEProblem(f, u0, tspan, p)

# Solve the ODE problem with an ODE filter
sol = solve(prob, EK1())

# Plot the solution
plot(sol)
```
""")

# ╔═╡ b55f6094-d95a-463b-8afb-e5e68a704140
md"""
## [probdiffeq](https://pnkraemer.github.io/probdiffeq/): *Probabilistic solvers for differential equations in Jax*

ProbDiffEq implements adaptive probabilistic numerical solvers for initial value problems.
It inherits automatic differentiation, vectorisation, and GPU capability from JAX, works well with JAX's just-in-time compilation to achieve high performance, and is compatible with other packages from the Jax ecosystem such as [Optax](https://optax.readthedocs.io/en/latest/index.html) or [Blackjax](https://blackjax.readthedocs.io/en/latest/).
Have a look at it's documentation [here](https://pnkraemer.github.io/probdiffeq/)!

**Example:** Solving the SIRD model as defined in this tutorial with `probdiffeq`

```python
import jax
import jax.numpy as jnp

from probdiffeq import solution_routines, solvers
from probdiffeq.implementations import recipes
from probdiffeq.strategies import smoothers

# Define the ODE problem 
@jax.jit
def vector_field(y, *, t, p):
    S, I, R, D = y
    β, γ, μ = p
    return jnp.array([ -β * S * I, β * S * I - γ * I - μ * I, γ * I, μ * I ])
u0 = jnp.asarray([0.99, 0.01, 0.0, 0.0])
t0, t1 = 0.0, 100.0
p = jnp.asarray([0.5, 0.06, 0.002])

# Solve the ODE problem with an ODE filter
implementation = recipes.DenseTS1.from_params(ode_shape=(len(u0),))
strategy = smoothers.Smoother(implementation)
solver = solvers.MLESolver(strategy)
solution = solution_routines.solve_with_python_while_loop(
    vector_field, initial_values=(u0,), t0=t0, t1=t1, solver=solver, parameters=p
)

# Look at the solution
print("u =", solution.u)
```
"""

# ╔═╡ c771995f-2ef9-46ed-98da-3545e34748ce
md"""
## [probnum](https://probnum.readthedocs.io/en/latest/index.html): *Probabilistic Numerics in Python*

ProbNum is a Python toolkit which provides probabilistic numerical methods not only for ODEs, but for a whole range of numerical problems in linear algebra, optimization, quadrature and differential equations.
It also contains quite a large range of methods for Bayesian filtering and smoothing that you can use to build your own probabilistic ODE solvers.
To learn more, have a look at its excellent documentation and the many tutorials [here](https://probnum.readthedocs.io/en/latest/index.html).

**Example:** Solving the SIRD model as defined in this tutorial with `probnum`

```python
import numpy as np
from probnum.diffeq import probsolve_ivp

# Define the ODE problem
p = [0.5, 0.06, 0.002]
def f(t, y):
    S, I, R, D = y
    β, γ, μ = p
    return np.array([ -β * S * I, β * S * I - γ * I - μ * I, γ * I, μ * I ])
def Jf(t, y): # define the jacobian manually
    S, I, R, D = y
    β, γ, μ = p
    return np.array([
        [-β * I, -β * S, 0, 0],
        [β * I, β * S - γ - μ, -γ, -μ],
        [0, γ, 0, 0],
        [0, μ, 0, 0]
    ])
y0 = np.array([0.99, 0.01, 0.0, 0.0])
t0, tmax = 0.0, 100.0

# Solve the ODE problem with an ODE filter
sol = probsolve_ivp(f, t0, tmax, y0, df=Jf, method="ek1")

# Look at the solution
print("mean: ", sol.states.mean)
```
"""

# ╔═╡ eba73262-bc98-4a9f-83db-c3ade9c6d09f
md"""
# References

(sorted roughly by date)

[^1]: *Bayesian Filtering and Smoothing*, Särkkä, 2013 ([link](https://www.cambridge.org/core/books/bayesian-filtering-and-smoothing/C372FB31C5D9A100F8476C1B23721A67))

[^2]: *Active Uncertainty Calibration in Bayesian ODE Solvers*, Kersting and Hennig, 2016 ([link](http://auai.org/uai2016/proceedings/papers/163.pdf))

[^3]: *A probabilistic model for the numerical solution of initial value problems*, Tiemann (né Schober) et al, 2019 ([link](https://link.springer.com/article/10.1007/s11222-017-9798-7))

[^4]: *Probabilistic solutions to ordinary differential equations as nonlinear Bayesian filtering: a new perspective*, Tronarp et al, 2019 ([link](https://link.springer.com/article/10.1007/s11222-019-09900-1))

[^5]: *Bayesian ODE solvers: the maximum a posteriori estimate*, Tronarp et al, 2021 ([link](https://link.springer.com/article/10.1007/s11222-021-09993-7))

[^6]: *Stable Implementation of Probabilistic ODE Solvers*, Krämer and Hennig, 2020 ([link](https://arxiv.org/abs/2012.10106))

[^7]: *Calibrated Adaptive Probabilistic ODE Solvers*, Bosch et al, 2021 ([link](http://proceedings.mlr.press/v130/bosch21a.html))

[^8]: *Pick-and-Mix Information Operators for Probabilistic ODE Solvers*, Bosch et al, 2022 ([link](https://proceedings.mlr.press/v151/bosch22a.html))

[^9]: *Probabilistic ODE Solutions in Millions of Dimensions*, Krämer et al, 2022 ([link](https://proceedings.mlr.press/v162/kramer22b.html))

[^10]: *Fenrir: Physics-Enhanced Regression for Initial Value Problems*, Tronarp et al, 2022 ([link](https://proceedings.mlr.press/v162/tronarp22a.html))


Finally, you can also find a lot of information covering (most of) the topics above in the PN textbook:

[^11]: *Probabilistic Numerics*, Hennig et al, 2021 ([link](https://www.probabilistic-numerics.org/textbooks/))
"""

# ╔═╡ e0774fef-aae0-4e16-8791-8051d7ee5ea3
md"# Notebook Ingredients"

# ╔═╡ c439eba4-073c-4f26-b8b9-5c2e3aa072b3
TableOfContents()

# ╔═╡ d670b87f-63e3-47c7-9478-0d322d407c70
present_button()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Distributions = "~0.25.107"
ForwardDiff = "~0.10.36"
Optim = "~1.9.2"
Plots = "~1.40.2"
PlutoTeachingTools = "~0.2.14"
PlutoUI = "~0.7.58"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "4ffa67398dd5cbf1d57157be124e721792ce25aa"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0f748c81756f2e5e6854298f11ad8b2dfae6911a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.0"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "44691067188f6bd1b2289552a23e4b7572f4528d"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.9.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a4c43f59baa34011e303e76f5c8c91bf58415aaf"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "c0216e792f518b39b22212127d4a84dc31e4e386"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "c955881e3c981181362ae4088b35995446298b80"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.14.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "6cbbd4d241d7e6579ab354737f4dd95ca43946e1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "260fd2400ed2dab602a7c15cf10c1933c59930a2"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.5"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "0f4b5d62a88d8f59003e43c25a8a90de9eb76317"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.18"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7c302d7a5fec5214eb8a5a4c466dcf7a51fcf169"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.107"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "bfe82a708416cf00b73a3198db0859c82f741558"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.10.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "bc0c5092d6caaea112d3c8e3b238d61563c58d5f"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.23.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "ff38ba61beff76b8f4acad8ab0c97ef73bb670cb"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.9+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "3437ade7073682993e092ca570ad68a2aba26983"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.3"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a96d5c713e6aa28c242b0d25c1347e258d6541ab"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.3+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "359a1ba2e320790ddbe4ee8b4d54a305c0ea2aff"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.0+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "8e59b47b9dc525b70550ca082ce85bcd7f5477cd"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.5"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3336abae9a713d2210bb57ab484b1e065edd7d23"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.2+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "e9648d90370e2d0317f9518c9c6e0841db54a90b"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.31"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cad560042a7cc108f5a4c24ea1431a9221f22c1b"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.2"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dae976433497a2f841baadea93d27e68f1a12a97"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.39.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0a04a1318df1bf510beb2562cf90fb0c386f58c4"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.39.3+1"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "31e27f0b0bf0df3e3e951bfcc43fe8c730a219f6"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.4.5"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "af81a32750ebc831ee28bdaaba6e1067decef51e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.2"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3da7367955dcc5c54c1ba4d402ccdc09a1a3e046"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "d9b79c4eed437421ac4285148fcadf42e0700e89"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.9.4"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "3bdfa4fa528ef21287ef659a89d686e8a1bcb1a9"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.3"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "89f57f710cc121a7f32473791af3d6beefc59051"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.14"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "71a22244e352aa8c5f0f2adde4150f62368a3f2e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.58"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "12aa2d7593df490c407a3bbd8b86b8b515017f3e"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.14"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "71509f04d045ec714c4748c785a59045c3736349"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.7"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "3c793be6df9dd77a0cf49d80984ef9ff996948fa"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.19.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "532e22cf7be8462035d092ff21fada7527e2c488"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.6+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ac88fb95ae6447c8dda6a5503f3bafd496ae8632"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.6+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e678132f07ddb5bfa46857f0d7620fb9be675d3b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─dc3a7c1b-512c-49c4-a96d-838556e1d9c2
# ╟─e58eea83-277b-40c7-ac9d-5f3300e59ffe
# ╟─799233c0-97b9-4620-8356-16c41239a92b
# ╟─0ad15284-e678-4341-8278-18e423b77034
# ╟─9593dfa0-821b-45b2-ba23-03ba29f3eb79
# ╟─369bea9e-994f-4168-b6bf-ea16c60734bb
# ╟─0ea5e615-e988-4f6d-bf72-66a9fed1b809
# ╠═5b24cacf-921b-4b73-b6f5-355fdf9f8adb
# ╟─11204fa2-4594-4244-a9ac-b600450014e6
# ╟─e97cf1b4-b254-4c12-b014-ef6f80362a5f
# ╠═3848eed0-8aca-4aaa-9d82-011f0977bc29
# ╠═cec7a2ee-d308-49aa-93dc-0f5584c4c874
# ╠═a0653a41-87de-4c97-9812-f1112fbfb195
# ╠═eba65e7a-5b38-49bc-b81b-e878cc0d33bd
# ╟─da6dcaf7-f8eb-4dc9-a1d4-79b1d86007e9
# ╠═992e7173-f70e-41a1-9588-fb67b56a84fe
# ╟─5544ded8-904e-4e88-8038-bc1e26035279
# ╠═72430f2d-55d2-4c65-bc99-f808fb153451
# ╠═e41d5a46-48ef-4891-87f9-b6909ebbdbe1
# ╟─0c5021ff-7bef-4863-90c5-55a507451299
# ╠═4bf13155-904d-4f5e-842e-8fa7f18bb807
# ╟─0add1185-f307-4dfe-8114-4280d5622fde
# ╠═e607ba98-bc21-4b04-ab55-d41ecff19182
# ╟─95eb44df-af71-488b-9d97-cafe1294a02a
# ╠═d7e85fa4-ad26-4c19-b86d-8c8346d5727b
# ╟─6a087440-6077-4773-a01f-d39dfbc34874
# ╟─2a211405-4d7d-478d-bd2e-b53719817df7
# ╠═e11629dc-1bf3-4dbe-977b-b8ca567dc6df
# ╠═51dd874e-8a3e-4eed-97f2-148ad5a82efe
# ╟─25023dba-be97-4cfb-a880-2be8c6e3330d
# ╠═048bed3a-e7aa-4866-ac4f-d8990bc1b844
# ╠═572111f8-92b1-45c6-868c-8abaa163a92a
# ╠═a73a21d0-6c09-4815-b92d-319b38be9292
# ╠═fdc4f920-a35e-4fed-8012-bd7e1bf624ac
# ╟─f9773354-6b28-4fd8-9a20-c31803785354
# ╟─df25ff79-d301-4ff7-97c2-eeff64eb6873
# ╟─534ee2de-afe5-4a03-b914-9619a3594e47
# ╠═233cc2c2-cf5d-472f-a690-76cd7b5c4e12
# ╟─365dfbf2-9288-49ab-b702-da010b078d47
# ╟─e00e8289-999d-4209-bef0-2e9318148fe2
# ╠═8dd03abf-05ba-47ed-93f3-c3b5e0dee217
# ╟─f8fab074-a2be-4218-8e0f-f85535c1764e
# ╟─a1bc57e5-d3a7-4ee0-ac6a-204ea4ddb2ff
# ╟─2101bfaf-33bd-42a5-a836-c0c253b253e8
# ╠═d8551bc9-6139-42f5-ab2c-95ad47163a50
# ╟─f35114c9-a5a2-4754-a4f1-b25f32842c73
# ╟─bdb5dddf-1435-4537-aa08-ba5dd35b301d
# ╟─f2fa37c4-488f-4dd8-8987-98f12436550a
# ╟─7b0e453c-e746-4995-84bf-282a65ee8d78
# ╠═436ee1d8-d9c1-4b49-a171-a806954cea80
# ╠═3348963c-4aa9-4d29-86ff-ebeb0351e3be
# ╠═2d4fb782-4cd4-4ab2-a28a-b6b0aff40622
# ╠═8ccb9b17-7a9a-4562-a217-1d0c462d62a0
# ╠═4cf6b346-bdbe-4095-830a-080e64d9e307
# ╟─bec9cd97-182b-411d-ba08-3072dccf0c4f
# ╠═c7c73c79-757b-4cb0-bca7-ff92a27d3aa3
# ╟─612a5c94-0969-4f3c-8c90-fa14e0624eb2
# ╠═15dcaa60-bb28-489e-9c8d-7edb1f62d904
# ╠═b014b770-422c-45b7-bfea-6f0c8cb744e7
# ╠═e316d446-f64f-4edc-80b0-5be5ee656ece
# ╟─a0440b5d-3b33-49b6-b132-dc1e8f02e66c
# ╟─8527e16e-8641-499a-bdf6-f3d87b0241b0
# ╟─519d00c3-ed43-40e1-a360-e95214552ade
# ╟─e8a99120-0e7f-43b4-bf9c-4e20b36ba291
# ╟─1a54e048-2153-4755-9a96-9b53ca10363a
# ╠═9d5d91b8-cd68-442d-82ec-b8571e8ab50c
# ╟─dca16cfc-1914-4eea-9356-a910b01fa299
# ╟─6b4b0659-026f-46d3-be2f-2617570506a5
# ╟─70d5b269-852e-4957-ad6e-5d00c29f3f7a
# ╟─f4c6d0b5-fc3a-4b47-9063-cdfe302e1148
# ╟─0a98a145-e3e9-4b31-ab57-5a7a928115f4
# ╟─b4a488cc-3d97-48ea-bc77-be8a2177682a
# ╠═99e1701c-94ba-4864-b30d-2067db333429
# ╠═a2443f6c-8dbb-47c9-a083-866cf75b54e1
# ╠═c06f27bd-8c14-4d0b-aaee-63f951e0c7b3
# ╠═28c262d4-899f-467d-8bfa-7c7d2b2f79cc
# ╠═7df7f8dd-3189-40cc-9320-b78646360c4f
# ╟─27415e7d-8a6b-448a-95ea-b663f260a288
# ╠═d5e8620b-9604-4f12-8bc6-2434bffec0c0
# ╠═36a2dfe7-0488-4b81-a308-97f0557773cd
# ╠═fe06c623-2c85-4f0a-b8b3-c131c6f48477
# ╟─afd8a4ab-1b54-4d44-9255-5d51c3e72815
# ╠═bb8302f5-4a5e-47f3-ab21-32e7f057f8c5
# ╠═1128efbe-bcd4-4ae7-b003-950528d272a7
# ╠═ac85113c-3125-480f-ad98-80100a8d0ff9
# ╟─3abe5e98-a2de-4d57-b2f8-b3ce07d887de
# ╠═1e9bb574-16ac-4ad9-aa7a-2c2cd5ae6b1b
# ╠═8e6cec5d-6c5f-4011-a79f-964b21368972
# ╟─73817e2a-ad4b-49c8-852c-3f75d76dbdcb
# ╠═d403bef1-bce5-4cfc-8ac0-52996010ad2f
# ╠═fbdf49b8-0d36-4d9d-8e13-e8fed2f86171
# ╟─0331404c-b18d-4ea7-b948-9d7138c6d2f8
# ╟─1168a27e-8c2f-49fb-9857-3dba65a3bf03
# ╟─11356108-c1dd-49c5-81ab-706936c0adf4
# ╟─ffc68588-f673-4382-94a1-8e4b212124a0
# ╟─9fa7ca21-c6cb-4de0-b3e4-c2e466fe0fb6
# ╟─c2428a5f-4553-4e07-bc8c-0c9671bea8b0
# ╟─5be999d6-70ec-40d7-9d33-37ca8bea155e
# ╟─cc1aa6b3-2c74-4e1e-90bd-f4c023d0a0c3
# ╟─cc259a06-e028-4a42-a07c-acdd0cbabab4
# ╠═4d7e17e4-4610-4ad1-8e2c-fad89dfa0e3c
# ╟─6ed274e4-dbc0-4629-a0d3-3d838876cb35
# ╟─52f23a1c-516c-4c0c-bc37-1af122a3900e
# ╠═13b9921d-6f6b-462e-90d4-725892d40707
# ╟─576e9138-8e8b-4751-b615-08c6a73e8dfb
# ╠═34a79302-be2a-4d6b-a509-753fc45abbf1
# ╟─55a6a562-1f81-442a-a049-fdea157d023b
# ╟─46f5dc2f-dc36-44c5-b0b3-a8bb063f0283
# ╟─69660525-2246-4a7a-9e3c-c99d1058f7a8
# ╟─9be29ad7-e1b1-4b4e-9b8a-5434b0ee21a8
# ╟─783e96cb-ce53-493f-9dcc-51ab620fca7f
# ╠═f80a5591-1387-40ee-8dad-c99c7df01fa0
# ╟─226b69db-cb33-477b-acbe-7e29a7776806
# ╟─fc15ff6a-0090-4c58-bc06-f994b4a42b2c
# ╠═d7b02583-b3e4-44db-96e8-0d8373290c54
# ╟─e6a691db-1ed0-4665-8f93-1593f170ee89
# ╟─e5da35ae-192a-4a38-ade1-6fe13f53ba07
# ╟─906aee43-d874-409b-b52a-35f40701dd58
# ╟─dbda9377-2e05-44b4-bd50-ec90f2ebcb0d
# ╟─e62c343e-616e-434b-af3d-c2d730c5bc5a
# ╠═3c511004-5e21-4975-88e1-a1e0b885225f
# ╟─432f9128-f899-438e-98ff-a9406b2745d2
# ╟─36fe7a32-1f31-471b-8c26-1fac7aef860d
# ╟─aee34cb8-f6c5-42fd-901e-a1f52c3567b1
# ╟─c4e594b7-9d4d-4b4f-add3-707dc3848158
# ╟─f45937d9-d127-42ce-b89e-9cd446d5bda7
# ╠═c8674da0-c7fa-4c5c-9e7e-c976a66d9c34
# ╠═62bad736-b3a0-4bfa-a55b-242bb7bc2d81
# ╟─81e0051f-70f0-4688-88e1-7ecb9c536cc9
# ╟─fe0f9312-83ca-4847-9384-5f51998769f8
# ╠═d339404a-2ae5-463b-b214-001aafd9a0ef
# ╟─4b53a0d2-9ee4-4c30-8dde-3ba7415ff16a
# ╠═9f2540af-aab9-4faa-823a-84fc00c239cb
# ╠═7204380c-5302-4763-b46b-da4e98f17006
# ╟─c47f009d-fad7-4507-939b-403cfe5c42df
# ╟─bd685522-ab71-4d41-bd93-d980695fd99f
# ╟─2fa961ad-8c37-4c85-8111-842ae292d9cb
# ╟─332ad601-66d0-45c4-a3c6-38d3285d8726
# ╠═aaaba38a-0c06-41e1-8f0e-1ce39af20377
# ╠═a6f27549-1584-4fee-a503-ad0bef8dc7a7
# ╟─7a475a2b-0e15-41d2-8fdf-4bfead53646c
# ╟─d83e49e6-3514-4de7-a96d-73310abf46e6
# ╟─04c3d31e-b5c2-4e99-820d-0895ebdde680
# ╠═167a7221-452a-4b83-a9fb-950b6feba5bf
# ╟─8a0b82ac-ec71-492e-91ca-6d4778bc8eeb
# ╠═9ba6b83b-4c1f-482f-b98f-11239ee7a48c
# ╟─d271a1ff-5432-4503-9c02-9954d40989db
# ╟─4f062a0e-dd9a-4994-83e5-921cc248f210
# ╠═0eb1590f-b7a0-4e00-aa4b-6de05607baf5
# ╠═3d198b59-0654-4ade-a82c-c047f96a5788
# ╟─674b8281-3df7-43b2-b412-65c94e8d2aa7
# ╟─dc490607-2ca4-435b-884d-57e808b04d5e
# ╠═bb268636-14b1-4235-825e-eb2c409e1838
# ╟─01978e85-7633-42ff-accd-8313a80ec632
# ╠═ba3ac875-3c83-4ee5-a88b-5f44b2e32d94
# ╟─a118e1ac-e499-4d57-bcd7-88c10c919b2b
# ╠═95e097b2-2ba9-4b33-a7c2-9f94d70dc655
# ╟─b6b7facb-9aa3-4423-a82e-56264f4cb6ad
# ╟─4912bea0-4af4-49f9-9ec5-a356a7d94ef8
# ╟─e0a58a9e-e7a7-4546-99be-7f55cce2bbdf
# ╠═04ac021b-1bcb-485d-aef7-a666f3c54b36
# ╟─f152c95a-6b69-4821-b522-8cc034d95200
# ╟─61ba4734-a538-4e1d-b770-3b50c839a2be
# ╟─779cdcfa-cadc-4986-bf8c-6e4d81e2c27d
# ╟─ea5ed6a6-bc35-402d-b56d-4179d6a9d923
# ╟─5ff96ed0-fbcf-4947-a878-cae942dc5550
# ╟─e804d104-3811-46f2-b014-d5931ab61f23
# ╟─4f2e6d01-8673-4b4e-a4c5-fe13c744e0f9
# ╟─284cfdd1-6d25-47fb-ba5d-e0a563197ce3
# ╟─9a815e39-65b8-4bd1-bb98-6e241f11f24d
# ╟─12ac610e-0709-43b6-8ee2-3c0befaccc48
# ╠═5f93f93a-f6df-4247-ab71-c8430e1b12ac
# ╟─b882cc9a-e52c-400b-84d3-aa77c83b559e
# ╟─1f9b1229-0b5d-4bb8-8edb-dca3f7fce44e
# ╠═a98eb4ad-4606-4f38-874d-c559ad7dfef1
# ╟─a0e66a7c-3c44-4ecb-9718-3334ed16acef
# ╟─d2caedba-3b90-4a2d-9167-985675dd77da
# ╟─e177c47e-2d16-4f71-946f-09e9d7a3d8b8
# ╟─e61b237d-e9ce-4ca2-8cdb-1e1fbab1a637
# ╟─67c90ed2-7035-4152-8b49-491c5c55bf72
# ╟─0f0ba357-bd35-4bc3-93e7-c0e7d69768e5
# ╟─9e2df89e-7faf-48a1-90c3-55e44b2d02fc
# ╠═ad5033a9-166b-44f6-99eb-03b64b38adbb
# ╠═cd63a01a-d24e-4786-822f-64356c994f28
# ╠═0b820d93-da41-43d3-9f57-b6d5cd1a0756
# ╠═9ed6966f-2934-4f92-a44e-da5a5150481d
# ╠═5b194301-c8b8-4024-9ac1-76e9593db32f
# ╠═83299d45-bb36-4f72-9a02-4682bdf91020
# ╟─ce4b0038-cd33-43a7-b3a8-e53c8f23eb5b
# ╟─af110785-8412-43d2-8d7a-f45c021ff7e4
# ╟─d8024d4a-42c0-4c2f-b8ee-5aeb19302100
# ╠═f13ec44f-307a-490f-a5e8-b642e48042ee
# ╠═dc0a8b43-00f9-4731-8b3e-289065e700f1
# ╠═e3df901b-bf88-4031-a25c-e01739284085
# ╠═244b8480-d74a-4734-a500-2d439af7546d
# ╟─656d511d-1f37-4288-a479-90112f84cc19
# ╟─8f3d7317-f8af-4aa5-b9ab-67456d22156e
# ╠═76706c29-fcec-4475-a766-fa32b0434a5c
# ╠═3a663281-c595-4233-8304-fb3daa1e3667
# ╠═f975608b-7271-4139-874e-08506e8d6375
# ╠═1f4cbb92-1478-4999-9d47-e56678e8eb7b
# ╟─81f18197-39f0-4c25-9e82-6d0071b99aee
# ╠═01055f56-a49a-470f-a021-e67e7f9efb87
# ╟─f01e08dc-a621-4a9a-b355-b78555d63216
# ╟─d37ade4a-cb4e-42f4-b3ed-997cd9e40d53
# ╟─c21f17bf-e4e2-4b73-b6d8-96e53dd576f6
# ╟─c09f1297-b309-4043-8c3f-5726bdaf203b
# ╟─b55f6094-d95a-463b-8afb-e5e68a704140
# ╟─c771995f-2ef9-46ed-98da-3545e34748ce
# ╟─eba73262-bc98-4a9f-83db-c3ade9c6d09f
# ╟─e0774fef-aae0-4e16-8791-8051d7ee5ea3
# ╠═5a3b056e-27fd-4828-a573-9a1e95a546eb
# ╠═c439eba4-073c-4f26-b8b9-5c2e3aa072b3
# ╠═d670b87f-63e3-47c7-9478-0d322d407c70
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
