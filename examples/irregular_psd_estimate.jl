"""
PSD estimate from irregularly sampled data with the irregular Burg algorithm
(Bos, de Waele & Broersen, IEEE Trans. Instrum. Meas. 51, 1289, 2002).

The script:
1. Generates a densely sampled AR(2) process with a known analytical spectrum.
2. Randomly discards most samples, leaving irregular observation times with
   roughly exponential gaps (mean rate λ).
3. Fits AR models directly to the irregular data for several slot widths Δτ
   and compares the resulting PSDs with the true spectrum.
4. Saves the plot to `examples/irregular_psd_estimate.png`.

Run from the repository root:

    julia --project=. examples/irregular_psd_estimate.jl
    julia --project=. examples/irregular_psd_estimate.jl --config examples/configs/irregular_psd_estimate.toml
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "Memspectrum.jl"))
using .Memspectrum

using ArgParse
using TOML
using Random
using Statistics
using Plots

# ---------------------------------------------------------------------------
# Argument parsing (command-line + TOML config file)
# ---------------------------------------------------------------------------

function parse_commandline()
    s = ArgParseSettings(
        description = "Irregular-sampling AR(2) PSD estimate with Memspectrum"
    )
    @add_arg_table! s begin
        "--config"
            help     = "Path to a TOML configuration file (optional)"
            arg_type = String
            default  = nothing
        "--N_dense"
            help     = "Number of samples of the dense underlying series"
            arg_type = Int
            default  = nothing
        "--keep_fraction"
            help     = "Probability of keeping each dense sample"
            arg_type = Float64
            default  = nothing
        "--dt_factor"
            help     = "AR grid spacing T in units of the mean interval 1/λ"
            arg_type = Float64
            default  = nothing
        "--seed"
            help     = "Random seed"
            arg_type = Int
            default  = nothing
        "--penalty"
            help     = "Penalty factor α of AICirreg"
            arg_type = Float64
            default  = nothing
    end
    return parse_args(s)
end

args = parse_commandline()

cfg = Dict{String,Any}()
if args["config"] !== nothing
    cfg = TOML.parsefile(args["config"])
end

_get(key, default) = args[key] !== nothing ? args[key] :
                     haskey(cfg, key)       ? cfg[key]  : default

const N_DENSE   = _get("N_dense",       400_000)
const KEEP      = _get("keep_fraction", 0.05)
const DT_FACTOR = _get("dt_factor",     1.0)
const SEED      = _get("seed",          42)
const PENALTY   = _get("penalty",       4.0)

# Dense AR(2): resonance at F0 (cycles per dense sample), pole radius R
const F0 = 0.01
const R  = 0.995
const AR_COEFF = [-2R * cos(2π * F0), R^2]

# ---------------------------------------------------------------------------
# 1.  Dense AR(2) series, then random thinning
# ---------------------------------------------------------------------------

Random.seed!(SEED)

x_dense = zeros(N_DENSE)
for t in 3:N_DENSE
    x_dense[t] = -AR_COEFF[1] * x_dense[t-1] - AR_COEFF[2] * x_dense[t-2] + randn()
end

keep  = rand(N_DENSE) .< KEEP
times = Float64.(findall(keep))
x     = x_dense[keep]

mean_interval = (times[end] - times[1]) / (length(times) - 1)
T = DT_FACTOR * mean_interval

println("Irregular series: $(length(x)) observations, mean interval 1/λ = ",
        round(mean_interval, digits=2))
println("AR grid spacing T = $(round(T, digits=2))  →  f_max = $(round(0.5 / T, sigdigits=3))")

# ---------------------------------------------------------------------------
# 2.  Irregular Burg fits for several slot widths
# ---------------------------------------------------------------------------

slot_factors = [0.5, 0.25]
fits = Dict{Float64, Any}()
for sf in slot_factors
    m = MESA()
    solve!(m, times, x; dt=T, slot_width=sf * mean_interval, penalty=PENALTY)
    fits[sf] = m
    println("Δτ = $(sf)/λ:  order p = $(m.p),  products N_p = ",
            m.n_products[1:min(end, 8)])
end

# ---------------------------------------------------------------------------
# 3.  True one-sided spectrum of the dense process (unit-spaced samples)
# ---------------------------------------------------------------------------

function ar_psd_onesided(f, a, sigma2, dt)
    z = exp.(-2π * im .* f .* dt)
    den = 1 .+ a[1] .* z .+ a[2] .* z .^ 2
    return 2 * dt * sigma2 ./ abs2.(den)
end

f_grid, _ = spectrum(fits[0.5], T; onesided=true)
psd_true  = ar_psd_onesided(f_grid, AR_COEFF, 1.0, 1.0)

# ---------------------------------------------------------------------------
# 4.  Plot
# ---------------------------------------------------------------------------

plt = plot(f_grid[2:end], psd_true[2:end];
           xscale = :log10, yscale = :log10,
           label = "True AR(2) spectrum", lw = 2, ls = :dash, color = :black,
           xlabel = "Frequency", ylabel = "One-sided PSD",
           title = "Burg for irregularly sampled data\n" *
                   "($(length(x)) observations, T = $(round(T, digits=1)))",
           legend = :bottomleft, size = (800, 500), dpi = 150)

colors = Dict(0.5 => :royalblue, 0.25 => :seagreen)
for sf in slot_factors
    _, psd = spectrum(fits[sf], T; onesided=true)
    plot!(plt, f_grid[2:end], psd[2:end];
          label = "Δτ = $(sf)/λ  (p = $(fits[sf].p))", lw = 2,
          color = colors[sf], alpha = 0.85)
end

out_path = joinpath(@__DIR__, "irregular_psd_estimate.png")
savefig(plt, out_path)
println("\nPlot saved to $out_path")
