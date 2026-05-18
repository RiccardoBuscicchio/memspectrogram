"""
Compare theoretical and empirical Fourier-domain covariances for an AR(p) process.

The script:
1. Builds an AD-friendly `MESAPSD` model for a user-defined AR process.
2. Computes the theoretical covariance `Cov(X(f_i), X(f_j))` with
   `frequency_covariance`, which uses automatic differentiation.
3. Estimates the same covariance matrix from many simulated realisations.
4. Saves a comparison plot to `examples/ar_covariance_ad.png`.

Run from the repository root:

    julia --project=. examples/ar_covariance_ad.jl

All parameters can be supplied via command-line flags or a TOML config file:

    julia --project=. examples/ar_covariance_ad.jl \
        --config examples/configs/ar_covariance_ad.toml

"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "Memspectrum.jl"))
using .Memspectrum

using ArgParse
using LinearAlgebra
using Random
using Statistics
using TOML
using Plots

function parse_commandline()
    s = ArgParseSettings(
        description = "Compare theoretical and empirical AR(p) Fourier covariances"
    )
    @add_arg_table! s begin
        "--config"
            help     = "Path to a TOML configuration file (optional)"
            arg_type = String
            default  = nothing
        "--N"
            help     = "Number of samples in each realisation"
            arg_type = Int
            default  = nothing
        "--dt"
            help     = "Sampling interval (seconds)"
            arg_type = Float64
            default  = nothing
        "--P"
            help     = "Innovation variance"
            arg_type = Float64
            default  = nothing
        "--a1"
            help     = "First AR coefficient"
            arg_type = Float64
            default  = nothing
        "--a2"
            help     = "Second AR coefficient"
            arg_type = Float64
            default  = nothing
        "--burnin"
            help     = "Number of burn-in samples discarded before each realisation"
            arg_type = Int
            default  = nothing
        "--realizations"
            help     = "Number of Monte-Carlo realisations"
            arg_type = Int
            default  = nothing
        "--seed"
            help     = "Random seed"
            arg_type = Int
            default  = nothing
        "--frequency_step"
            help     = "Spacing between selected DFT bins"
            arg_type = Int
            default  = nothing
        "--max_bin"
            help     = "Largest DFT bin included in the comparison"
            arg_type = Int
            default  = nothing
    end
    return parse_args(s)
end

function empirical_fourier_covariance(model::MESAPSD, dt::Float64,
                                      frequencies::AbstractVector{<:Real};
                                      realizations::Int,
                                      burnin::Int,
                                      seed::Int)
    Random.seed!(seed)
    n_freq = length(frequencies)
    samples = Matrix{ComplexF64}(undef, realizations, n_freq)
    sigma = sqrt(model.P)

    for r in 1:realizations
        x_full = zeros(Float64, model.N + burnin)
        for t in eachindex(x_full)
            value = sigma * randn()
            for k in 1:min(model.p, t - 1)
                value -= model.a_k[k + 1] * x_full[t - k]
            end
            x_full[t] = value
        end
        x = x_full[burnin+1:end]
        for (j, frequency) in enumerate(frequencies)
            coeff = zero(ComplexF64)
            for n in eachindex(x)
                coeff += dt * x[n] * cis(-2π * frequency * dt * (n - 1))
            end
            samples[r, j] = coeff
        end
    end

    sample_mean = vec(mean(samples; dims=1))
    centered = samples .- sample_mean'
    return centered' * centered / (realizations - 1)
end

args = parse_commandline()
cfg = args["config"] === nothing ? Dict{String,Any}() : TOML.parsefile(args["config"])
_get(key, default) = args[key] !== nothing ? args[key] :
                     haskey(cfg, key)       ? cfg[key]  : default

const N = _get("N", 64)
const DT = _get("dt", 1.0 / 128.0)
const P = _get("P", 1.0)
const A1 = _get("a1", -1.5)
const A2 = _get("a2", 0.9)
const BURNIN = _get("burnin", 256)
const REALIZATIONS = _get("realizations", 4000)
const SEED = _get("seed", 21)
const FREQUENCY_STEP = _get("frequency_step", 4)
const MAX_BIN = _get("max_bin", 20)

model = MESAPSD(P, [1.0, A1, A2], N)
bins = collect(0:FREQUENCY_STEP:MAX_BIN)
freqs = bins ./ (N * DT)

theoretical = frequency_covariance(model, DT; frequencies=freqs, burnin=BURNIN)
empirical = empirical_fourier_covariance(
    model,
    DT,
    freqs;
    realizations = REALIZATIONS,
    burnin = BURNIN,
    seed = SEED,
)

diag_theoretical = real.(diag(theoretical))
diag_empirical = real.(diag(empirical))
rel_diag_error = maximum(abs.(diag_empirical .- diag_theoretical) ./ diag_theoretical)

println("AR(2) covariance comparison")
println("  N=$N, dt=$DT s, P=$P, a = [$A1, $A2]")
println("  burn-in samples: $BURNIN")
println("  realizations:    $REALIZATIONS")
println("  max rel. diag error: $(round(rel_diag_error * 100, digits=2)) %")

bin_labels = string.(bins)
heat_theory = heatmap(
    bin_labels,
    bin_labels,
    abs.(theoretical);
    title = "Theoretical |Cov(X(fi), X(fj))|",
    xlabel = "DFT bin",
    ylabel = "DFT bin",
    colorbar_title = "Magnitude",
    aspect_ratio = :equal,
)

heat_empirical = heatmap(
    bin_labels,
    bin_labels,
    abs.(empirical);
    title = "Empirical |Cov(X(fi), X(fj))|",
    xlabel = "DFT bin",
    ylabel = "DFT bin",
    colorbar_title = "Magnitude",
    aspect_ratio = :equal,
)

diag_plot = plot(
    bins,
    diag_theoretical;
    label = "Theoretical",
    lw = 2,
    marker = :circle,
    xlabel = "DFT bin",
    ylabel = "Variance",
    title = "Diagonal covariance entries",
)
plot!(diag_plot, bins, diag_empirical; label = "Empirical", lw = 2, marker = :diamond)

plt = plot(heat_theory, heat_empirical, diag_plot; layout = @layout([a b; c]), size = (1200, 900), dpi = 150)
out_path = joinpath(@__DIR__, "ar_covariance_ad.png")
savefig(plt, out_path)
println("Plot saved to $out_path")
display(plt)
