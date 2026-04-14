"""
Toy-model Memgram from a non-stationary synthetic time series.

The script:
1. Generates a time series whose AR(2) peak frequency sweeps linearly with time
   (two concatenated AR(2) blocks with different coefficients).
2. Computes the Memgram by fitting an AR model on short overlapping segments.
3. Saves the Memgram image to `examples/toy_spectrogram.png`.

Run from the repository root:

    julia examples/toy_spectrogram.jl

All parameters can be supplied via command-line flags or a TOML config file:

    julia examples/toy_spectrogram.jl --config examples/configs/toy_spectrogram.toml
    julia examples/toy_spectrogram.jl --seg_len 256 --overlap 0.5

"""

# Make the src directory visible so we can load Memspectrum without installing it.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "Memspectrum.jl"))
using .Memspectrum

using ArgParse
using TOML
using Random
using Plots

# ---------------------------------------------------------------------------
# Argument parsing (command-line + TOML config file)
# ---------------------------------------------------------------------------

function parse_commandline()
    s = ArgParseSettings(description = "Toy Memgram from non-stationary AR(2)")
    @add_arg_table! s begin
        "--config"
            help     = "Path to a TOML configuration file (optional)"
            arg_type = String
            default  = nothing
        "--fs"
            help     = "Sampling rate (Hz)"
            arg_type = Float64
            default  = nothing
        "--n_total"
            help     = "Total number of samples"
            arg_type = Int
            default  = nothing
        "--sigma"
            help     = "Innovation standard deviation"
            arg_type = Float64
            default  = nothing
        "--seed"
            help     = "Random seed"
            arg_type = Int
            default  = nothing
        "--seg_len"
            help     = "Samples per spectrogram segment"
            arg_type = Int
            default  = nothing
        "--overlap"
            help     = "Fractional overlap between segments (0–1)"
            arg_type = Float64
            default  = nothing
        "--optimisation_method"
            help     = "AR order-selection criterion"
            arg_type = String
            default  = nothing
        "--method"
            help     = "Burg algorithm variant (Fast or Standard)"
            arg_type = String
            default  = nothing
    end
    return parse_args(s)
end

args = parse_commandline()
cfg  = Dict{String,Any}()
if args["config"] !== nothing
    cfg = TOML.parsefile(args["config"])
end

_get(key, default) = args[key] !== nothing ? args[key] :
                     haskey(cfg, key)       ? cfg[key]  : default

const FS      = _get("fs",      256.0)
const DT      = 1.0 / FS
const N_TOTAL = _get("n_total", 8_192)
const SIGMA   = _get("sigma",   1.0)
const SEED    = _get("seed",    0)
const SEG_LEN = _get("seg_len", 512)
const OVERLAP = _get("overlap", 0.75)
const OPT_METHOD = _get("optimisation_method", "FPE")
const METHOD     = _get("method",              "Fast")

# Two AR(2) blocks, each half of the signal.
# Block 1: resonance near  30 Hz  (a1=-1.5, a2=0.9)
# Block 2: resonance near  80 Hz  (a1=-0.5, a2=0.9)
const COEFF_A = [-1.5, 0.9]
const COEFF_B = [-0.5, 0.9]

# ---------------------------------------------------------------------------
# 1.  Generate a non-stationary AR(2) time series
# ---------------------------------------------------------------------------

Random.seed!(SEED)

x    = zeros(N_TOTAL)
half = N_TOTAL ÷ 2

x[1] = SIGMA * randn()
x[2] = SIGMA * randn()
for t in 3:N_TOTAL
    c = t <= half ? COEFF_A : COEFF_B
    x[t] = -c[1] * x[t-1] - c[2] * x[t-2] + SIGMA * randn()
end

println("Generated non-stationary AR(2) time series: N=$N_TOTAL samples, dt=$DT s")
println("  Block 1 (t=0..$(half*DT) s):  a1=$(COEFF_A[1]), a2=$(COEFF_A[2])")
println("  Block 2 (t=$(half*DT)..$(N_TOTAL*DT) s): a1=$(COEFF_B[1]), a2=$(COEFF_B[2])")

# ---------------------------------------------------------------------------
# 2.  Compute MESA spectrogram
# ---------------------------------------------------------------------------

println("\nComputing Memgram  (segment_length=$SEG_LEN, overlap=$(Int(OVERLAP*100)) %) …")
t_centers, f_grid, psd_mat = memgram(
    x, DT;
    segment_length      = SEG_LEN,
    overlap             = OVERLAP,
    optimisation_method = OPT_METHOD,
    method              = METHOD,
    verbose             = true,
)

println("\nMemgram: $(size(psd_mat, 2)) segments × $(length(f_grid)) frequency bins")
println("  Time range:  $(round(t_centers[1],  digits=2)) – $(round(t_centers[end], digits=2)) s")
println("  Freq range:  $(round(f_grid[1],     digits=2)) – $(round(f_grid[end],    digits=2)) Hz")

# ---------------------------------------------------------------------------
# 3.  Plot and save
# ---------------------------------------------------------------------------

plt = plot_spectrogram(
    t_centers, f_grid, psd_mat;
    title = "Memgram  (segment=$(SEG_LEN) samples, $(Int(OVERLAP*100)) % overlap)\n" *
            "AR(2) block 1: a1=$(COEFF_A[1])  →  block 2: a1=$(COEFF_B[1])",
    size  = (960, 500),
    dpi   = 150,
)

# Mark the transition between the two blocks with a vertical dashed line.
vline!(plt, [half * DT];
       lw      = 2,
       ls      = :dash,
       color   = :white,
       label   = "Regime change",
       legend  = :topright,
)

out_path = joinpath(@__DIR__, "toy_spectrogram.png")
savefig(plt, out_path)
println("\nMemgram saved to $out_path")
