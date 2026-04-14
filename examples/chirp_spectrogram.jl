"""
Toy-model Memgram from a chirping sinusoidal signal.

The script:
1. Generates a sinusoidal chirp whose instantaneous frequency sweeps linearly
   from F_START to F_END over the full duration, with added Gaussian noise.
2. Computes the Memgram on overlapping short segments.
3. Overlays the theoretical instantaneous frequency as a dashed white line.
4. Saves the Memgram image to `examples/chirp_spectrogram.png`.

Run from the repository root:

    julia examples/chirp_spectrogram.jl

All parameters can be supplied via command-line flags or a TOML config file:

    julia examples/chirp_spectrogram.jl --config examples/configs/chirp_spectrogram.toml
    julia examples/chirp_spectrogram.jl --f_start 10 --f_end 200

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
    s = ArgParseSettings(description = "Memgram of a linear chirp signal")
    @add_arg_table! s begin
        "--config"
            help     = "Path to a TOML configuration file (optional)"
            arg_type = String
            default  = nothing
        "--fs"
            help     = "Sampling rate (Hz)"
            arg_type = Float64
            default  = nothing
        "--t"
            help     = "Total duration (seconds)"
            arg_type = Float64
            default  = nothing
        "--sigma"
            help     = "Additive noise amplitude"
            arg_type = Float64
            default  = nothing
        "--seed"
            help     = "Random seed"
            arg_type = Int
            default  = nothing
        "--f_start"
            help     = "Chirp start frequency (Hz)"
            arg_type = Float64
            default  = nothing
        "--f_end"
            help     = "Chirp end frequency (Hz)"
            arg_type = Float64
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

const FS      = _get("fs",      512.0)
const DT      = 1.0 / FS
const T       = _get("t",       16.0)
const N_TOTAL = round(Int, T * FS)
const SIGMA   = _get("sigma",   0.25)
const SEED    = _get("seed",    7)
const F_START = _get("f_start", 5.0)
const F_END   = _get("f_end",   120.0)
const SEG_LEN = _get("seg_len", 512)
const OVERLAP = _get("overlap", 0.875)
const OPT_METHOD = _get("optimisation_method", "FPE")
const METHOD     = _get("method",              "Fast")

# ---------------------------------------------------------------------------
# 1.  Generate a linear chirp signal with additive noise
#     φ(t) = 2π * (F_START * t + (F_END - F_START) / (2 * T) * t²)
#     x(t) = sin(φ(t)) + σ * noise(t)
# ---------------------------------------------------------------------------

Random.seed!(SEED)

t_vec = collect((0:N_TOTAL-1) .* DT)
phase = @. 2π * (F_START * t_vec + (F_END - F_START) / (2 * T) * t_vec^2)
x     = sin.(phase) .+ SIGMA .* randn(N_TOTAL)

println("Generated linear chirp: N=$N_TOTAL samples, dt=$DT s")
println("  Frequency sweep: $(F_START) → $(F_END) Hz over $(T) s")
println("  Noise amplitude: σ=$SIGMA")

# ---------------------------------------------------------------------------
# 2.  Compute MESA spectrogram
# ---------------------------------------------------------------------------

println("\nComputing Memgram  (segment_length=$SEG_LEN, overlap=$(round(Int, OVERLAP*100)) %) …")
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
    title = "Memgram – linear chirp  ($(F_START) → $(F_END) Hz)\n" *
            "segment=$(SEG_LEN) samples, $(round(Int, OVERLAP*100)) % overlap, σ_noise=$(SIGMA)",
    size  = (960, 500),
    dpi   = 150,
)

# Overlay the theoretical instantaneous frequency of the chirp.
f_inst = @. F_START + (F_END - F_START) / T * t_centers
plot!(plt, t_centers, f_inst;
      lw    = 2,
      ls    = :dash,
      color = :white,
      label = "Theoretical f(t)",
      legend = :topleft,
)

out_path = joinpath(@__DIR__, "chirp_spectrogram.png")
savefig(plt, out_path)
println("\nMemgram saved to $out_path")
