"""
Memgram of a quadratic-chirp signal (512-sample segments).

The script:
1. Generates a time series of exactly 16 × 2048 = 32768 samples consisting of
   a quadratic chirp (instantaneous frequency sweeping as f(t) = F_START + α·t²)
   plus additive Gaussian noise.
2. Times the `memgram` call (warm-up run first, then N_REPS timed runs).
3. Overlays the theoretical instantaneous-frequency curve on the spectrogram.
4. Saves the Memgram image to `examples/quadratic_chirp_spectrogram.png`.

Run from the repository root:

    julia examples/quadratic_chirp_spectrogram.jl

All parameters can be supplied via command-line flags or a TOML config file:

    julia examples/quadratic_chirp_spectrogram.jl --config examples/configs/quadratic_chirp_spectrogram.toml
    julia examples/quadratic_chirp_spectrogram.jl --f_start 5 --f_end 200

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
    s = ArgParseSettings(description = "Memgram of a quadratic chirp signal")
    @add_arg_table! s begin
        "--config"
            help     = "Path to a TOML configuration file (optional)"
            arg_type = String
            default  = nothing
        "--fs"
            help     = "Sampling rate (Hz)"
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
            help     = "Samples per spectrogram segment (default: 4096)"
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
        "--n_total"
            help     = "Total number of samples (default: 16 × 2048 = 32768)"
            arg_type = Int
            default  = nothing
        "--n_reps"
            help     = "Number of timing repetitions"
            arg_type = Int
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

# Total length defaults to 16 * 2048 = 32 768 samples; override with --n_total.
const N_TOTAL    = _get("n_total", 16 * 2048)

const FS      = _get("fs",      512.0)
const DT      = 1.0 / FS
const T       = N_TOTAL * DT                  # total duration in seconds
const SIGMA   = _get("sigma",   0.25)
const SEED    = _get("seed",    42)
const F_START = _get("f_start", 5.0)
const F_END   = _get("f_end",   200.0)
const SEG_LEN = _get("seg_len", 512)
const OVERLAP = _get("overlap", 0.875)
const OPT_METHOD = _get("optimisation_method", "FPE")
const METHOD     = _get("method",              "Fast")
const N_REPS     = _get("n_reps", 5)

# ---------------------------------------------------------------------------
# 1.  Generate a quadratic-chirp signal with additive noise
#
#   Instantaneous frequency:
#       f(t) = F_START + (F_END - F_START) / T² · t²
#
#   Phase (integral of 2π·f(t)):
#       φ(t) = 2π · [ F_START · t  +  (F_END - F_START) / (3 · T²) · t³ ]
#
#   Signal:
#       x(t) = sin(φ(t)) + σ · noise(t)
# ---------------------------------------------------------------------------

Random.seed!(SEED)

t_vec = collect((0:N_TOTAL-1) .* DT)
alpha = (F_END - F_START) / T^2               # quadratic sweep coefficient
phase = @. 2π * (F_START * t_vec + alpha / 3 * t_vec^3)
x     = sin.(phase) .+ SIGMA .* randn(N_TOTAL)

println("Generated quadratic chirp: N=$N_TOTAL samples  (16 × 2048),  dt=$(round(DT, digits=6)) s")
println("  Duration:              T=$(round(T, digits=3)) s")
println("  Frequency sweep:       f(t) = $(F_START) + $(round(alpha, sigdigits=4))·t²  Hz")
println("  f(0)=$(F_START) Hz  →  f(T)=$(F_END) Hz")
println("  Noise amplitude:       σ=$SIGMA")

# ---------------------------------------------------------------------------
# 2.  Warm-up run (triggers JIT compilation; not counted in timing)
# ---------------------------------------------------------------------------

println("\nWarm-up run …")
memgram(
    x, DT;
    segment_length      = SEG_LEN,
    overlap             = OVERLAP,
    optimisation_method = OPT_METHOD,
    method              = METHOD,
    verbose             = false,
)
println("Warm-up done.")

# ---------------------------------------------------------------------------
# 3.  Timed runs
# ---------------------------------------------------------------------------

println("\nTiming Memgram  (segment_length=$SEG_LEN, overlap=$(round(Int, OVERLAP*100)) %,  $N_REPS repetitions) …")
times = Vector{Float64}(undef, N_REPS)
for i in 1:N_REPS
    times[i] = @elapsed memgram(
        x, DT;
        segment_length      = SEG_LEN,
        overlap             = OVERLAP,
        optimisation_method = OPT_METHOD,
        method              = METHOD,
        verbose             = false,
    )
    println("  rep $i / $N_REPS :  $(round(times[i], digits=3)) s")
end

mean_t = sum(times) / N_REPS
min_t  = minimum(times)
println("\nTiming summary:")
println("  threads=$(Threads.nthreads())  N=$N_TOTAL  seg=$SEG_LEN  " *
        "overlap=$(round(Int, OVERLAP*100))%  reps=$N_REPS")
println("  mean = $(round(mean_t, digits=3)) s    min = $(round(min_t, digits=3)) s")

# ---------------------------------------------------------------------------
# 4.  Final run for the plot
# ---------------------------------------------------------------------------

println("\nComputing Memgram for plot …")
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
# 5.  Plot and save
# ---------------------------------------------------------------------------

plt = plot_spectrogram(
    t_centers, f_grid, psd_mat;
    title = "Memgram – quadratic chirp  ($(F_START) → $(F_END) Hz)\n" *
            "N=$(N_TOTAL) (16×2048), segment=$(SEG_LEN) samples, " *
            "$(round(Int, OVERLAP*100)) % overlap  |  " *
            "mean Δt=$(round(mean_t, digits=3)) s  ($(Threads.nthreads()) thread(s))",
    size  = (960, 500),
    dpi   = 150,
)

# Overlay the theoretical instantaneous frequency of the quadratic chirp.
f_inst = @. F_START + alpha * t_centers^2
plot!(plt, t_centers, f_inst;
      lw     = 2,
      ls     = :dash,
      color  = :white,
      label  = "Theoretical f(t)",
      legend = :topleft,
)

out_path = joinpath(@__DIR__, "quadratic_chirp_spectrogram.png")
savefig(plt, out_path)
println("\nMemgram saved to $out_path")
