"""
LISA-inspired Memgram of a millihertz chirp signal.

Simulates a gravitational-wave chirp sweeping from F_START to F_END over a
10-day observation with 5-second cadence (0.2 Hz sampling rate), as expected
for the LISA space-based detector.  The Memgram is computed on overlapping
segments of 10⁵ seconds (20 000 samples) with 95 % overlap.

Run from the repository root:

    julia examples/lisa_memgram.jl

Use multiple CPU threads (recommended for large spectrograms):

    julia -t auto examples/lisa_memgram.jl

Use a CUDA GPU (auto-detected; requires CUDA.jl):

    julia examples/lisa_memgram.jl --use_gpu true

All parameters can be supplied via command-line flags or a TOML config file:

    julia -t auto examples/lisa_memgram.jl --config examples/configs/lisa_memgram.toml
    julia examples/lisa_memgram.jl --f_start 1e-4 --f_end 1e-3

"""

# Make the src directory visible so we can load Memspectrum without installing it.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "Memspectrum.jl"))
using .Memspectrum

using ArgParse
using TOML
using Random
using Plots

# CUDA is loaded conditionally so the script also runs on CPU-only machines.
const HAVE_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

# ---------------------------------------------------------------------------
# Argument parsing (command-line + TOML config file)
# ---------------------------------------------------------------------------

function parse_commandline()
    s = ArgParseSettings(description = "LISA-inspired Memgram of a millihertz chirp")
    @add_arg_table! s begin
        "--config"
            help     = "Path to a TOML configuration file (optional)"
            arg_type = String
            default  = nothing
        "--dt"
            help     = "Sampling cadence in seconds (default: 5.0)"
            arg_type = Float64
            default  = nothing
        "--t"
            help     = "Total duration in seconds (default: 10 days = 864 000 s)"
            arg_type = Float64
            default  = nothing
        "--n_total"
            help     = "Total number of samples (overrides --t if provided)"
            arg_type = Int
            default  = nothing
        "--sigma"
            help     = "Additive noise amplitude (default: 0.1)"
            arg_type = Float64
            default  = nothing
        "--seed"
            help     = "Random seed"
            arg_type = Int
            default  = nothing
        "--f_start"
            help     = "Chirp start frequency in Hz (default: 1e-4)"
            arg_type = Float64
            default  = nothing
        "--f_end"
            help     = "Chirp end frequency in Hz (default: 1e-3)"
            arg_type = Float64
            default  = nothing
        "--seg_s"
            help     = "Segment duration in seconds (default: 1e5)"
            arg_type = Float64
            default  = nothing
        "--overlap"
            help     = "Fractional overlap between segments, 0–1 (default: 0.95)"
            arg_type = Float64
            default  = nothing
        "--optimisation_method"
            help     = "AR order-selection criterion (default: FPE)"
            arg_type = String
            default  = nothing
        "--method"
            help     = "Burg algorithm variant: Fast or Standard (default: Fast)"
            arg_type = String
            default  = nothing
        "--use_gpu"
            help     = "Use GPU acceleration if CUDA is available (default: auto)"
            arg_type = Bool
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

const DT      = _get("dt",      5.0)
const T       = _get("t",       10.0 * 86400.0)   # 10 days in seconds
const N_TOTAL = args["n_total"] !== nothing ? args["n_total"] :
                haskey(cfg, "n_total")      ? cfg["n_total"]  :
                round(Int, T / DT)
const SIGMA   = _get("sigma",   0.1)
const SEED    = _get("seed",    11)
const F_START = _get("f_start", 1e-4)
const F_END   = _get("f_end",   1e-3)
const SEG_S   = _get("seg_s",   1e5)
const SEG_LEN = round(Int, SEG_S / DT)
const OVERLAP = _get("overlap", 0.95)
const OPT_METHOD = _get("optimisation_method", "FPE")
const METHOD     = _get("method",              "Fast")
# Use GPU if explicitly requested, or auto-detect from CUDA availability.
const USE_GPU = if args["use_gpu"] !== nothing
    args["use_gpu"]
elseif haskey(cfg, "use_gpu")
    cfg["use_gpu"]
else
    HAVE_CUDA
end

# ---------------------------------------------------------------------------
# 1.  Generate a linear chirp signal with additive noise
#     φ(t) = 2π · (F_START · t  +  (F_END − F_START) / (2 · T) · t²)
#     x(t) = sin(φ(t))  +  σ · noise(t)
# ---------------------------------------------------------------------------

Random.seed!(SEED)

t_vec = collect((0:N_TOTAL-1) .* DT)
T_eff = N_TOTAL * DT
phase = @. 2π * (F_START * t_vec + (F_END - F_START) / (2 * T_eff) * t_vec^2)
x     = sin.(phase) .+ SIGMA .* randn(N_TOTAL)

println("LISA-inspired chirp: N=$N_TOTAL samples, DT=$DT s,  " *
        "T=$(round(T_eff/86400, digits=2)) days")
println("  Frequency sweep:  $(F_START) → $(F_END) Hz")
println("  Nyquist:          $(round(0.5/DT, sigdigits=3)) Hz")
println("  Segment:          $SEG_S s  =  $SEG_LEN samples")
println("  Overlap:          $(round(Int, OVERLAP*100)) %")
println("  Noise amplitude:  σ=$SIGMA")
println("  CPU threads:      $(Threads.nthreads())" *
        (Threads.nthreads() == 1 ?
         "  (tip: re-run with `julia -t auto` to use all cores)" : ""))
if USE_GPU
    if HAVE_CUDA
        println("  GPU:              $(CUDA.name(CUDA.device()))  [use_gpu=true]")
    else
        @warn "use_gpu=true requested but CUDA is not available – falling back to CPU."
    end
else
    println("  GPU:              disabled  (pass --use_gpu true to enable)")
end

# ---------------------------------------------------------------------------
# 2.  Compute MESA spectrogram (Memgram)
# ---------------------------------------------------------------------------

println("\nComputing Memgram …")
t_centers, f_grid, psd_mat = memgram(
    x, DT;
    segment_length      = SEG_LEN,
    overlap             = OVERLAP,
    optimisation_method = OPT_METHOD,
    method              = METHOD,
    verbose             = true,
    use_gpu             = USE_GPU && HAVE_CUDA,
)

println("\nMemgram: $(size(psd_mat, 2)) segments × $(length(f_grid)) frequency bins")
println("  Time range:  $(round(t_centers[1]/86400, digits=2)) – " *
        "$(round(t_centers[end]/86400, digits=2)) days")
println("  Freq range:  $(round(f_grid[1],  sigdigits=3)) – " *
        "$(round(f_grid[end], sigdigits=3)) Hz")

# ---------------------------------------------------------------------------
# 3.  Plot and save
# ---------------------------------------------------------------------------

plt = plot_spectrogram(
    t_centers ./ 86400, f_grid, psd_mat;
    title = "LISA Memgram – chirp  ($(F_START) → $(F_END) Hz)\n" *
            "DT=$(DT) s,  segment=$(SEG_S) s,  " *
            "$(round(Int, OVERLAP*100)) % overlap,  σ_noise=$(SIGMA)",
    size  = (960, 500),
    dpi   = 150,
)
xlabel!(plt, "Time (days)")
ylabel!(plt, "Frequency (Hz)")

# Overlay the theoretical instantaneous frequency of the chirp.
f_inst = @. F_START + (F_END - F_START) / T_eff * (t_centers)
plot!(plt, t_centers ./ 86400, f_inst;
      lw     = 2,
      ls     = :dash,
      color  = :white,
      label  = "Theoretical f(t)",
      legend = :topleft,
)

out_path = joinpath(@__DIR__, "lisa_memgram.png")
savefig(plt, out_path)
println("\nMemgram saved to $out_path")
