"""
GPU vs CPU timing benchmark and spectrogram for a rich synthetic signal.

The script:
1. Generates a long time series of N_TOTAL = 16 × 32768 = 524288 samples
   composed of:
     • three stationary sinusoids at distinct frequencies,
     • a linear chirp whose instantaneous frequency sweeps from F_CHIRP_START
       to F_CHIRP_END, together with its 2nd and 3rd harmonics,
     • additive Gaussian noise.
2. Computes the Memgram twice — first on the CPU, then on the GPU — using
   segments of SEG_LEN = 4096 samples with fractional overlap OVERLAP.
3. Prints a timing summary so you can directly compare CPU and GPU throughput.
4. Saves a side-by-side spectrogram image to
   `examples/gpu_sinusoids_chirp_spectrogram.png`.

Prerequisites
-------------
  • Julia ≥ 1.9
  • A CUDA-capable GPU with CUDA.jl installed:
        julia> using Pkg; Pkg.add("CUDA")
  • The Memspectrum package (loaded from source via LOAD_PATH below).

Run from the repository root:

    julia --project examples/gpu_sinusoids_chirp_spectrogram.jl

Or with custom parameters:

    julia --project examples/gpu_sinusoids_chirp_spectrogram.jl \\
        --fs 4096 --seg_len 4096 --overlap 0.75 --n_reps 3

Or via a TOML config file:

    julia --project examples/gpu_sinusoids_chirp_spectrogram.jl \\
        --config examples/configs/gpu_sinusoids_chirp_spectrogram.toml
"""

# ---------------------------------------------------------------------------
# Load Memspectrum from source (no installation required)
# ---------------------------------------------------------------------------
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
# Argument parsing (command-line flags + optional TOML config file)
# ---------------------------------------------------------------------------

function parse_commandline()
    s = ArgParseSettings(
        description = "GPU vs CPU Memgram: sinusoids + chirp with harmonics")
    @add_arg_table! s begin
        "--config"
            help     = "Path to a TOML configuration file (optional)"
            arg_type = String
            default  = nothing
        "--fs"
            help     = "Sampling rate (Hz)"
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
            help     = "AR order-selection criterion (FPE, AIC, ...)"
            arg_type = String
            default  = nothing
        "--method"
            help     = "Burg algorithm variant (Fast or Standard)"
            arg_type = String
            default  = nothing
        "--sigma"
            help     = "Additive white noise amplitude"
            arg_type = Float64
            default  = nothing
        "--seed"
            help     = "Random seed"
            arg_type = Int
            default  = nothing
        "--n_reps"
            help     = "Number of timed repetitions for each backend"
            arg_type = Int
            default  = nothing
        # Sinusoid frequencies
        "--f1"
            help     = "Frequency of sinusoid 1 (Hz)"
            arg_type = Float64
            default  = nothing
        "--f2"
            help     = "Frequency of sinusoid 2 (Hz)"
            arg_type = Float64
            default  = nothing
        "--f3"
            help     = "Frequency of sinusoid 3 (Hz)"
            arg_type = Float64
            default  = nothing
        # Chirp parameters
        "--f_chirp_start"
            help     = "Chirp start frequency (Hz)"
            arg_type = Float64
            default  = nothing
        "--f_chirp_end"
            help     = "Chirp end frequency (Hz)"
            arg_type = Float64
            default  = nothing
        "--n_harmonics"
            help     = "Number of chirp harmonics to include (≥1)"
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

# ---------------------------------------------------------------------------
# Parameters
# N_TOTAL is fixed at 16 × 32768 per the problem statement.
# ---------------------------------------------------------------------------

const FS             = _get("fs",                 4096.0)   # Hz
const DT             = 1.0 / FS
const N_TOTAL        = 16 * 32768                           # 524288 samples
const T_TOTAL        = N_TOTAL * DT                         # total duration (s)
const SEG_LEN        = _get("seg_len",            4096)
const OVERLAP        = _get("overlap",            0.75)
const OPT_METHOD     = _get("optimisation_method","FPE")
const METHOD         = _get("method",             "Fast")
const SIGMA          = _get("sigma",              0.1)
const SEED           = _get("seed",               42)
const N_REPS         = _get("n_reps",             3)

# Three stationary sinusoids (well separated in frequency)
const F1             = _get("f1",                 50.0)     # Hz
const F2             = _get("f2",                 200.0)    # Hz
const F3             = _get("f3",                 600.0)    # Hz

# Chirp: linear sweep + harmonics
const F_CHIRP_START  = _get("f_chirp_start",      10.0)     # Hz
const F_CHIRP_END    = _get("f_chirp_end",         400.0)   # Hz
const N_HARMONICS    = _get("n_harmonics",         3)        # fundamental + 2 overtones

# ---------------------------------------------------------------------------
# 1.  Build synthetic signal
# ---------------------------------------------------------------------------

Random.seed!(SEED)

t_vec = collect((0:N_TOTAL-1) .* DT)   # time axis

# Three sinusoids (unit amplitude each)
x  = sin.(2π .* F1 .* t_vec)
x .+= sin.(2π .* F2 .* t_vec)
x .+= sin.(2π .* F3 .* t_vec)

# Linear-chirp phase:  φ(t) = 2π · [f_start · t  +  (f_end – f_start) / (2T) · t²]
chirp_phase = @. 2π * (F_CHIRP_START * t_vec +
                       (F_CHIRP_END - F_CHIRP_START) / (2 * T_TOTAL) * t_vec^2)

# Add fundamental + harmonics of the chirp (harmonic k has phase k·φ(t))
for k in 1:N_HARMONICS
    x .+= (1.0 / k) .* sin.(k .* chirp_phase)   # amplitude ∝ 1/k
end

# Additive white Gaussian noise
x .+= SIGMA .* randn(N_TOTAL)

println("="^70)
println("Signal summary")
println("="^70)
println("  N_TOTAL     = $N_TOTAL  (16 × 32768)")
println("  FS          = $(FS) Hz   →  T = $(round(T_TOTAL, digits=3)) s")
println("  Sinusoids   : $(F1) Hz,  $(F2) Hz,  $(F3) Hz")
println("  Chirp       : $(F_CHIRP_START) → $(F_CHIRP_END) Hz  " *
        "($(N_HARMONICS) harmonic$(N_HARMONICS > 1 ? "s" : ""))")
println("  Noise σ     : $SIGMA")
println("  SEG_LEN     = $SEG_LEN,  overlap = $(Int(round(OVERLAP * 100))) %")
println()

# ---------------------------------------------------------------------------
# Helper: time a memgram call N_REPS times and return (times, result)
# ---------------------------------------------------------------------------

function time_memgram(x, dt; kwargs...)
    # Warm-up: triggers JIT compilation (not counted)
    memgram(x, dt; kwargs...)
    # Timed runs
    times = Vector{Float64}(undef, N_REPS)
    local result
    for i in 1:N_REPS
        times[i] = @elapsed begin
            result = memgram(x, dt; kwargs...)
        end
        print("  rep $i/$(N_REPS): $(round(times[i], digits=3)) s\n")
    end
    return times, result
end

# ---------------------------------------------------------------------------
# 2.  CPU timing
# ---------------------------------------------------------------------------

println("─"^70)
println("CPU benchmark  ($(Threads.nthreads()) thread(s))")
println("─"^70)

cpu_times, (t_centers, f_grid, psd_cpu) = time_memgram(
    x, DT;
    segment_length      = SEG_LEN,
    overlap             = OVERLAP,
    optimisation_method = OPT_METHOD,
    method              = METHOD,
    verbose             = false,
    use_gpu             = false,
)

cpu_mean = sum(cpu_times) / N_REPS
cpu_min  = minimum(cpu_times)
println("  → mean $(round(cpu_mean, digits=3)) s   min $(round(cpu_min, digits=3)) s")
println()

# ---------------------------------------------------------------------------
# 3.  GPU timing  (only when CUDA is functional)
# ---------------------------------------------------------------------------

gpu_times  = Float64[]
psd_gpu    = nothing

if HAVE_CUDA
    println("─"^70)
    println("GPU benchmark  ($(CUDA.name(CUDA.device())))")
    println("─"^70)

    gpu_times, (_, _, psd_gpu) = time_memgram(
        x, DT;
        segment_length      = SEG_LEN,
        overlap             = OVERLAP,
        optimisation_method = OPT_METHOD,
        method              = METHOD,
        verbose             = false,
        use_gpu             = true,
    )

    gpu_mean = sum(gpu_times) / N_REPS
    gpu_min  = minimum(gpu_times)
    println("  → mean $(round(gpu_mean, digits=3)) s   min $(round(gpu_min, digits=3)) s")
    println()

    speedup = cpu_mean / gpu_mean
    println("  GPU speedup (mean): $(round(speedup, digits=2))×")
else
    println("CUDA not available – skipping GPU benchmark.")
    println("  Install CUDA.jl and run on a CUDA-capable GPU to enable the GPU path.")
end

println("="^70)
println("Spectrogram: $(size(psd_cpu, 2)) segments × $(length(f_grid)) freq bins")
println("  Time range : $(round(t_centers[1], digits=3)) – " *
        "$(round(t_centers[end], digits=3)) s")
println("  Freq range : $(round(f_grid[1], digits=3)) – " *
        "$(round(f_grid[end], digits=3)) Hz")
println()

# ---------------------------------------------------------------------------
# 4.  Plot and save
# ---------------------------------------------------------------------------

# Use the CPU result for the saved figure (GPU result is identical up to
# floating-point rounding on the FFT step).
psd_plot = psd_gpu !== nothing ? psd_gpu : psd_cpu

# Theoretical instantaneous frequency of the chirp fundamental
f_inst = @. F_CHIRP_START + (F_CHIRP_END - F_CHIRP_START) / T_TOTAL * t_centers

title_str = "Memgram – sinusoids ($(F1), $(F2), $(F3) Hz) + chirp " *
            "($(F_CHIRP_START)→$(F_CHIRP_END) Hz, $(N_HARMONICS) harmonics)\n" *
            "N=$(N_TOTAL),  seg=$(SEG_LEN),  overlap=$(Int(round(OVERLAP*100))) %,  " *
            "CPU mean=$(round(cpu_mean, digits=2)) s" *
            (isempty(gpu_times) ? "" :
             "  GPU mean=$(round(sum(gpu_times)/N_REPS, digits=2)) s")

plt = plot_spectrogram(
    t_centers, f_grid, psd_plot;
    title = title_str,
    size  = (1200, 550),
    dpi   = 150,
)

# Overlay chirp fundamental and harmonics as dashed lines
for k in 1:N_HARMONICS
    f_k = clamp.(k .* f_inst, f_grid[1], f_grid[end])
    plot!(plt, t_centers, f_k;
          lw     = 1.5,
          ls     = :dash,
          color  = :white,
          label  = k == 1 ? "Chirp harmonics" : "",
          legend = :topleft,
    )
end

# Mark the three sinusoid frequencies as horizontal dashed lines
for (fi, label) in [(F1, "$(F1) Hz"), (F2, "$(F2) Hz"), (F3, "$(F3) Hz")]
    hline!(plt, [fi];
           lw    = 1,
           ls    = :dot,
           color = :yellow,
           label = label,
    )
end

out_path = joinpath(@__DIR__, "gpu_sinusoids_chirp_spectrogram.png")
savefig(plt, out_path)
println("Spectrogram saved to $out_path")
