"""
Online Memgram animation from a streaming chirp signal.

The script:
1. Generates a long linear-chirp time series (simulating a real-time sensor
   stream).
2. Feeds the signal in small chunks to `MemgramOnline.start_processor`, which
   asynchronously fits an AR model on each new overlapping window and calls
   back with the updated PSD column.
3. Collects all columns as they arrive, then renders an animated GIF where
   each frame shows the spectrogram grown by one more segment.

This demonstrates the Julia `Channel`/`Task` based async pattern (analogous
to Python's `asyncio`).

Run from the repository root:

    julia examples/online_streaming_animation.jl

All parameters can be supplied via command-line flags or a TOML config file:

    julia examples/online_streaming_animation.jl --config examples/configs/online_streaming_animation.toml
    julia examples/online_streaming_animation.jl --t 20 --chunk_size 64

"""

# ---------------------------------------------------------------------------
# Load Memspectrum (and its MemgramOnline sub-module)
# ---------------------------------------------------------------------------

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "Memspectrum.jl"))
using .Memspectrum
using .Memspectrum.MemgramOnline

using ArgParse
using TOML
using Random
using Plots

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

function parse_commandline()
    s = ArgParseSettings(
        description = "Online Memgram animation from a streaming chirp")
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
            help     = "Total signal duration (seconds)"
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
        "--chunk_size"
            help     = "Number of samples per streaming chunk"
            arg_type = Int
            default  = nothing
        "--fps"
            help     = "Frames per second for the output GIF"
            arg_type = Int
            default  = nothing
        "--frame_stride"
            help     = "Animate every N-th segment (1 = every segment)"
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

const FS           = _get("fs",                   256.0)
const DT           = 1.0 / FS
const T_TOTAL      = _get("t",                    16.0)
const N_TOTAL      = round(Int, T_TOTAL * FS)
const SIGMA        = _get("sigma",                0.2)
const SEED         = _get("seed",                 42)
const F_START      = _get("f_start",              5.0)
const F_END        = _get("f_end",                100.0)
const SEG_LEN      = _get("seg_len",              256)
const OVERLAP      = _get("overlap",              0.75)
const OPT_METHOD   = _get("optimisation_method",  "FPE")
const METHOD       = _get("method",               "Fast")
const CHUNK_SIZE   = _get("chunk_size",           64)
const FPS          = _get("fps",                  12)
const FRAME_STRIDE = _get("frame_stride",         1)

# ---------------------------------------------------------------------------
# 1.  Generate a long linear-chirp signal (simulates a real-time stream)
# ---------------------------------------------------------------------------

Random.seed!(SEED)

t_vec = collect((0:N_TOTAL-1) .* DT)
# Instantaneous frequency sweeps linearly: f(t) = F_START + (F_END - F_START) * t / T_TOTAL
# The accumulated phase is the integral: φ(t) = 2π * [F_START*t + (F_END-F_START)/(2*T)*t²]
phase = @. 2π * (F_START * t_vec + (F_END - F_START) / (2 * T_TOTAL) * t_vec^2)
x     = sin.(phase) .+ SIGMA .* randn(N_TOTAL)

println("Generated linear chirp:  N=$N_TOTAL samples  dt=$DT s  T=$(T_TOTAL) s")
println("  Frequency sweep : $(F_START) → $(F_END) Hz")
println("  Noise amplitude : σ=$SIGMA")

# ---------------------------------------------------------------------------
# 2.  Stream signal in chunks through the async MemgramOnline processor
#
#     start_processor spawns a background Task (Julia coroutine) that drains
#     a Channel[Vector{Float64}], maintains a sliding buffer, and fires the
#     on_update callback each time a full segment is ready – exactly the same
#     producer/consumer pattern as Python's asyncio.Queue + asyncio.create_task.
# ---------------------------------------------------------------------------

t_centers = Float64[]
psd_cols  = Vector{Float64}[]
f_grid    = Float64[]           # filled on first callback

proc = start_processor(;
    segment_length      = SEG_LEN,
    overlap             = OVERLAP,
    dt                  = DT,
    optimisation_method = OPT_METHOD,
    method              = METHOD,
    on_update = (t_c, f_g, psd_c) -> begin
        push!(t_centers, t_c)
        push!(psd_cols,  copy(psd_c))
        isempty(f_grid) && append!(f_grid, f_g)
    end,
)

println("\nStreaming $N_TOTAL samples in chunks of $CHUNK_SIZE …")
n_chunks = 0
for start in 1:CHUNK_SIZE:N_TOTAL
    chunk = x[start : min(start + CHUNK_SIZE - 1, N_TOTAL)]
    push_chunk!(proc, chunk)
    n_chunks += 1
end

close_processor!(proc)      # wait for the background Task to finish

println("Done.  $n_chunks chunks sent → $(length(t_centers)) segments produced.")

isempty(t_centers) && error("No segments were produced.  " *
    "Try increasing T or decreasing seg_len.")

# ---------------------------------------------------------------------------
# 3.  Build animated GIF – each frame adds one more spectrogram column
# ---------------------------------------------------------------------------

println("\nBuilding animation (frame_stride=$FRAME_STRIDE, fps=$FPS) …")

# Assemble the full PSD matrix and compute global colour limits once.
# Use a small floor value to avoid log10(0) for zero or near-zero PSD values.
const MIN_PSD_FLOOR = 1e-300
psd_mat  = hcat(psd_cols...)
log_psd  = log10.(max.(psd_mat, MIN_PSD_FLOOR))
clim_lo  = minimum(log_psd)
clim_hi  = maximum(log_psd)

# Theoretical instantaneous frequency of the chirp at each time centre.
f_inst_full = @. F_START + (F_END - F_START) / T_TOTAL * t_centers

frame_indices = collect(1 : FRAME_STRIDE : length(t_centers))

anim = @animate for k in frame_indices
    t_sub      = t_centers[1:k]
    log_sub    = log_psd[:, 1:k]
    f_inst_sub = f_inst_full[1:k]

    heatmap(t_sub, f_grid, log_sub;
            xlabel         = "Time (s)",
            ylabel         = "Frequency (Hz)",
            title          = "Online Memgram – chirp $(F_START)→$(F_END) Hz" *
                             "\nseg=$(SEG_LEN) samples, $(round(Int, OVERLAP*100)) % overlap",
            colorbar_title = "log₁₀ PSD",
            clims          = (clim_lo, clim_hi),
            xlims          = (0.0, T_TOTAL),
            ylims          = (0.0, 0.5 / DT),
            size           = (900, 450),
            dpi            = 100,
    )

    # Overlay theoretical chirp frequency as a dashed white line.
    plot!(t_sub, f_inst_sub;
          lw     = 2,
          ls     = :dash,
          color  = :white,
          label  = "f(t) theory",
          legend = :topleft,
    )
end

out_path = joinpath(@__DIR__, "online_streaming_animation.gif")
gif(anim, out_path; fps = FPS)
println("Animation saved to $out_path  ($(length(frame_indices)) frames @ $FPS fps)")
