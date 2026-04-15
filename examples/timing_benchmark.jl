"""
Timing benchmark for `memgram` using Julia multi-threading.

The script times the Memgram computation for N_REPS repetitions and
prints a one-line summary.  Run with different -t flags to compare thread counts:

    julia -t 1 examples/timing_benchmark.jl
    julia -t 2 examples/timing_benchmark.jl
    julia -t 4 examples/timing_benchmark.jl

Parameters can also be supplied via a TOML config file:

    julia examples/timing_benchmark.jl --config examples/configs/timing_benchmark.toml

The output can be redirected to a text file for collection:

    julia -t 1 examples/timing_benchmark.jl >> timing_results.txt
    julia -t 2 examples/timing_benchmark.jl >> timing_results.txt

"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "Memspectrum.jl"))
using .Memspectrum

using ArgParse
using TOML
using Random
using FFTW

# Restrict FFTW to a single thread so that all measured parallelism comes
# exclusively from Julia's own thread pool (Threads.@threads in memgram).
FFTW.set_num_threads(1)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

function parse_commandline()
    s = ArgParseSettings(description = "Timing benchmark for the Memgram computation")
    @add_arg_table! s begin
        "--config"
            help     = "Path to a TOML configuration file (optional)"
            arg_type = String
            default  = nothing
        "--n_total"
            help     = "Total number of samples"
            arg_type = Int
            default  = nothing
        "--seg_len"
            help     = "Samples per segment"
            arg_type = Int
            default  = nothing
        "--overlap"
            help     = "Fractional overlap (0–1)"
            arg_type = Float64
            default  = nothing
        "--n_reps"
            help     = "Number of timing repetitions"
            arg_type = Int
            default  = nothing
        "--seed"
            help     = "Random seed"
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

const FS       = 256.0
const DT       = 1.0 / FS
const N_TOTAL  = _get("n_total",  32_768)
const SEG_LEN  = _get("seg_len",  512)
const OVERLAP  = _get("overlap",  0.75)
const N_REPS   = _get("n_reps",   5)
const SEED     = _get("seed",     42)

Random.seed!(SEED)
x = randn(N_TOTAL)

n_threads = Threads.nthreads()

# ---------------------------------------------------------------------------
# Warm-up (triggers JIT compilation; not counted in timing)
# ---------------------------------------------------------------------------

memgram(x, DT; segment_length=SEG_LEN, overlap=OVERLAP,
        optimisation_method="FPE", method="Fast")

# ---------------------------------------------------------------------------
# Timing loop
# ---------------------------------------------------------------------------

times = Vector{Float64}(undef, N_REPS)
for i in 1:N_REPS
    times[i] = @elapsed memgram(x, DT; segment_length=SEG_LEN,
                                 overlap=OVERLAP,
                                 optimisation_method="FPE",
                                 method="Fast")
end

mean_t = sum(times) / N_REPS
min_t  = minimum(times)

println("threads=$(n_threads)  N=$(N_TOTAL)  seg=$(SEG_LEN)  overlap=$(Int(OVERLAP*100))%  " *
        "reps=$(N_REPS)  mean=$(round(mean_t, digits=3))s  min=$(round(min_t, digits=3))s")
