"""
Timing benchmark for `mesa_spectrogram` using Julia multi-threading.

The script times the MESA spectrogram computation for N_REPS repetitions and
prints a one-line summary.  Run with different -t flags to compare thread counts:

    julia -t 1 examples/timing_benchmark.jl
    julia -t 2 examples/timing_benchmark.jl
    julia -t 4 examples/timing_benchmark.jl

The output can be redirected to a text file for collection:

    julia -t 1 examples/timing_benchmark.jl >> timing_results.txt
    julia -t 2 examples/timing_benchmark.jl >> timing_results.txt

"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "Memspectrum.jl"))
using .Memspectrum

using Random
using FFTW

# Restrict FFTW to a single thread so that all measured parallelism comes
# exclusively from Julia's own thread pool (Threads.@threads in mesa_spectrogram).
FFTW.set_num_threads(1)

# ---------------------------------------------------------------------------
# Signal parameters
# ---------------------------------------------------------------------------

const FS       = 256.0
const DT       = 1.0 / FS
const N_TOTAL  = 32_768          # ~128 s of data
const SEG_LEN  = 512             # ~2 s segments
const OVERLAP  = 0.75            # 75 % overlap  →  ~240 segments
const SEED     = 42

Random.seed!(SEED)
x = randn(N_TOTAL)

n_threads = Threads.nthreads()

# ---------------------------------------------------------------------------
# Warm-up (triggers JIT compilation; not counted in timing)
# ---------------------------------------------------------------------------

mesa_spectrogram(x, DT; segment_length=SEG_LEN, overlap=OVERLAP,
                 optimisation_method="FPE", method="Fast")

# ---------------------------------------------------------------------------
# Timing loop
# ---------------------------------------------------------------------------

const N_REPS = 5

times = Vector{Float64}(undef, N_REPS)
for i in 1:N_REPS
    times[i] = @elapsed mesa_spectrogram(x, DT; segment_length=SEG_LEN,
                                          overlap=OVERLAP,
                                          optimisation_method="FPE",
                                          method="Fast")
end

mean_t = sum(times) / N_REPS
min_t  = minimum(times)

println("threads=$(n_threads)  N=$(N_TOTAL)  seg=$(SEG_LEN)  overlap=$(Int(OVERLAP*100))%  " *
        "reps=$(N_REPS)  mean=$(round(mean_t, digits=3))s  min=$(round(min_t, digits=3))s")
