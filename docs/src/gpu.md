# GPU acceleration

GPU support is provided by the package extension `MemspectrumCUDAExt`.  CUDA.jl
is an optional (weak) dependency: it is not installed with `Memspectrum`, and
the extension loads automatically when both packages are present in the same
session.

## Setup

Install CUDA.jl in the environment you work in (or in your default
environment, which is stacked on top of project environments):

```julia
using Pkg
Pkg.add("CUDA")
```

Then load both packages:

```julia
using CUDA
using Memspectrum

CUDA.functional()   # should be true
```

Passing `use_gpu=true` without CUDA.jl loaded raises an error rather than
silently running on the CPU.

## Supported functions

| Call | What runs on the GPU |
|------|----------------------|
| `forecast(m, data, len; number_of_simulations, use_gpu=true)` | All simulated realisations in parallel (`Float32`) |
| `memgram(x, dt; segment_length, use_gpu=true)` | FFT of each segment's AR polynomial on separate CUDA streams; Burg runs on CPU threads |
| `solve!(m, times, x; ..., use_gpu=true)` | Pair search and prediction-error updates of the irregular Burg recursion |
| `memgram(times, x; dt, segment_duration, use_gpu=true)` | Irregular Burg recursion of *all* segments in one batch |

All functions return ordinary CPU arrays.

```julia
m = MESA()
solve!(m, data)
sims = forecast(m, data, 1000; number_of_simulations=2048, use_gpu=true)
t, f, S = memgram(x, dt; segment_length=512, use_gpu=true)

# Irregularly sampled data
solve!(m, times, y; dt=1.0, slot_width=0.5, use_gpu=true)
t, f, S = memgram(times, y; dt=1.0, segment_duration=512.0, use_gpu=true)
```

## Irregular data on the GPU

All series are concatenated (for a spectrogram, one per window), and each Burg
order is a fixed sequence of kernels over all of them:

1. one thread per observation finds its partners within the slot, accumulates
   the products of eq. (5), and records its closest partner;
2. one thread per observation records its closest earlier partner;
3. one block per series sums the products;
4. the reflection coefficients are computed on the host, and one thread per
   observation applies the error update of eq. (6).

Pair searches never cross series boundaries.  Only the per-series sums are
copied back to the host at each order; the Levinson recursion, order selection
and PSD evaluation run on the CPU.  The computation is in `Float64` and selects
exactly the same pairs (hence the same `n_products`) as the CPU path; the
reflection coefficients agree to round-off, since only the summation order
differs.

## When is it faster?

Indicative timings on a laptop RTX 3050 Ti (whose double-precision throughput
is 1/64 of single precision), against a 16-thread CPU:

| Task | CPU | GPU |
|------|-----|-----|
| irregular `solve!`, ``N = 10^5`` | 0.01 s | 0.05 s |
| irregular `solve!`, ``N = 10^6`` | 0.14 s | 0.05 s |
| irregular `solve!`, ``N = 4\times10^6`` | 0.7–1.1 s | 0.22 s |
| irregular `memgram`, ``N = 10^7`` | 0.66 s | 1.15 s |

The GPU pays off for long single series (``\gtrsim 10^6`` observations).
Spectrograms already run in parallel on CPU threads (`julia -t auto`), which
can be faster than the GPU on cards with weak double-precision performance.

## Testing

```sh
julia --project=. test/gpu_tests.jl
```

The script needs CUDA.jl in the load path.  CPU-fallback tests always run;
GPU tests, including the CPU/GPU agreement checks for irregular data, run only
when `CUDA.functional()` is true.
