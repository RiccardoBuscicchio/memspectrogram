"""
    MemspectrumCUDAExt

CUDA extension for Memspectrum.  Loaded automatically when both `Memspectrum`
and `CUDA` are active in the same Julia session.

Provides GPU-accelerated overloads of `forecast` and `mesa_spectrogram`
(`memgram`) via the keyword argument `use_gpu=true`.

## Example

```julia
using CUDA
using Memspectrum

m = MESA()
solve!(m, data)

# GPU-accelerated multi-simulation forecast
sims = forecast(m, data, 1000; number_of_simulations=2048, use_gpu=true)

# GPU-accelerated spectrogram (segment loop on GPU)
t, f, S = memgram(x, dt; segment_length=512, use_gpu=true)
```
"""
module MemspectrumCUDAExt

using Memspectrum
using CUDA

# ---------------------------------------------------------------------------
# GPU-accelerated forecast
# ---------------------------------------------------------------------------

"""
    Memspectrum.forecast(m, data, len; ..., use_gpu=true)

GPU-accelerated variant of `forecast`.  When `use_gpu=true`, all
`number_of_simulations` realisations are generated in parallel on the GPU.

The returned matrix is a regular CPU `Matrix{Float64}` (copied back from the
device).  All other keyword arguments are identical to the CPU version.
"""
function Memspectrum.forecast(mesa::Memspectrum.MESA, data::AbstractVector,
                               len::Int;
                               number_of_simulations::Int=1,
                               P=nothing,
                               include_data::Bool=false,
                               seed::Union{Int,Nothing}=nothing,
                               verbose::Bool=false,
                               use_gpu::Bool=false)
    use_gpu || return invoke(Memspectrum.forecast,
                             Tuple{Memspectrum.MESA, AbstractVector, Int},
                             mesa, data, len;
                             number_of_simulations=number_of_simulations,
                             P=P, include_data=include_data,
                             seed=seed, verbose=verbose)

    (mesa.P === nothing || mesa.a_k === nothing) &&
        error("Model not fitted. Call solve! before forecast.")

    P_use = P === nothing ? mesa.P : P
    p     = length(mesa.a_k) - 1
    sigma = Float32(sqrt(P_use))

    # AR coefficients reversed for dot-product prediction, on GPU as Float32
    coef_cpu = Float32.(-reverse(mesa.a_k[2:end]))
    coef_gpu = CuArray(coef_cpu)           # length p

    # Initialise prediction matrix on GPU: shape (number_of_simulations, p + len)
    preds = CUDA.zeros(Float32, number_of_simulations, p + len)

    # Seed from data (last p points)
    data_f = Float64.(vec(data))
    if length(data_f) >= p > 0
        seed_row = Float32.(data_f[end-p+1:end]')   # 1 × p
        preds[:, 1:p] .= repeat(CuArray(seed_row), number_of_simulations, 1)
    elseif p != 0
        error("Data too short for forecasting: need at least $p points.")
    end

    seed !== nothing && Random.seed!(seed)

    # Iterative AR prediction on GPU
    for i in 1:len
        verbose && print("\r $(i) of $(len)")
        # past shape: (nsims, p)
        past  = preds[:, i:i+p-1]              # CuArray view
        noise = CUDA.randn(Float32, number_of_simulations) .* sigma
        preds[:, p+i] = past * coef_gpu .+ noise
    end
    verbose && println()

    result_gpu = include_data ? preds : preds[:, p+1:end]
    return Float64.(Array(result_gpu))
end

# ---------------------------------------------------------------------------
# GPU-accelerated mesa_spectrogram / memgram
# ---------------------------------------------------------------------------

"""
    Memspectrum.mesa_spectrogram(x, dt; ..., use_gpu=true)

GPU-accelerated variant of `mesa_spectrogram`.  When `use_gpu=true`, segment
PSDs are computed in parallel using multiple CUDA streams.  Each segment still
runs the Burg algorithm on the CPU (which is inherently sequential), but all
segments are dispatched concurrently via Julia `Tasks` pinned to CUDA streams,
maximising GPU utilisation for the FFT step inside each `spectrum` call.

Returns the same `(t_centers, f_grid, psd_matrix)` triple as the CPU version.
"""
function Memspectrum.mesa_spectrogram(x::AbstractVector, dt::Float64;
                                       segment_length::Int,
                                       overlap::Float64=0.5,
                                       optimisation_method::String="FPE",
                                       method::String="Fast",
                                       verbose::Bool=false,
                                       use_gpu::Bool=false)
    use_gpu || return invoke(Memspectrum.mesa_spectrogram,
                             Tuple{AbstractVector, Float64},
                             x, dt;
                             segment_length=segment_length,
                             overlap=overlap,
                             optimisation_method=optimisation_method,
                             method=method,
                             verbose=verbose)

    0.0 <= overlap < 1.0 ||
        error("overlap must be in [0, 1).")
    segment_length >= 4 ||
        error("segment_length must be at least 4.")

    x_cpu = Float64.(vec(x))
    N     = length(x_cpu)
    stride = max(1, round(Int, segment_length * (1.0 - overlap)))
    starts = collect(1 : stride : N - segment_length + 1)
    n_seg  = length(starts)
    n_seg >= 1 || error("Time series too short for the requested segment_length.")

    n_freq   = segment_length ÷ 2
    f_grid   = collect(0:n_freq-1) ./ (segment_length * dt)
    psd_matrix = Matrix{Float64}(undef, n_freq, n_seg)
    t_centers  = Vector{Float64}(undef, n_seg)

    # Use one CUDA stream per segment for concurrent kernel dispatch.
    # 8 streams is a pragmatic upper bound: enough concurrency for most GPUs
    # without excessive stream-management overhead.
    n_streams = min(n_seg, 8)
    streams   = [CuStream() for _ in 1:n_streams]

    verbose_lock = ReentrantLock()

    Threads.@threads for j in 1:n_seg
        s   = starts[j]
        seg = x_cpu[s : s + segment_length - 1]
        t_centers[j] = (s - 1 + 0.5 * segment_length) * dt

        mj = Memspectrum.MESA()
        Memspectrum.solve!(mj, seg;
                           method=method,
                           optimisation_method=optimisation_method,
                           verbose=false)

        # Run FFT on GPU using the assigned stream
        stream = streams[mod1(j, n_streams)]
        CUDA.stream!(stream) do
            a_k_gpu = CuArray(ComplexF32.(mj.a_k))
            padded = vcat(a_k_gpu, CUDA.zeros(ComplexF32,
                                            segment_length - length(mj.a_k)))
            den    = CUFFT.fft(padded)
            spec   = Float32(dt * real(mj.P)) ./ abs2.(den)
            psd_matrix[:, j] = Float64.(Array(spec[1:n_freq])) .* 2
        end

        if verbose
            lock(verbose_lock) do
                print("\r  Segment $j / $n_seg")
            end
        end
    end
    CUDA.synchronize()
    verbose && println()

    return t_centers, f_grid, psd_matrix
end

end # module MemspectrumCUDAExt
