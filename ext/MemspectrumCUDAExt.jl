"""
    MemspectrumCUDAExt

CUDA extension for Memspectrum.  Loaded automatically when both `Memspectrum`
and `CUDA` are active in the same Julia session.

Provides the GPU paths taken by `forecast`, `mesa_spectrogram` (`memgram`) and,
for irregularly sampled data, `solve!(m, times, data)` and
`mesa_spectrogram(times, x)`, when they are called with `use_gpu=true`.  The
extension adds methods to the `_*_gpu` hooks declared in `Memspectrum`, so the
CPU methods are never overwritten.

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

# Irregularly sampled data: single fit and batched spectrogram
solve!(m, times, y; dt=1.0, slot_width=0.5, use_gpu=true)
t, f, S = memgram(times, y; dt=1.0, segment_duration=512.0, use_gpu=true)
```
"""
module MemspectrumCUDAExt

using Memspectrum
using CUDA
using Random

# ---------------------------------------------------------------------------
# GPU-accelerated forecast
# ---------------------------------------------------------------------------

"""
    Memspectrum._forecast_gpu(m, data, len; ...)

GPU path of `forecast(...; use_gpu=true)`.  All
`number_of_simulations` realisations are generated in parallel on the GPU.

The returned matrix is a regular CPU `Matrix{Float64}` (copied back from the
device).  All other keyword arguments are identical to the CPU version.
"""
function Memspectrum._forecast_gpu(mesa::Memspectrum.MESA, data::AbstractVector,
                                   len::Int;
                                   number_of_simulations::Int=1,
                                   P=nothing,
                                   include_data::Bool=false,
                                   seed::Union{Int,Nothing}=nothing,
                                   verbose::Bool=false)
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

    if seed !== nothing
        Random.seed!(seed)
        CUDA.seed!(seed)
    end

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
    Memspectrum._mesa_spectrogram_gpu(x, dt; ...)

GPU path of `mesa_spectrogram(x, dt; ..., use_gpu=true)`.  Segment
PSDs are computed in parallel using multiple CUDA streams.  Each segment still
runs the Burg algorithm on the CPU (which is inherently sequential), but all
segments are dispatched concurrently via Julia `Tasks` pinned to CUDA streams,
maximising GPU utilisation for the FFT step inside each `spectrum` call.

Returns the same `(t_centers, f_grid, psd_matrix)` triple as the CPU version.
"""
function Memspectrum._mesa_spectrogram_gpu(x::AbstractVector, dt::Float64;
                                           segment_length::Int,
                                           overlap::Float64=0.5,
                                           optimisation_method::String="FPE",
                                           method::String="Fast",
                                           verbose::Bool=false)
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

# ---------------------------------------------------------------------------
# Irregularly sampled data: batched Burg recursion (eqs. 5–6 of Bos et al. 2002)
#
# All series (one per spectrogram segment, or a single one for `solve!`) are
# concatenated; every kernel processes all of them at once and pair searches
# never cross a series boundary.  Pair selection and tie-breaking follow
# `Memspectrum._irregular_reflection` exactly; only the summation order of the
# reflection-coefficient sums differs (round-off level).
# ---------------------------------------------------------------------------

const _THREADS = 256

# Thread i treats observation i as the earlier point of a pair: accumulates
# f[j] b[i] and f[j]² + b[i]² over partners j and keeps the j whose lag is
# closest to p dt (first one on ties), which updates b[i].
function _forward_pairs_kernel!(num, den, cnt, best_j, t, f, b, seg_of, seg_last,
                                active, lo, hi, lag)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(t) && return nothing
    s = seg_of[i]
    acc_num = 0.0
    acc_den = 0.0
    c = 0
    bj = 0
    @inbounds if active[s] && !isnan(b[i])
        ti = t[i]
        bi = b[i]
        last_e = seg_last[s]
        a, z = i + 1, last_e + 1          # first j with t[j] - ti >= lo
        while a < z
            mid = (a + z) >>> 1
            if t[mid] - ti >= lo
                z = mid
            else
                a = mid + 1
            end
        end
        best = Inf
        j = a
        while j <= last_e
            d = t[j] - ti
            d >= hi && break
            fj = f[j]
            if d > 0 && !isnan(fj)
                acc_num += fj * bi
                acc_den += fj^2 + bi^2
                c += 1
                err = abs(d - lag)
                if err < best
                    best = err
                    bj = j
                end
            end
            j += 1
        end
    end
    @inbounds begin
        num[i] = acc_num
        den[i] = acc_den
        cnt[i] = c
        best_j[i] = bj
    end
    return nothing
end

# Thread j treats observation j as the later point: keeps the partner i whose
# lag is closest to p dt (first one on ties), which updates f[j].
function _backward_pairs_kernel!(best_i, t, f, b, seg_of, seg_first, active, lo, hi, lag)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > length(t) && return nothing
    s = seg_of[j]
    bi = 0
    @inbounds if active[s] && !isnan(f[j])
        tj = t[j]
        a, z = seg_first[s], j            # first i with tj - t[i] < hi
        while a < z
            mid = (a + z) >>> 1
            if tj - t[mid] < hi
                z = mid
            else
                a = mid + 1
            end
        end
        best = Inf
        i = a
        while i < j
            d = tj - t[i]
            d < lo && break
            if d > 0 && !isnan(b[i])
                err = abs(d - lag)
                if err < best
                    best = err
                    bi = i
                end
            end
            i += 1
        end
    end
    @inbounds best_i[j] = bi
    return nothing
end

# One block per series: tree reduction of the per-observation sums.
function _segment_sum_kernel!(seg_num, seg_den, seg_cnt, num, den, cnt,
                              seg_first, seg_last)
    s = blockIdx().x
    tid = threadIdx().x
    sh_num = CuStaticSharedArray(Float64, _THREADS)
    sh_den = CuStaticSharedArray(Float64, _THREADS)
    sh_cnt = CuStaticSharedArray(Int, _THREADS)
    acc_num = 0.0
    acc_den = 0.0
    c = 0
    @inbounds begin
        e = seg_first[s] + tid - 1
        while e <= seg_last[s]
            acc_num += num[e]
            acc_den += den[e]
            c += cnt[e]
            e += _THREADS
        end
        sh_num[tid] = acc_num
        sh_den[tid] = acc_den
        sh_cnt[tid] = c
    end
    sync_threads()
    step = _THREADS ÷ 2
    while step > 0
        @inbounds if tid <= step
            sh_num[tid] += sh_num[tid + step]
            sh_den[tid] += sh_den[tid + step]
            sh_cnt[tid] += sh_cnt[tid + step]
        end
        sync_threads()
        step ÷= 2
    end
    @inbounds if tid == 1
        seg_num[s] = sh_num[1]
        seg_den[s] = sh_den[1]
        seg_cnt[s] = sh_cnt[1]
    end
    return nothing
end

# Eq. (6): new errors from the closest partner; NaN where there is none.
function _update_errors_kernel!(new_f, new_b, f, b, best_i, best_j, seg_of, k)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    e > length(f) && return nothing
    @inbounds begin
        ks = k[seg_of[e]]
        bj = best_j[e]
        new_b[e] = bj > 0 ? b[e] + ks * f[bj] : NaN
        bi = best_i[e]
        new_f[e] = bi > 0 ? f[e] + ks * b[bi] : NaN
    end
    return nothing
end

"""
    Memspectrum._irregular_reflection_gpu(ts, xs, dt, slot_width, mmaxs, verbose)

GPU version of `Memspectrum._irregular_reflection` for a batch of zero-mean
irregular series `xs[s](ts[s])`, each with its own maximum order `mmaxs[s]`.
Returns `(ks, n_products)`, vectors with one entry per series.
"""
function Memspectrum._irregular_reflection_gpu(ts::Vector{Vector{Float64}},
                                               xs::Vector{Vector{Float64}},
                                               dt::Float64, slot_width::Float64,
                                               mmaxs::Vector{Int}, verbose::Bool)
    n_seg = length(ts)
    lens = length.(ts)
    seg_last_h = cumsum(lens)
    seg_first_h = seg_last_h .- lens .+ 1
    n_tot = seg_last_h[end]

    ks = [Float64[] for _ in 1:n_seg]
    n_products = [Int[] for _ in 1:n_seg]
    active_h = [mmaxs[s] > 0 && lens[s] >= 2 for s in 1:n_seg]   # Vector{Bool}
    any(active_h) || return ks, n_products

    t = CuArray(reduce(vcat, ts))
    f = CuArray(reduce(vcat, xs))
    b = copy(f)
    new_f = similar(f)
    new_b = similar(b)
    seg_of = CuArray(reduce(vcat, [fill(s, lens[s]) for s in 1:n_seg]))
    seg_first = CuArray(seg_first_h)
    seg_last = CuArray(seg_last_h)
    num = CUDA.zeros(Float64, n_tot)
    den = CUDA.zeros(Float64, n_tot)
    cnt = CUDA.zeros(Int, n_tot)
    best_i = CUDA.zeros(Int, n_tot)
    best_j = CUDA.zeros(Int, n_tot)
    seg_num = CUDA.zeros(Float64, n_seg)
    seg_den = CUDA.zeros(Float64, n_seg)
    seg_cnt = CUDA.zeros(Int, n_seg)
    active = CuArray(active_h)
    k = CUDA.zeros(Float64, n_seg)
    k_h = zeros(n_seg)

    blocks = cld(n_tot, _THREADS)
    half_slot = slot_width / 2
    p = 0
    while any(active_h)
        p += 1
        if verbose
            print("\r\tIteration $p (GPU, $(count(active_h)) active series)")
            flush(stdout)
        end
        lag = p * dt
        lo, hi = lag - half_slot, lag + half_slot
        copyto!(active, active_h)

        @cuda threads=_THREADS blocks=blocks _forward_pairs_kernel!(
            num, den, cnt, best_j, t, f, b, seg_of, seg_last, active, lo, hi, lag)
        @cuda threads=_THREADS blocks=blocks _backward_pairs_kernel!(
            best_i, t, f, b, seg_of, seg_first, active, lo, hi, lag)
        @cuda threads=_THREADS blocks=n_seg _segment_sum_kernel!(
            seg_num, seg_den, seg_cnt, num, den, cnt, seg_first, seg_last)

        num_h, den_h, cnt_h = Array(seg_num), Array(seg_den), Array(seg_cnt)
        fill!(k_h, 0.0)
        for s in 1:n_seg
            active_h[s] || continue
            if cnt_h[s] == 0 || den_h[s] == 0
                active_h[s] = false
                continue
            end
            k_h[s] = -2 * num_h[s] / den_h[s]
            push!(ks[s], k_h[s])
            push!(n_products[s], cnt_h[s])
            p >= mmaxs[s] && (active_h[s] = false)
        end
        copyto!(k, k_h)

        @cuda threads=_THREADS blocks=blocks _update_errors_kernel!(
            new_f, new_b, f, b, best_i, best_j, seg_of, k)
        f, new_f = new_f, f
        b, new_b = new_b, b
    end
    verbose && println()
    return ks, n_products
end

end # module MemspectrumCUDAExt
