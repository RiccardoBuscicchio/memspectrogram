# ---------------------------------------------------------------------------
# Burg algorithm for irregularly sampled data
#
# R. Bos, S. de Waele, P. M. T. Broersen, "Autoregressive spectral estimation
# by application of the Burg algorithm to irregularly sampled data",
# IEEE Trans. Instrum. Meas. 51(6), 1289–1294 (2002).
#
# The AR model is defined on a regular grid of spacing `dt` (T in the paper).
# At order p, a product f_{p-1}(t_j) b_{p-1}(t_i) contributes to k_p whenever
# t_j - t_i falls inside the slot [p dt - Δτ/2, p dt + Δτ/2)   (eq. 5).
# Only points that contributed to k_p carry prediction errors to order p+1
# (eq. 6); duplicate predictions are resolved by keeping the pair whose lag is
# closest to p dt.  For regularly sampled data with Δτ ≤ dt this reduces to
# the standard Burg recursion.
# ---------------------------------------------------------------------------

# AIC_irreg (eq. 11); with penalty = 2 and min_products = 0 it is AIC_mod (eq. 10).
# N_0 = N, the number of observations used for the order-0 variance.
function _AIC_irregular(P::Vector, n_products::Vector{Int}, N::Int,
                        penalty::Real, min_products::Int)
    m = length(n_products)
    m > 0 && n_products[end] < min_products && return Inf
    return log(P[end]) + penalty * (1 / N + sum(inv, n_products; init=0.0))
end

function _irregular_loss(method::String, P::Vector, n_products::Vector{Int}, N::Int,
                         penalty::Real, min_products::Int)
    if method == "AICirreg"
        return _AIC_irregular(P, n_products, N, penalty, min_products)
    elseif method == "AICmod"
        return _AIC_irregular(P, n_products, N, 2.0, 0)
    elseif method == "Fixed"
        return _Fixed(length(n_products))
    else
        error("Unknown optimisation method '$method' for irregular data. " *
              "Valid choices are 'AICirreg', 'AICmod', 'Fixed'.")
    end
end

# First index j in [from, to] with t[j] - ti >= lo (to + 1 if none).  Uses the
# difference itself so CPU and GPU (ext/MemspectrumCUDAExt.jl) select exactly
# the same pairs.
function _first_at_lag(t::AbstractVector{Float64}, ti::Float64, lo::Float64,
                       from::Int, to::Int)
    a, z = from, to + 1
    while a < z
        mid = (a + z) >>> 1
        if t[mid] - ti >= lo
            z = mid
        else
            a = mid + 1
        end
    end
    return a
end

# Recursion of eqs. (5)–(6) on zero-mean data.  Returns the reflection
# coefficients and the number of products used for each of them.
function _irregular_reflection(t::Vector{Float64}, x::Vector{Float64},
                               dt::Float64, slot_width::Float64, mmax::Int,
                               verbose::Bool)
    N = length(x)
    half_slot = slot_width / 2

    # Prediction errors at order p-1: f indexed by the latest point of its
    # chain, b by the earliest.  NaN marks points without an error.
    f = copy(x)
    b = copy(x)
    new_f = similar(f)
    new_b = similar(b)
    best_f = similar(f)
    best_b = similar(b)
    pairs_i = Int[]
    pairs_j = Int[]

    ks = Float64[]
    n_products = Int[]

    for p in 1:mmax
        if verbose
            print("\r\tIteration $p of $mmax")
            flush(stdout)
        end

        lag = p * dt
        lo, hi = lag - half_slot, lag + half_slot
        empty!(pairs_i)
        empty!(pairs_j)
        num = 0.0
        den = 0.0
        for i in 1:N
            isnan(b[i]) && continue
            for j in _first_at_lag(t, t[i], lo, i + 1, N):N
                d = t[j] - t[i]
                d >= hi && break
                (d <= 0 || isnan(f[j])) && continue
                push!(pairs_i, i)
                push!(pairs_j, j)
                num += f[j] * b[i]
                den += f[j]^2 + b[i]^2
            end
        end

        n_p = length(pairs_i)
        if n_p == 0 || den == 0
            verbose && print("\n\tNo products available at order $p; stopping.")
            break
        end
        k = -2 * num / den

        fill!(new_f, NaN)
        fill!(new_b, NaN)
        fill!(best_f, Inf)
        fill!(best_b, Inf)
        for (i, j) in zip(pairs_i, pairs_j)
            err = abs(t[j] - t[i] - lag)
            if err < best_f[j]
                best_f[j] = err
                new_f[j] = f[j] + k * b[i]
            end
            if err < best_b[i]
                best_b[i] = err
                new_b[i] = b[i] + k * f[j]
            end
        end
        f, new_f = new_f, f
        b, new_b = new_b, b

        push!(ks, k)
        push!(n_products, n_p)
    end
    verbose && println()
    return ks, n_products
end

# Levinson recursion, AIC_irreg/AIC_mod/Fixed order selection, and MESA fields.
# `N` is the number of observations (N_0 in eq. 11).
function _irregular_fit!(mesa::MESA, ks::Vector{Float64}, n_products::Vector{Int},
                         P_0::Float64, N::Int, optimisation_method::String,
                         penalty::Float64, min_products::Int)
    P = [P_0]
    a_k = [Float64[1.0]]
    optimization = Float64[]
    loss_0 = _irregular_loss(optimisation_method, P, Int[], N, penalty, min_products)
    for (p, k) in enumerate(ks)
        push!(a_k, _update_prediction_coefficient(a_k[end], k))
        push!(P, P[end] * (1 - k^2))
        push!(optimization, _irregular_loss(optimisation_method, P,
                                            n_products[1:p], N,
                                            penalty, min_products))
    end

    # optimization[i] is the loss of order i; order 0 wins if nothing beats it.
    order = 0
    if !isempty(optimization)
        best = argmin(optimization)
        optimization[best] < loss_0 && (order = best)
    end

    mesa.P = P[order + 1]
    mesa.a_k = a_k[order + 1]
    mesa.ref_coefficients = copy(ks)
    mesa.optimization = optimization
    mesa.n_products = n_products
    return mesa.P, mesa.a_k, optimization
end

_default_mmax(N::Int, m::Union{Int, Nothing}) =
    m === nothing ? Int(floor(2 * N / log(2 * N))) : m

function _prepare_irregular(times::AbstractVector, data::AbstractVector)
    length(times) == length(data) ||
        error("times and data must have the same length.")
    t = Float64.(vec(times))
    x = Float64.(vec(data))
    all(isfinite, t) || error("times must be finite.")
    all(isfinite, x) || error("data must be finite.")
    if !issorted(t)
        perm = sortperm(t)
        t, x = t[perm], x[perm]
    end
    return t, x
end

_mean_interval(t::Vector{Float64}) = (t[end] - t[1]) / (length(t) - 1)

"""
    solve!(m::MESA, times, data; dt=nothing, slot_width=nothing, m=nothing,
           optimisation_method="AICirreg", penalty=4.0, min_products=15,
           verbose=false, use_gpu=false)

Fit an AR model to the irregularly sampled series `data(times)` with the Burg
algorithm for irregular data of Bos, de Waele & Broersen (2002).

The AR model lives on a regular grid of spacing `dt`, so the PSD is obtained
with `spectrum(m, dt)` and extends up to `1 / (2 dt)`.  At order `p`, every
pair of observations whose separation lies in `[p dt - slot_width/2,
p dt + slot_width/2)` contributes one product to the reflection coefficient
`k_p`.  The number of products `N_p` decays quickly with `p`, which limits the
attainable order; it is stored in `m.n_products`.

The data mean is removed before fitting (and stored in `m.mu`).  `m.N` is set
to the number of grid points spanned by the observations,
`round((times[end] - times[1]) / dt) + 1`, which fixes the resolution of the
default frequency grid of `spectrum`.

# Arguments
- `times`              : observation times (sorted automatically if needed)
- `data`               : observed values (real-valued `Vector`)
- `dt`                 : grid spacing `T` of the AR model (default: mean
                         sampling interval `1/λ`)
- `slot_width`         : slot width `Δτ` (default: `0.5/λ`).  Smaller values
                         reduce bias but make `N_p` decay faster.
- `m`                  : maximum AR order (default: `2N/log(2N)`); the
                         recursion also stops when no products are left
- `optimisation_method`: `"AICirreg"` (eq. 11, default), `"AICmod"` (eq. 10)
                         or `"Fixed"`
- `penalty`            : penalty factor `α` of `"AICirreg"` (default 4, as in
                         the paper)
- `min_products`       : `"AICirreg"` rejects orders with `N_p` below this
                         (default 15)
- `verbose`            : print iteration progress (default `false`)
- `use_gpu`            : run the pair search and error updates on a CUDA GPU
                         (requires `using CUDA`; default `false`).  Worth it
                         for long series (≳10⁵ observations).

# Returns
`(P, a_k, optimization)` – noise variance, AR coefficients, loss-function
history (entry `i` is the loss of order `i`).
"""
function solve!(mesa::MESA, times::AbstractVector, data::AbstractVector;
                dt::Union{Real, Nothing}=nothing,
                slot_width::Union{Real, Nothing}=nothing,
                m::Union{Int, Nothing}=nothing,
                optimisation_method::String="AICirreg",
                penalty::Real=4.0,
                min_products::Int=15,
                verbose::Bool=false,
                use_gpu::Bool=false)
    t, x = _prepare_irregular(times, data)
    N = length(x)
    N >= 2 || error("At least two observations are required.")
    t[end] > t[1] || error("times must span a non-zero interval.")

    mean_dt = _mean_interval(t)
    dt_use = dt === nothing ? mean_dt : Float64(dt)
    slot_use = slot_width === nothing ? 0.5 * mean_dt : Float64(slot_width)
    dt_use > 0 || error("dt must be positive.")
    slot_use > 0 || error("slot_width must be positive.")
    mmax = _default_mmax(N, m)
    mmax >= 0 || error("m must be non-negative.")

    mesa.mu = mean(x)
    mesa.N = round(Int, (t[end] - t[1]) / dt_use) + 1
    xc = x .- mesa.mu

    if use_gpu
        kss, npss = _irregular_reflection_gpu([t], [xc], dt_use, slot_use, [mmax], verbose)
        ks, n_products = kss[1], npss[1]
    else
        ks, n_products = _irregular_reflection(t, xc, dt_use, slot_use, mmax, verbose)
    end
    return _irregular_fit!(mesa, ks, n_products, var(xc), N, optimisation_method,
                           Float64(penalty), min_products)
end

"""
    mesa_spectrogram(times, x; dt, segment_duration, overlap=0.5,
                     slot_width=nothing, m=nothing,
                     optimisation_method="AICirreg", penalty=4.0,
                     min_products=15, verbose=false, use_gpu=false)

Compute a MESA spectrogram of irregularly sampled data.  The time axis is cut
into overlapping windows of length `segment_duration`, and each window is fit
with the irregular Burg algorithm (see `solve!(m, times, data)`).

All segments share the one-sided grid
`f_k = k / (segment_length * dt)`, `k = 0, …, segment_length ÷ 2 - 1`, with
`segment_length = round(segment_duration / dt)`, so the output matches the
layout of the regularly sampled `mesa_spectrogram`.  The default `slot_width`
is half of the mean sampling interval of the whole series.  Windows with fewer
than two observations yield a column of `NaN`.

With `use_gpu=true` (requires `using CUDA`) all segments are fit at once: each
Burg order is a single batch of GPU kernels over every segment.

# Returns
`(t_centers, f_grid, psd_matrix)` with `psd_matrix` of shape `(n_freq, n_seg)`.
"""
function mesa_spectrogram(times::AbstractVector, x::AbstractVector;
                          dt::Real,
                          segment_duration::Real,
                          overlap::Float64=0.5,
                          slot_width::Union{Real, Nothing}=nothing,
                          m::Union{Int, Nothing}=nothing,
                          optimisation_method::String="AICirreg",
                          penalty::Real=4.0,
                          min_products::Int=15,
                          verbose::Bool=false,
                          use_gpu::Bool=false)
    0.0 <= overlap < 1.0 || error("overlap must be in [0, 1).")
    dt > 0 || error("dt must be positive.")
    t, x = _prepare_irregular(times, x)
    length(t) >= 2 || error("At least two observations are required.")

    dt = Float64(dt)
    segment_length = round(Int, segment_duration / dt)
    segment_length >= 4 || error("segment_duration must span at least 4 * dt.")
    duration = segment_length * dt
    stride = duration * (1.0 - overlap)
    slot_use = slot_width === nothing ? 0.5 * _mean_interval(t) : Float64(slot_width)

    n_seg = floor(Int, (t[end] - t[1] - duration) / stride + 1e-9) + 1
    n_seg >= 1 || error("Time series too short for the requested segment_duration.")
    starts = t[1] .+ stride .* (0:n_seg-1)

    n_freq = segment_length ÷ 2
    f_grid = collect(0:n_freq-1) ./ duration
    psd_matrix = Matrix{Float64}(undef, n_freq, n_seg)
    t_centers = collect(starts .+ duration / 2)

    if use_gpu
        _irregular_spectrogram_gpu!(psd_matrix, t, x, starts, duration, dt,
                                    segment_length, slot_use, m,
                                    optimisation_method, Float64(penalty),
                                    min_products, verbose)
        return t_centers, f_grid, psd_matrix
    end

    verbose_lock = ReentrantLock()

    Threads.@threads for j in 1:n_seg
        lo = searchsortedfirst(t, starts[j])
        hi = searchsortedfirst(t, starts[j] + duration) - 1
        if hi - lo + 1 < 2 || t[hi] == t[lo]
            psd_matrix[:, j] .= NaN
        else
            mj = MESA()
            solve!(mj, view(t, lo:hi), view(x, lo:hi); dt=dt, slot_width=slot_use,
                   m=m, optimisation_method=optimisation_method,
                   penalty=penalty, min_products=min_products)
            mj.N = segment_length
            _, psd_j = spectrum(mj, dt; onesided=true)
            psd_matrix[:, j] = psd_j
        end

        if verbose
            lock(verbose_lock) do
                print("\r  Segment $j / $n_seg")
            end
        end
    end
    verbose && println()

    return t_centers, f_grid, psd_matrix
end

# Batched GPU path of the irregular spectrogram: the Burg recursion of every
# segment runs in one set of kernels, Levinson and order selection on the CPU.
function _irregular_spectrogram_gpu!(psd_matrix::Matrix{Float64},
                                     t::Vector{Float64}, x::Vector{Float64},
                                     starts::AbstractVector, duration::Float64,
                                     dt::Float64, segment_length::Int,
                                     slot_width::Float64, m::Union{Int, Nothing},
                                     optimisation_method::String, penalty::Float64,
                                     min_products::Int, verbose::Bool)
    ranges = map(starts) do s0
        searchsortedfirst(t, s0):searchsortedfirst(t, s0 + duration) - 1
    end
    valid = findall(r -> length(r) >= 2 && t[last(r)] > t[first(r)], ranges)
    psd_matrix[:, setdiff(eachindex(ranges), valid)] .= NaN
    isempty(valid) && return psd_matrix

    mus = [mean(view(x, ranges[j])) for j in valid]
    ts = [t[ranges[j]] for j in valid]
    xs = [x[ranges[j]] .- mu for (j, mu) in zip(valid, mus)]
    mmaxs = [_default_mmax(length(r), m) for r in ts]
    kss, npss = _irregular_reflection_gpu(ts, xs, dt, slot_width, mmaxs, verbose)

    Threads.@threads for s in eachindex(valid)
        j = valid[s]
        mj = MESA()
        mj.mu = mus[s]
        mj.N = segment_length
        _irregular_fit!(mj, kss[s], npss[s], var(xs[s]), length(xs[s]),
                        optimisation_method, penalty, min_products)
        _, psd_j = spectrum(mj, dt; onesided=true)
        psd_matrix[:, j] = psd_j
    end
    return psd_matrix
end
