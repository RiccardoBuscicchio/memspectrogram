"""
    Memspectrum

Package that uses Maximum Entropy Spectral Analysis (MESA) to compute the
spectrum of a given time-series, implemented with Burg's algorithm.

Basic usage:

```julia
using Memspectrum

m = MESA()
solve!(m, time_series)
f, psd = spectrum(m, dt)
```
"""
module Memspectrum

using FFTW
using DSP
using LinearAlgebra
using Statistics
using Random
using Interpolations
using DelimitedFiles

export MESA, solve!, spectrum, forecast, whiten, entropy_rate, logL,
       generate_data, save_mesa, load_mesa,
       mesa_spectrogram, plot_spectrogram

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

"""
    loss_function(method, P, a_k, N, m; kwargs...)

Evaluate the order-selection loss function specified by `method`.

Supported methods: `"FPE"`, `"MDL"`, `"AIC"`, `"CAT"`, `"OBD"`, `"Fixed"`.
"""
function loss_function(method::String, P::Vector, a_k, N::Int, m::Int;
                       spec=nothing, k=nothing)
    if method == "FPE"
        return _FPE(P, N, m)
    elseif method == "MDL"
        return _MDL(P, N, m)
    elseif method == "AIC"
        return _AIC(P, N, m)
    elseif method == "CAT"
        return _CAT(P, N, m)
    elseif method == "OBD"
        return _OBD(P, a_k, N, m)
    elseif method == "Fixed"
        return _Fixed(m)
    else
        error("Unknown optimisation method '$method'. " *
              "Valid choices are 'FPE', 'MDL', 'AIC', 'CAT', 'OBD', 'Fixed'.")
    end
end

# Akaike Final Prediction Error
function _FPE(P::Vector, N::Int, m::Int)
    return P[end] * (N + m + 1) / (N - m - 1)
end

# Minimum Description Length
function _MDL(P::Vector, N::Int, m::Int)
    return N * log(P[end]) + m * log(N)
end

# Akaike Information Criterion
function _AIC(P::Vector, N::Int, m::Int)
    return log(P[end]) + 2 * m / N
end

# Parzen's CAT criterion
function _CAT(P::Vector, N::Int, m::Int)
    m == 0 && return Inf
    Pv = P[2:end]  # P[1] is P_0; skip it (same as Python P[1:])
    k_vec = 1:m
    PW_k = (N .- k_vec) ./ (N .* Pv)
    return sum(PW_k) / N - PW_k[end]
end

# Rao's Optimum Bayes Decision rule
function _OBD(P::Vector, a_k, N::Int, m::Int)
    P_m = P[end]
    Pv = P[1:end-1]
    return (N - m - 2) * log(abs(P_m)) + m * log(N) + sum(log.(abs.(Pv))) +
           sum(abs2, a_k)
end

# Fixed order (monotonically decreasing loss → runs to specified m)
function _Fixed(m::Int)
    return 1.0 / (m + 1)
end

# ---------------------------------------------------------------------------
# MESA struct
# ---------------------------------------------------------------------------

"""
    MESA

Mutable struct that implements Maximum Entropy Spectral Analysis via Burg's
algorithm.

Fields set after calling `solve!`:
- `P`   – variance of the white-noise innovation
- `a_k` – autoregressive (AR) coefficients (length p+1, with `a_k[1] = 1`)
- `N`   – length of the time series used to fit the model
- `mu`  – mean of the time series
"""
mutable struct MESA
    P::Union{Float64, ComplexF64, Nothing}
    a_k::Union{Vector{Float64}, Vector{ComplexF64}, Nothing}
    N::Union{Int, Nothing}
    mu::Union{Float64, ComplexF64, Nothing}
    ref_coefficients::Vector
    optimization::Union{Vector, Nothing}

    MESA() = new(nothing, nothing, nothing, nothing, Float64[], nothing)
end

"""
    p(m::MESA) -> Int

Return the autoregressive order of the fitted model.
"""
function Base.getproperty(m::MESA, s::Symbol)
    s == :p && return length(m.a_k) - 1
    return getfield(m, s)
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

function _update_prediction_coefficient(x::Vector, k)
    new_x = vcat(x, zero(eltype(x)))
    return new_x .+ k .* conj.(reverse(new_x))
end

function _update_coefficients_fast(a::Vector, g::Vector)
    a = vcat(a, zero(eltype(a)))
    k = -dot(conj.(a), reverse(g)) / dot(a, g)
    aUpd = a .+ k .* conj.(reverse(a))
    return k, aUpd
end

function _update_r(i::Int, r::Vector, aCorr, data::AbstractVector, N::Int)
    # Julia is 1-indexed; Python's data[:i+1] = data[1:i+1], data[i+1] = data[i+2]
    # Python's data[N-i-1:] = data[N-i:N], data[N-i-2] = data[N-i-1]
    rUp   = [2 * aCorr]
    rDown = r .- data[1:i+1] .* conj(data[i+2]) .-
            conj.(reverse(data[N-i:N])) .* data[N-i-1]
    return vcat(rUp, rDown)
end

function _construct_dr2(i::Int, a::Vector, data::AbstractVector, N::Int)
    # Python's data[:i+2][::-1] = reverse of data[1:i+2] (Julia)
    # Python's data[N-i-2:]     = data[N-i-1:N]   (Julia)
    data1 = reverse(data[1:i+2])
    data2 = data[N-i-1:N]
    d1 = -data1 .* conj(dot(data1, a))
    d2 = -conj.(data2) .* dot(data2, conj.(a))
    return d1 .+ d2
end

function _update_g(g::Vector, k, r::Vector, a::Vector, dra::Vector)
    gUp   = g .+ conj.(k .* reverse(g)) .+ dra
    gDown = [dot(r, conj.(a))]
    return vcat(gUp, gDown)
end

function _spectrum_internal(dt::Float64, N::Int, P, a_k::Vector)
    # Zero-pad a_k to length N, then FFT
    padded = vcat(a_k, zeros(eltype(a_k), N - length(a_k)))
    den = fft(padded)
    spec = dt .* real(P) ./ abs2.(den)
    # Frequencies: same convention as numpy.fft.fftfreq(N, dt)
    freqs = [(k <= N ÷ 2 ? k : k - N) for k in 0:N-1] ./ (N * dt)
    return freqs, spec
end

# ---------------------------------------------------------------------------
# FastBurg algorithm
# ---------------------------------------------------------------------------

function _fast_burg!(mesa::MESA, data::Vector, mmax::Int,
                     optimisation_method::String, regularisation::Float64,
                     early_stop::Bool, verbose::Bool)
    N = mesa.N

    # Full autocorrelation; center (lag-0) is at index N (1-indexed)
    full_c = xcorr(data, data)
    # Take lags 0 .. mmax+1 → indices N .. N+mmax+1 (Julia 1-indexed)
    c = full_c[N:N+mmax+1]
    c[1] *= regularisation

    # Initialise
    a  = [ComplexF64[1.0 + 0im]]
    P  = [c[1] / N]
    r  = ComplexF64[2 * c[2]]
    g  = ComplexF64[2 * c[1] - data[1] * conj(data[1]) - data[end] * conj(data[end]),
                    r[1]]

    optimization = Float64[]
    idx     = 1
    old_idx = 1

    for i in 0:mmax-1
        if verbose
            print("\r\tIteration $(i+1) of $mmax")
            flush(stdout)
        end

        k, new_a = _update_coefficients_fast(a[end], g)
        r_new    = _update_r(i, r, c[i+3], data, N)   # c[i+3] = lag i+2
        dra      = _construct_dr2(i, new_a, data, N)
        g        = _update_g(g, k, r_new, new_a, dra)
        r        = r_new

        push!(a, new_a)
        push!(mesa.ref_coefficients, k)
        push!(P, P[end] * (1 - k * conj(k)))

        lv = loss_function(optimisation_method, P, new_a, N, i + 1)
        push!(optimization, lv)

        has_nan = any(isnan, new_a)
        if abs(k) > 1 || has_nan
            @warn "Numerical stability issue at order $(i+1). Results may be unreliable."
        end

        if has_nan && !early_stop
            idx = argmin(optimization)
            break
        end
        if ((i % 100 == 0 && i != 0) || i >= mmax - 1) && early_stop
            idx = argmin(optimization)
            if old_idx < idx
                old_idx = idx
            else
                break
            end
        end
    end

    if !early_stop
        idx = argmin(optimization)
    end

    verbose && println()

    mesa.P   = real(P[idx + 1])
    mesa.a_k = real(a[idx + 1])
    mesa.optimization = optimization
    return mesa.P, mesa.a_k, optimization
end

# ---------------------------------------------------------------------------
# Standard Burg algorithm
# ---------------------------------------------------------------------------

function _standard_burg!(mesa::MESA, data::Vector, mmax::Int,
                         optimisation_method::String,
                         early_stop::Bool, verbose::Bool)
    N = mesa.N

    P_0 = var(data)
    P   = [P_0]
    a_k = [Float64[1.0]]
    _f  = copy(data)
    _b  = copy(data)

    optimization = Float64[]
    idx     = 1
    old_idx = 1

    for i in 0:mmax-1
        if verbose
            print("\r\tIteration $(i+1) of $mmax")
            flush(stdout)
        end

        f = _f[2:end]
        b = _b[1:end-1]

        den = sum(abs2, f) + sum(abs2, b)
        k   = -2 * dot(f, b) / den

        push!(a_k, _update_prediction_coefficient(a_k[end], k))
        push!(P, P[end] * (1 - k * conj(k)))
        push!(mesa.ref_coefficients, k)

        _f = f .+ k .* b
        _b = b .+ k .* f

        lv = loss_function(optimisation_method, P, a_k[end], N, i + 1)
        push!(optimization, lv)

        if ((i % 100 == 0 && i != 0) || i >= mmax - 1) && early_stop
            idx = argmin(optimization)
            if old_idx < idx && optimization[idx] * 1.01 < optimization[old_idx]
                old_idx = idx
            else
                old_idx = idx
                break
            end
        end
    end

    if !early_stop
        idx = argmin(optimization)
    end

    verbose && println()

    mesa.P   = real(P[idx + 1])
    mesa.a_k = real(a_k[idx + 1])
    mesa.optimization = optimization
    return mesa.P, mesa.a_k, optimization
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    solve!(m::MESA, data; m=nothing, optimisation_method="FPE",
           method="Fast", regularisation=1.0, early_stop=true, verbose=false)

Fit the MESA model to `data` using Burg's algorithm.

# Arguments
- `data`               : one-dimensional time-series (real-valued `Vector`)
- `m`                  : maximum AR order (default: `2N/log(2N)`)
- `optimisation_method`: order selection criterion – `"FPE"` (default), `"MDL"`,
                         `"AIC"`, `"CAT"`, `"OBD"`, or `"Fixed"`
- `method`             : algorithm variant – `"Fast"` (default) or `"Standard"`
- `regularisation`     : Tikhonov regularisation factor (default 1.0 = none)
- `early_stop`         : stop when no improvement for 100 iterations (default `true`)
- `verbose`            : print iteration progress (default `false`)

# Returns
`(P, a_k, optimization)` – noise variance, AR coefficients, loss-function history.
"""
function solve!(mesa::MESA, data::AbstractVector;
                m::Union{Int, Nothing}=nothing,
                optimisation_method::String="FPE",
                method::String="Fast",
                regularisation::Float64=1.0,
                early_stop::Bool=true,
                verbose::Bool=false)

    data = Float64.(vec(data))
    N    = length(data)
    mesa.N   = N
    mesa.mu  = mean(data)
    mesa.ref_coefficients = Float64[]

    mmax = m === nothing ? Int(floor(2 * N / log(2 * N))) : m
    if optimisation_method == "Fixed"
        early_stop = false
    end

    if lowercase(method) == "fast"
        return _fast_burg!(mesa, data, mmax, optimisation_method,
                           regularisation, early_stop, verbose)
    elseif lowercase(method) == "standard"
        return _standard_burg!(mesa, data, mmax, optimisation_method,
                               early_stop, verbose)
    else
        error("Unknown method '$method'. Valid choices are 'Fast' and 'Standard'.")
    end
end

"""
    spectrum(m::MESA, dt=1.0; frequencies=nothing, onesided=false)

Compute the power spectral density of the fitted MESA model.

# Arguments
- `dt`          : sampling interval
- `frequencies` : custom frequency grid (if `nothing`, a standard grid is used)
- `onesided`    : return only positive frequencies (default `false`)

# Returns
If `frequencies === nothing`: `(f, psd)` tuple.
If `frequencies` is a `Vector`: interpolated PSD values on that grid.
"""
function spectrum(mesa::MESA, dt::Float64=1.0;
                  frequencies::Union{Vector, Nothing}=nothing,
                  onesided::Bool=false)
    mesa.a_k === nothing &&
        error("Model not fitted. Call solve! first.")

    f_ny = 0.5 / dt
    f_spec, spec = _spectrum_internal(dt, mesa.N, mesa.P, mesa.a_k)

    if frequencies === nothing
        if onesided
            half = mesa.N ÷ 2
            return f_spec[1:half], spec[1:half] .* 2
        else
            return f_spec, spec
        end
    elseif frequencies isa AbstractVector
        if maximum(frequencies) > f_ny * 1.01
            @warn "Some requested frequencies exceed Nyquist ($f_ny Hz): returning zero PSD there."
        end
        half = mesa.N ÷ 2
        f_interp = LinearInterpolation(f_spec[1:half], spec[1:half],
                                       extrapolation_bc=Flat())
        return f_interp.(clamp.(frequencies, f_spec[1], f_spec[half]))
    else
        error("'frequencies' must be nothing or a Vector.")
    end
end

"""
    forecast(m::MESA, data, length; number_of_simulations=1,
             P=nothing, include_data=false, seed=nothing, verbose=false)

Forecast `length` new points from the fitted AR process.

# Arguments
- `data`                  : seed data (at least `p` points, where `p = m.p`)
- `length`                : number of future points to generate
- `number_of_simulations` : number of independent realisations
- `P`                     : innovation variance (default: use fitted `m.P`)
- `include_data`          : prepend seed data to output (default `false`)
- `seed`                  : random seed for reproducibility

# Returns
Matrix of shape `(number_of_simulations, length)`, or
`(number_of_simulations, p + length)` when `include_data = true`.
"""
function forecast(mesa::MESA, data::AbstractVector, len::Int;
                  number_of_simulations::Int=1,
                  P=nothing,
                  include_data::Bool=false,
                  seed::Union{Int, Nothing}=nothing,
                  verbose::Bool=false)
    (mesa.P === nothing || mesa.a_k === nothing) &&
        error("Model not fitted. Call solve! before forecast.")

    P_use = P === nothing ? mesa.P : P
    p     = length(mesa.a_k) - 1
    predictions = zeros(number_of_simulations, p + len)

    data = Float64.(vec(data))
    if length(data) >= p > 0
        predictions[:, 1:p] .= data[end-p+1:end]'
    elseif p != 0
        error("Data too short for forecasting: need at least $p points.")
    end

    seed !== nothing && Random.seed!(seed)

    coef = -reverse(mesa.a_k[2:end])
    sigma = sqrt(P_use)
    for i in 1:len
        verbose && print("\r $(i) of $(len)")
        predictions[:, p + i] = predictions[:, i:i+p-1] * coef .+
                                 randn(number_of_simulations) .* sigma
    end
    verbose && println()

    return include_data ? predictions : predictions[:, p+1:end]
end

"""
    whiten(m::MESA, data; trim=nothing, zero_phase=false)

Whiten `data` by convolving with the AR filter coefficients `a_k`.

# Arguments
- `data`       : time series to whiten
- `trim`       : points to remove at each end (default: `m.p`)
- `zero_phase` : apply zero-phase filtering (default `false`)

# Returns
Whitened (and trimmed) time series.
"""
function whiten(mesa::MESA, data::AbstractVector;
                trim::Union{Int, Nothing}=nothing, zero_phase::Bool=false)
    data = Float64.(vec(data))
    if !zero_phase
        white_data = conv(data, mesa.a_k) ./ sqrt(mesa.P)
        # conv gives length n+m-1; take central portion same as 'same' mode
        n = length(data)
        p = length(mesa.a_k) - 1
        white_data = white_data[p+1:p+n]
    else
        c = xcorr(mesa.a_k, mesa.a_k)
        c ./= maximum(abs, c)
        white_data = conv(c, data) ./ sqrt(mesa.P)
        n = length(data)
        white_data = white_data[length(mesa.a_k):length(mesa.a_k)+n-1]
    end
    t = trim === nothing ? (length(mesa.a_k) - 1) : trim
    if t > 0
        white_data = white_data[t+1:end-t]
    end
    return white_data
end

"""
    entropy_rate(m::MESA, dt)

Compute the entropy rate Δ H = ∫ log S(f) df for the fitted power spectrum.
"""
function entropy_rate(mesa::MESA, dt::Float64)
    f, psd = spectrum(mesa, dt)
    df = f[2] - f[1]
    return sum(log.(psd)) * df / (4 * maximum(f)) + 0.5 * log(2π * ℯ)
end

"""
    logL(m::MESA, data, dt)

Compute the log-likelihood of `data` under the fitted MESA power spectrum.
"""
function logL(mesa::MESA, data::AbstractVector, dt::Float64)
    N    = length(data)
    f    = rfftfreq(N, 1 / dt)
    psd  = spectrum(mesa, dt; frequencies=collect(f), onesided=true)
    d    = rfft(Float64.(vec(data))) .* dt
    TwoDeltaTOverN = 2 * dt / N
    return -TwoDeltaTOverN * real(dot(d, d ./ (psd .* dt^2))) -
           0.5 * sum(log.(0.5 .* π .* N .* dt .* psd))
end

"""
    save_mesa(m::MESA, filename)

Save the fitted model to a text file.
"""
function save_mesa(mesa::MESA, filename::String)
    (mesa.P === nothing || mesa.a_k === nothing) &&
        error("Model not fitted. Call solve! before saving.")
    header = "(1,1,1,$(length(mesa.a_k)))"
    data   = vcat([mesa.P], [Float64(mesa.N)], [mesa.mu], mesa.a_k)
    open(filename, "w") do io
        println(io, "# $header")
        for v in data
            println(io, v)
        end
    end
end

"""
    load_mesa(filename) -> MESA

Load a MESA model from a file saved with `save_mesa`.
"""
function load_mesa(filename::String)
    lines  = readlines(filename)
    header = lines[1]
    shapes = eval(Meta.parse(replace(replace(header, "#" => ""), " " => "")))
    data   = parse.(Float64, filter(!isempty, lines[2:end]))

    mesa = MESA()
    idx  = 1
    mesa.P   = data[idx]; idx += shapes[1]
    mesa.N   = Int(data[idx]); idx += shapes[2]
    mesa.mu  = data[idx]; idx += shapes[3]
    mesa.a_k = data[idx:idx+shapes[4]-1]
    return mesa
end

# ---------------------------------------------------------------------------
# generate_data – equivalent of GenerateTimeSeries.py
# ---------------------------------------------------------------------------

"""
    generate_data(f, psd, T; sampling_rate=1.0, fmin=nothing, fmax=nothing,
                  asd=false, seed=nothing)

Generate a time series whose power spectral density matches a given template.

# Arguments
- `f`            : frequency array at which the template PSD is evaluated
- `psd`          : template PSD (or ASD if `asd = true`)
- `T`            : duration of the output time series (seconds)
- `sampling_rate`: output sampling rate (Hz), default 1.0
- `fmin`         : lower frequency cut-off (default: 0)
- `fmax`         : upper frequency cut-off (default: Nyquist)
- `asd`          : if `true`, interpret `psd` as amplitude spectral density
- `seed`         : random seed for reproducibility

# Returns
`(times, time_series, frequencies, frequency_series, interpolated_psd)`
"""
function generate_data(f_arr::AbstractVector, psd_arr::AbstractVector, T::Float64;
                       sampling_rate::Float64=1.0,
                       fmin::Union{Float64, Nothing}=nothing,
                       fmax::Union{Float64, Nothing}=nothing,
                       asd::Bool=false,
                       seed::Union{Int, Nothing}=nothing)

    seed !== nothing && Random.seed!(seed)
    psd_use = asd ? psd_arr .^ 2 : copy(psd_arr)

    # Build interpolant with linear extrapolation outside data range
    psd_interp = LinearInterpolation(f_arr, psd_use, extrapolation_bc=Line())

    df  = 1.0 / T
    N   = Int(sampling_rate * T)
    times = range(0.0, T, length=N)

    fmin_val = fmin === nothing ? 0.0 : fmin
    fmax_val = fmax === nothing ? (N / 2) / T : fmax

    kmin = Int(floor(fmin_val / df))
    kmax = Int(floor(fmax_val / df)) + 1

    frequencies    = df .* collect(kmin:kmax-1)
    psd_at_f       = psd_interp.(frequencies)
    sigma          = sqrt.(psd_at_f ./ df .* 0.5)
    frequency_series = sigma .* (randn(length(sigma)) .+ im .* randn(length(sigma)))

    # Inverse FFT – output N real points
    time_series = irfft(frequency_series, N) .* df .* N

    return collect(times), time_series, frequencies, frequency_series, psd_at_f
end

# ---------------------------------------------------------------------------
# MESA Spectrogram
# ---------------------------------------------------------------------------

"""
    mesa_spectrogram(x, dt; segment_length, overlap=0.5,
                     optimisation_method="FPE", method="Fast",
                     verbose=false)

Compute a MESA-based spectrogram by fitting an AR model on overlapping
segments of `x` and collecting the resulting one-sided PSDs.

# Arguments
- `x`                   : input time series (real-valued `Vector`)
- `dt`                  : sampling interval (seconds)
- `segment_length`      : number of samples per segment
- `overlap`             : fractional overlap between consecutive segments
                          (0 = no overlap, 0.5 = 50 %, default 0.5)
- `optimisation_method` : AR order-selection criterion passed to `solve!`
                          (default `"FPE"`)
- `method`              : Burg variant passed to `solve!`
                          (default `"Fast"`)
- `verbose`             : print per-segment progress (default `false`)

# Returns
`(t_centers, f_grid, psd_matrix)` where
- `t_centers`  : vector of length `n_seg` with the time (in seconds) at the
                 centre of each segment
- `f_grid`     : one-sided frequency grid (Hz), length `n_freq`
- `psd_matrix` : matrix of shape `(n_freq, n_seg)`; each column is the
                 one-sided PSD of the corresponding segment
"""
function mesa_spectrogram(x::AbstractVector, dt::Float64;
                          segment_length::Int,
                          overlap::Float64=0.5,
                          optimisation_method::String="FPE",
                          method::String="Fast",
                          verbose::Bool=false)
    0.0 <= overlap < 1.0 ||
        error("overlap must be in [0, 1).")
    segment_length >= 4 ||
        error("segment_length must be at least 4.")

    x = Float64.(vec(x))
    N  = length(x)
    stride = max(1, round(Int, segment_length * (1.0 - overlap)))

    starts = collect(1 : stride : N - segment_length + 1)
    n_seg  = length(starts)
    n_seg >= 1 || error("Time series too short for the requested segment_length.")

    # Build common one-sided frequency grid from the first segment
    seg1 = x[starts[1] : starts[1] + segment_length - 1]
    m1   = MESA()
    solve!(m1, seg1; method=method, optimisation_method=optimisation_method,
           verbose=false)
    f_grid, psd1 = spectrum(m1, dt; onesided=true)
    n_freq = length(f_grid)

    psd_matrix = Matrix{Float64}(undef, n_freq, n_seg)
    psd_matrix[:, 1] = psd1

    t_centers = Vector{Float64}(undef, n_seg)
    t_centers[1] = (starts[1] - 1 + 0.5 * segment_length) * dt

    for (j, s) in enumerate(starts[2:end])
        seg = x[s : s + segment_length - 1]
        t_centers[j + 1] = (s - 1 + 0.5 * segment_length) * dt

        verbose && print("\r  Segment $(j+1) / $n_seg")

        mj = MESA()
        solve!(mj, seg; method=method, optimisation_method=optimisation_method,
               verbose=false)
        # Evaluate on the common frequency grid
        psd_matrix[:, j + 1] = spectrum(mj, dt;
                                        frequencies=collect(f_grid),
                                        onesided=true)
    end
    verbose && println()

    return t_centers, f_grid, psd_matrix
end

"""
    plot_spectrogram(t_centers, f_grid, psd_matrix;
                     title="MESA spectrogram", clim=nothing,
                     size=(900, 500), dpi=150)

Plot a MESA spectrogram as a heat-map with time on the x-axis and
frequency on the y-axis (vertical).  Colour encodes `log10(PSD)`.

# Arguments
- `t_centers`  : time centres of each segment (seconds)
- `f_grid`     : one-sided frequency grid (Hz)
- `psd_matrix` : `(n_freq, n_seg)` matrix returned by `mesa_spectrogram`
- `title`      : plot title (default `"MESA spectrogram"`)
- `clim`       : optional `(lo, hi)` colour-axis limits in `log10` units
- `size`       : figure size in pixels (default `(900, 500)`)
- `dpi`        : figure DPI (default `150`)

# Returns
The `Plots.Plot` object.
"""
function plot_spectrogram(t_centers::AbstractVector, f_grid::AbstractVector,
                          psd_matrix::AbstractMatrix;
                          title::String="MESA spectrogram",
                          clim::Union{Tuple{<:Real,<:Real}, Nothing}=nothing,
                          size::Tuple{Int,Int}=(900, 500),
                          dpi::Int=150)
    log_psd = log10.(max.(psd_matrix, 1e-300))   # avoid log(0)

    kw = (xlabel="Time (s)", ylabel="Frequency (Hz)",
          title=title, colorbar_title="log₁₀ PSD",
          size=size, dpi=dpi)

    if clim !== nothing
        return heatmap(t_centers, f_grid, log_psd; clims=clim, kw...)
    else
        return heatmap(t_centers, f_grid, log_psd; kw...)
    end
end

end # module

