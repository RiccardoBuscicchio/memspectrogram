**Authors** Alessandro Martini, Stefano Schmidt, Walter del Pozzo, Riccardo Buscicchio

**Licence** CC BY 4.0

**Version** 1.4.0

# Memspectrum.jl — Maximum Entropy Spectral Analysis

`Memspectrum.jl` is a Julia package for Maximum Entropy Spectral Analysis (MESA)
via Burg's algorithm.  It provides two core outputs:

| Output | Function | Description |
|--------|----------|-------------|
| **Memspectrum** | `memspectrum` | Power spectral density (PSD) of a time series |
| **Memgram**     | `memgram`     | Time–frequency spectrogram via overlapping MESA estimates |

The method is fast, reliable, and outperforms classical spectral estimators
(e.g. the periodogram).

> **Python counterpart:** this package is the Julia port of
> [`memspectrum`](https://github.com/martini-alessandro/Maximum-Entropy-Spectral-Analysis),
> the original Python implementation by Alessandro Martini et al.

The PSD is expressed in terms of autoregressive (AR) coefficients `a_k` plus an
overall scale factor `P`.  The AR coefficients are obtained recursively through
the Levinson recursion and characterise the time series as an AR(p) process,
enabling high-quality forecasting.

## Installation

From Julia's package manager:

```julia
using Pkg
Pkg.add(url="https://github.com/RiccardoBuscicchio/memspectrogram")
```

Or, if working from a local clone:

```julia
using Pkg
Pkg.activate(".")   # from the repository root
Pkg.instantiate()
```

## Usage

```julia
using Memspectrum
```

### Compute the Memspectrum (PSD)

```julia
m = MESA()
solve!(m, data)                              # fit AR model (Float64 vector)
f, psd = memspectrum(m, dt)                 # Memspectrum on sampling frequencies
psd_custom = memspectrum(m, dt; frequencies=f_grid)  # on a custom grid
```

### Build an AD-friendly PSD model and Fourier covariance matrix

```julia
m = MESA()
solve!(m, data)

psd_model = MESAPSD(m)
freqs = [10.0, 20.0, 30.0]
cov = frequency_covariance(psd_model, dt; frequencies=freqs)
```

### Compute the Memgram (spectrogram)

```julia
t_centers, f_grid, psd_matrix = memgram(x, dt; segment_length=512)
plt = plot_spectrogram(t_centers, f_grid, psd_matrix)
```

### Irregularly sampled data

`solve!` and `memgram` accept observation times as an extra argument and then
use the Burg algorithm for irregularly sampled data of Bos, de Waele &
Broersen (2002).  The AR model is defined on a regular grid of spacing `dt`
(default: the mean sampling interval); pairs of observations whose separation
is within `slot_width/2` of `p*dt` are used to estimate the `p`-th reflection
coefficient.  Order selection defaults to the paper's `AICirreg` criterion.

```julia
m = MESA()
solve!(m, times, x; dt=1.0, slot_width=0.5)   # irregular times
f, psd = memspectrum(m, 1.0; onesided=true)   # PSD up to 1/(2 dt)
m.n_products                                   # products used for each k_p

t_c, f_grid, S = memgram(times, x; dt=1.0, segment_duration=512.0)
```

### Forecast future observations

```julia
predicted = forecast(m, data, 100; number_of_simulations=1000)
# predicted has shape (1000, 100)
```

### Whiten data

```julia
white_data = whiten(m, data)
```

### Generate coloured noise matching a template PSD

```julia
t, ts, freqs, fs, psd_interp = generate_data(f, psd_template, T;
                                              sampling_rate=4096.0, seed=0)
```

### Save / load a fitted model

```julia
save_mesa(m, "model.txt")
m2 = load_mesa("model.txt")
```

### GPU acceleration

Load `CUDA.jl` before `Memspectrum` to enable GPU-accelerated `forecast`
and `memgram`:

```julia
using CUDA
using Memspectrum

t, f, S = memgram(x, dt; segment_length=512, use_gpu=true)
sims = forecast(m, data, 1000; number_of_simulations=2048, use_gpu=true)

# Irregularly sampled data
solve!(m, times, x; dt=1.0, slot_width=0.5, use_gpu=true)
t, f, S = memgram(times, x; dt=1.0, segment_duration=512.0, use_gpu=true)
```

For irregular data the GPU runs the pair search and prediction-error updates
of every Burg order (all spectrogram segments in one batch, in `Float64`) and
gives the same pairs and product counts as the CPU.  It pays off for long
single series (≳10⁶ observations); for spectrograms of short segments, or on
GPUs with weak double-precision throughput, the threaded CPU path
(`julia -t auto`) can be faster.  Calling `use_gpu=true` without `using CUDA`
raises an error.

## Examples

### Memspectrum — PSD estimate vs true AR(2) spectrum

![Toy PSD estimate](examples/toy_psd_estimate.png)

### Memgram — non-stationary AR(2) signal

![Toy Memgram](examples/toy_spectrogram.png)

### Memgram — linear chirp signal

![Chirp Memgram](examples/chirp_spectrogram.png)

### Memgram — GW150914 (real LIGO data)

![GW150914 Memgram](examples/gw150914_spectrogram.png)

### Memgram — GW170817 (real LIGO data)

![GW170817 Memgram](examples/gw170817_spectrogram.png)

Run the example scripts from the repository root:

```sh
julia --project=. examples/toy_psd_estimate.jl
julia --project=. examples/ar_covariance_ad.jl
julia --project=. examples/toy_spectrogram.jl
julia --project=. examples/chirp_spectrogram.jl
julia --project=. examples/gw150914_spectrogram.jl
julia --project=. examples/gw170817_spectrogram.jl
```

Every example accepts command-line flags **and** an optional TOML config file:

```sh
julia --project=. examples/toy_psd_estimate.jl \
    --config examples/configs/toy_psd_estimate.toml
```

Generate LIGO-like noise from the O3 design PSD:

```sh
julia examples/generate_white_noise.jl --p 300 --t 32 --srate 4096
```

## References

- Original Burg's algorithm: [J.P. Burg – Maximum Entropy Spectral Analysis](http://sepwww.stanford.edu/data/media/public/oldreports/sep06/)
- Fast implementation: [V. Fastubrg – A Fast Implementation of Burg Method](https://svn.xiph.org/websites/opus-codec.org/docs/vos_fastburg.pdf)
- Irregular sampling: [R. Bos, S. de Waele, P. M. T. Broersen – Autoregressive spectral estimation by application of the Burg algorithm to irregularly sampled data, IEEE Trans. Instrum. Meas. 51, 1289 (2002)](https://doi.org/10.1109/TIM.2002.808031)
- Method paper: [Maximum Entropy Spectral Analysis: a case study](https://arxiv.org/abs/2106.09499)
- Python package: [martini-alessandro/Maximum-Entropy-Spectral-Analysis](https://github.com/martini-alessandro/Maximum-Entropy-Spectral-Analysis)
