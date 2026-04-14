**Authors** Alessandro Martini, Stefano Schmidt, Walter del Pozzo, Riccardo Buscicchio

**Licence** CC BY 4.0

**Version** 1.3.0

# MAXIMUM ENTROPY SPECTRAL ANALYSIS FOR ACCURATE PSD COMPUTATION

`Memspectrum` is a Julia package for computing the power spectral density (PSD)
of time series using Maximum Entropy Spectral Analysis (MESA) via Burg's algorithm.
The method is fast and reliable and shows better performance than other standard methods.

The maximum entropy spectral estimation is based on the maximum entropy principle.
The PSD is expressed in terms of a set of autoregressive (AR) coefficients `a_k` plus
an overall scale factor `P`. The AR coefficients are obtained recursively through the
Levinson recursion and characterise the time series as an AR(p) process, enabling
high-quality forecasting.

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
include("src/Memspectrum.jl")
using .Memspectrum
```

### Compute the PSD

```julia
m = MESA()
solve!(m, data)                       # fit AR model (data is a Float64 vector)
f, psd = spectrum(m, dt)              # PSD on sampling frequencies
psd_custom = spectrum(m, dt; frequencies=f_grid)  # PSD on custom grid
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

## Example

```julia
include("src/Memspectrum.jl")
using .Memspectrum

N, dt = 1000, 0.01
t = range(0, N*dt, length=N)
data = sin.(2π * 2 .* t) .+ 0.4 .* randn(N)

m = MESA()
solve!(m, data)
f, psd = spectrum(m, dt; onesided=true)
```

See `examples/generate_white_noise.jl` for a complete example generating
LIGO-like noise from the O3 design PSD.

```
julia examples/generate_white_noise.jl --p 300 --t 32 --srate 4096
```

## References

- Original Burg's algorithm: [J.P. Burg – Maximum Entropy Spectral Analysis](http://sepwww.stanford.edu/data/media/public/oldreports/sep06/)
- Fast implementation: [V. Fastubrg – A Fast Implementation of Burg Method](https://svn.xiph.org/websites/opus-codec.org/docs/vos_fastburg.pdf)
- Method paper: [Maximum Entropy Spectral Analysis: a case study](https://arxiv.org/abs/2106.09499)
