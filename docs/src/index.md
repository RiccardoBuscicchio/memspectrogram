# Memspectrum.jl

**Authors:** Alessandro Martini, Stefano Schmidt, Walter del Pozzo, Riccardo Buscicchio

**Licence:** CC BY 4.0 · **Version:** 1.4.0

---

`Memspectrum.jl` is a Julia package for Maximum Entropy Spectral Analysis (MESA)
via Burg's algorithm.  It provides two main outputs:

| Name | Function | Description |
|------|----------|-------------|
| **Memspectrum** | [`memspectrum`](@ref) | Power spectral density (PSD) of a time series |
| **Memgram**     | [`memgram`](@ref)     | Time–frequency spectrogram computed from overlapping MESA estimates |

It is the Julia counterpart of the original Python package
[`memspectrum`](https://github.com/martini-alessandro/Maximum-Entropy-Spectral-Analysis).

---

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/RiccardoBuscicchio/memspectrogram")
```

Or from a local clone:

```julia
using Pkg
Pkg.activate(".")   # from the repository root
Pkg.instantiate()
```

---

## Quick start

### Compute the Memspectrum (PSD)

```julia
using Memspectrum

m = MESA()
solve!(m, data)                          # fit AR model
f, psd = memspectrum(m, dt)             # Memspectrum on sampling frequencies
psd_custom = memspectrum(m, dt; frequencies=f_grid)  # on a custom grid
```

### Compute the Memgram (spectrogram)

```julia
t_centers, f_grid, psd_matrix = memgram(x, dt; segment_length=512)
plt = plot_spectrogram(t_centers, f_grid, psd_matrix)
```

### Forecast future observations

```julia
predicted = forecast(m, data, 100; number_of_simulations=1000)
# predicted has shape (1000, 100)
```

### GPU acceleration

Load `CUDA.jl` before `Memspectrum` to enable GPU-accelerated `forecast`
and `memgram`:

```julia
using CUDA
using Memspectrum

t, f, S = memgram(x, dt; segment_length=512, use_gpu=true)
sims = forecast(m, data, 1000; number_of_simulations=2048, use_gpu=true)
```

---

## See also

* [API reference](@ref "API")
* [Examples](@ref "Examples")
