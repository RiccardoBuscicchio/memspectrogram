# API Reference

## Fitting the model

```@docs
MESA
MESAPSD
solve!
```

## Memspectrum — Power Spectral Density

```@docs
spectrum
memspectrum
frequency_covariance
```

## Irregularly sampled data

`solve!(m, times, data; ...)` and `mesa_spectrogram(times, x; ...)` (alias
`memgram`) implement the Burg algorithm for irregularly sampled data; their
docstrings are listed with [`solve!`](@ref) above and
[`mesa_spectrogram`](@ref) below.  See [Irregularly sampled data](irregular.md) for a
guide.

## Memgram — Spectrogram

```@docs
mesa_spectrogram
memgram
plot_spectrogram
```

## Forecasting

```@docs
forecast
```

## Whitening

```@docs
whiten
```

## Likelihood and entropy

```@docs
entropy_rate
logL
```

## Data generation

```@docs
generate_data
```

## Saving and loading

```@docs
save_mesa
load_mesa
```
