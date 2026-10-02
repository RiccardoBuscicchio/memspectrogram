# Irregularly sampled data

`Memspectrum` can fit an autoregressive (AR) model directly to unevenly
sampled data, without resampling or interpolation, using the Burg algorithm for
irregularly sampled data of Bos, de Waele & Broersen[^Bos2002].  The estimated
spectrum is always positive and the fitted model is always stationary.

## Quick start

```julia
using Memspectrum

m = MESA()
solve!(m, times, x; dt=1.0, slot_width=0.5)   # times need not be sorted
f, psd = memspectrum(m, 1.0; onesided=true)   # same dt as the fit

m.p             # selected AR order
m.n_products    # number of products N_p behind each reflection coefficient

# Spectrogram: windows of fixed duration along the time axis
t_c, f_grid, S = memgram(times, x; dt=1.0, segment_duration=512.0, overlap=0.5)
plot_spectrogram(t_c, f_grid, S)
```

Calling `solve!` or `memgram` with a vector of times before the data selects
the irregular algorithm; the regularly sampled methods are unchanged.

## How it works

The AR model is defined on a regular grid of spacing `dt` (``T`` in the paper),
so its PSD extends up to ``1/(2T)``.  Irregular data almost never contain two
points exactly ``pT`` apart, but they do contain pairs separated by
``pT \pm \Delta\tau/2``, where ``\Delta\tau`` is the *slot width*.  At order
``p`` the reflection coefficient is

```math
k_p = \frac{-2 \sum f_{p-1}(t_j)\, b_{p-1}(t_i)}
           {\sum f_{p-1}(t_j)^2 + \sum b_{p-1}(t_i)^2},
\qquad t_j - t_i \in \left[pT - \tfrac{\Delta\tau}{2},\; pT + \tfrac{\Delta\tau}{2}\right),
```

with ``f_0 = b_0 = x - \bar{x}``.  Every such pair contributes one *product*;
their number ``N_p`` is stored in `m.n_products`.  Only points that took part
in estimating ``k_p`` carry prediction errors to the next order:

```math
f_p(t_j) = f_{p-1}(t_j) + k_p\, b_{p-1}(t_i), \qquad
b_p(t_i) = b_{p-1}(t_i) + k_p\, f_{p-1}(t_j).
```

When a point has several candidate partners, the one whose separation is
closest to ``pT`` is used.  On a regular grid with ``\Delta\tau \le T`` the
method reduces exactly to the standard Burg algorithm.

The algorithm can be seen as finding sequences of points that are spaced
*almost* regularly.  The probability of finding such a sequence drops roughly
exponentially with its length, so ``N_p`` decays quickly with ``p``, and high
orders need many observations.  The recursion stops when no products are left
or the maximum order `m` is reached.

## Choosing `dt` and `slot_width`

Let ``\lambda`` be the mean number of observations per unit time.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `dt` | ``1/\lambda`` (mean interval) | Sets the maximum frequency ``1/(2\,dt)``. Irregular data have no hard Nyquist limit, so smaller `dt` is allowed, but it needs more data. |
| `slot_width` | ``0.5/\lambda`` | Larger values give more products (lower variance) but more bias, especially at high frequencies. Smaller values give less bias, but ``N_p`` decays faster, which limits the order. |

The paper uses ``\Delta\tau \in \{0.25, 0.5, 1\}/\lambda``.  In practice keep
``\Delta\tau \lesssim 0.5/\lambda``: wider slots produce erratic high-order
reflection coefficients and spurious peaks, and the order selection may still
pick those orders.

## Order selection

| `optimisation_method` | Criterion |
|-----------------------|-----------|
| `"AICirreg"` (default) | ``\ln \mathrm{RES}(p) + \alpha \sum_{i=0}^{p} 1/N_i``, and ``\infty`` if ``N_p <`` `min_products` (eq. 11) |
| `"AICmod"` | the same with ``\alpha = 2`` and no product threshold (eq. 10) |
| `"Fixed"` | always use order `m` (or the highest order reached) |

Here ``\mathrm{RES}(p) = \sigma_x^2 \prod_{i=1}^p (1 - k_i^2)`` and
``N_0 = N``.  The defaults `penalty = 4` (``\alpha``) and `min_products = 15`
follow the paper.  Order 0 (white noise) is selected if no higher order beats
it.  `m.optimization[i]` holds the criterion value for order `i`.

## Fitted model

After `solve!(m, times, x; dt=...)`:

- `m.P`, `m.a_k`: AR model on the grid of spacing `dt`; pass the same `dt` to
  [`spectrum`](@ref)/[`memspectrum`](@ref).
- `m.mu`: the data mean, removed before fitting.
- `m.N`: number of grid points spanned by the data,
  `round((times[end] - times[1]) / dt) + 1`; it sets the resolution of the
  default frequency grid.
- `m.ref_coefficients`, `m.n_products`: ``k_p`` and ``N_p`` for every order
  computed (not only up to the selected one).

Since the result is an ordinary AR model, [`MESAPSD`](@ref),
[`frequency_covariance`](@ref) and the other model-based functions work as
usual.  Functions that take regularly sampled data as input, such as
[`whiten`](@ref), [`forecast`](@ref) and [`logL`](@ref), assume a regular
series with spacing `dt`.

## Spectrogram

`memgram(times, x; dt, segment_duration, overlap)` cuts the time axis into
windows of duration `segment_duration` (rounded to a multiple of `dt`) and
fits each one separately.  All windows share the one-sided grid
``f_k = k/(L\,dt)``, ``k = 0, \dots, L/2 - 1``, with
``L = \mathrm{round}(\text{segment\_duration}/dt)``, which is the same layout
as the regularly sampled Memgram.  The default `slot_width` is half the mean
sampling interval of the *whole* series, so it is the same in every window.
Windows with fewer than two observations give a column of `NaN`.

Each window must contain enough observations to estimate the orders you need:
with ``n`` points per window, ``N_1 \approx n \lambda \Delta\tau``.

## GPU

Both functions accept `use_gpu=true` once `CUDA.jl` is loaded; see
[GPU acceleration](gpu.md).

## Example

See [Irregularly sampled PSD estimate](@ref) in the examples.

[^Bos2002]: R. Bos, S. de Waele, P. M. T. Broersen, "Autoregressive spectral
    estimation by application of the Burg algorithm to irregularly sampled
    data", *IEEE Trans. Instrum. Meas.* **51**, 1289–1294 (2002),
    [doi:10.1109/TIM.2002.808031](https://doi.org/10.1109/TIM.2002.808031).
