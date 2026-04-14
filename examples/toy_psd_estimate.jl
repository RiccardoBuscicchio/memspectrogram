"""
Toy-model PSD estimate from a synthetic time series.

The script:
1. Generates an AR(2) toy time series with a known analytical spectrum.
2. Estimates the PSD from the time series with the MESA (Burg) method.
3. Compares the MESA estimate against the true analytical AR(2) spectrum.
4. Saves the plot to `examples/toy_psd_estimate.png`.

Run from the repository root:

    julia examples/toy_psd_estimate.jl

"""

# Make the src directory visible so we can load Memspectrum without installing it.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "Memspectrum.jl"))
using .Memspectrum

using Random
using Statistics
using Plots

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const N        = 4_096    # number of time-series samples
const dt       = 1.0 / 256.0   # sampling interval [s]  →  fs = 256 Hz
const AR_COEFF = [-1.5, 0.9]   # AR(2) coefficients  (a1, a2, not including a0=1)
const SIGMA    = 1.0            # white-noise standard deviation
const SEED     = 42

# ---------------------------------------------------------------------------
# 1.  Generate the AR(2) time series
#     x[t] = -a1*x[t-1] - a2*x[t-2] + sigma*noise[t]
#     (sign convention matches Burg's: a_k[1]=1, a_k[2]=a1, a_k[3]=a2)
# ---------------------------------------------------------------------------

Random.seed!(SEED)

x = zeros(N)
x[1] = SIGMA * randn()
x[2] = SIGMA * randn()
for t in 3:N
    x[t] = -AR_COEFF[1] * x[t-1] - AR_COEFF[2] * x[t-2] + SIGMA * randn()
end

println("Generated AR(2) time series: N=$N samples, dt=$dt s")
println("  AR coefficients: a1=$(AR_COEFF[1]), a2=$(AR_COEFF[2])")
println("  Innovation std:  σ=$(SIGMA)")
println("  Sample variance: $(round(var(x), digits=3))")

# ---------------------------------------------------------------------------
# 2.  Estimate PSD with MESA
# ---------------------------------------------------------------------------

m = MESA()
solve!(m, x; method="Fast", optimisation_method="FPE", verbose=false)

println("\nMESA fit complete: AR order p=$(m.p),  P=$(round(m.P, digits=4))")

f_mesa, psd_mesa = spectrum(m, dt; onesided=true)

# ---------------------------------------------------------------------------
# 3.  Analytical one-sided AR(2) PSD
#     S(f) = 2 * dt * σ² / |1 + a1·e^{-2πi·f·dt} + a2·e^{-4πi·f·dt}|²
# ---------------------------------------------------------------------------

function ar2_psd_onesided(f::AbstractVector, a1, a2, sigma, dt)
    psd = similar(f)
    for (k, fk) in enumerate(f)
        z1 = exp(-2π * im * fk * dt)
        z2 = exp(-4π * im * fk * dt)
        den = 1 + a1 * z1 + a2 * z2
        psd[k] = 2 * dt * sigma^2 / abs2(den)   # factor 2 for one-sided
    end
    return psd
end

psd_true = ar2_psd_onesided(f_mesa, AR_COEFF[1], AR_COEFF[2], SIGMA, dt)

# ---------------------------------------------------------------------------
# 4.  Plot
# ---------------------------------------------------------------------------

plt = plot(
    f_mesa, psd_true;
    yscale  = :log10,
    label   = "True AR(2) spectrum",
    lw      = 2,
    ls      = :dash,
    color   = :black,
    xlabel  = "Frequency (Hz)",
    ylabel  = "PSD  (V²/Hz)",
    title   = "MESA PSD estimate vs true AR(2) spectrum\n" *
              "(N=$N, dt=$dt s, p=$(m.p))",
    legend  = :topright,
    size    = (800, 500),
    dpi     = 150,
)

plot!(plt, f_mesa, psd_mesa;
      label  = "MESA estimate (p=$(m.p))",
      lw     = 2,
      color  = :royalblue,
      alpha  = 0.85,
)

out_path = joinpath(@__DIR__, "toy_psd_estimate.png")
savefig(plt, out_path)
println("\nPlot saved to $out_path")

display(plt)
