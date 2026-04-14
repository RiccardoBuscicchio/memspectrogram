"""
Toy-model MESA spectrogram from a chirping sinusoidal signal.

The script:
1. Generates a sinusoidal chirp whose instantaneous frequency sweeps linearly
   from F_START to F_END over the full duration, with added Gaussian noise.
2. Computes the MESA spectrogram on overlapping short segments.
3. Overlays the theoretical instantaneous frequency as a dashed white line.
4. Saves the spectrogram image to `examples/chirp_spectrogram.png`.

Run from the repository root:

    julia examples/chirp_spectrogram.jl

"""

# Make the src directory visible so we can load Memspectrum without installing it.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "Memspectrum.jl"))
using .Memspectrum

using Random
using Plots

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const FS      = 512.0            # sampling rate [Hz]
const DT      = 1.0 / FS         # sampling interval [s]
const T       = 16.0             # total duration [s]
const N_TOTAL = round(Int, T * FS)  # total samples (8 192)
const SIGMA   = 0.25             # additive noise amplitude
const SEED    = 7

const F_START = 5.0              # starting frequency [Hz]
const F_END   = 120.0            # ending frequency [Hz]

# Spectrogram parameters
const SEG_LEN = 512              # samples per segment (~1 s)
const OVERLAP = 0.875            # 87.5 % overlap → dense time grid

# ---------------------------------------------------------------------------
# 1.  Generate a linear chirp signal with additive noise
#     φ(t) = 2π * (F_START * t + (F_END - F_START) / (2 * T) * t²)
#     x(t) = sin(φ(t)) + σ * noise(t)
# ---------------------------------------------------------------------------

Random.seed!(SEED)

t_vec = collect((0:N_TOTAL-1) .* DT)
phase = @. 2π * (F_START * t_vec + (F_END - F_START) / (2 * T) * t_vec^2)
x     = sin.(phase) .+ SIGMA .* randn(N_TOTAL)

println("Generated linear chirp: N=$N_TOTAL samples, dt=$DT s")
println("  Frequency sweep: $(F_START) → $(F_END) Hz over $(T) s")
println("  Noise amplitude: σ=$SIGMA")

# ---------------------------------------------------------------------------
# 2.  Compute MESA spectrogram
# ---------------------------------------------------------------------------

println("\nComputing spectrogram  (segment_length=$SEG_LEN, overlap=$(round(Int, OVERLAP*100)) %) …")
t_centers, f_grid, psd_mat = mesa_spectrogram(
    x, DT;
    segment_length      = SEG_LEN,
    overlap             = OVERLAP,
    optimisation_method = "FPE",
    method              = "Fast",
    verbose             = true,
)

println("\nSpectrogram: $(size(psd_mat, 2)) segments × $(length(f_grid)) frequency bins")
println("  Time range:  $(round(t_centers[1],  digits=2)) – $(round(t_centers[end], digits=2)) s")
println("  Freq range:  $(round(f_grid[1],     digits=2)) – $(round(f_grid[end],    digits=2)) Hz")

# ---------------------------------------------------------------------------
# 3.  Plot and save
# ---------------------------------------------------------------------------

plt = plot_spectrogram(
    t_centers, f_grid, psd_mat;
    title = "MESA spectrogram – linear chirp  ($(F_START) → $(F_END) Hz)\n" *
            "segment=$(SEG_LEN) samples, $(round(Int, OVERLAP*100)) % overlap, σ_noise=$(SIGMA)",
    size  = (960, 500),
    dpi   = 150,
)

# Overlay the theoretical instantaneous frequency of the chirp.
f_inst = @. F_START + (F_END - F_START) / T * t_centers
plot!(plt, t_centers, f_inst;
      lw    = 2,
      ls    = :dash,
      color = :white,
      label = "Theoretical f(t)",
      legend = :topleft,
)

out_path = joinpath(@__DIR__, "chirp_spectrogram.png")
savefig(plt, out_path)
println("\nSpectrogram saved to $out_path")
