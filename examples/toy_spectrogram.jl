"""
Toy-model MESA spectrogram from a non-stationary synthetic time series.

The script:
1. Generates a time series whose AR(2) peak frequency sweeps linearly with time
   (two concatenated AR(2) blocks with different coefficients).
2. Computes the MESA spectrogram by fitting an AR model on short overlapping
   segments.
3. Saves the spectrogram image to `examples/toy_spectrogram.png`.

Run from the repository root:

    julia examples/toy_spectrogram.jl

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

const FS       = 256.0           # sampling rate [Hz]
const DT       = 1.0 / FS        # sampling interval [s]
const N_TOTAL  = 8_192           # total samples  (~32 s)
const SIGMA    = 1.0             # innovation std
const SEED     = 0

# Two AR(2) blocks, each half of the signal.
# Block 1: resonance near  30 Hz  (a1=-1.5, a2=0.9)
# Block 2: resonance near  80 Hz  (a1=-0.5, a2=0.9)
const COEFF_A  = [-1.5, 0.9]
const COEFF_B  = [-0.5, 0.9]

# Spectrogram parameters
const SEG_LEN  = 512             # samples per segment  (~2 s)
const OVERLAP  = 0.75            # 75 % overlap between consecutive segments

# ---------------------------------------------------------------------------
# 1.  Generate a non-stationary AR(2) time series
# ---------------------------------------------------------------------------

Random.seed!(SEED)

x    = zeros(N_TOTAL)
half = N_TOTAL ÷ 2

x[1] = SIGMA * randn()
x[2] = SIGMA * randn()
for t in 3:N_TOTAL
    c = t <= half ? COEFF_A : COEFF_B
    x[t] = -c[1] * x[t-1] - c[2] * x[t-2] + SIGMA * randn()
end

println("Generated non-stationary AR(2) time series: N=$N_TOTAL samples, dt=$DT s")
println("  Block 1 (t=0..$(half*DT) s):  a1=$(COEFF_A[1]), a2=$(COEFF_A[2])")
println("  Block 2 (t=$(half*DT)..$(N_TOTAL*DT) s): a1=$(COEFF_B[1]), a2=$(COEFF_B[2])")

# ---------------------------------------------------------------------------
# 2.  Compute MESA spectrogram
# ---------------------------------------------------------------------------

println("\nComputing spectrogram  (segment_length=$SEG_LEN, overlap=$(Int(OVERLAP*100)) %) …")
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
    title = "MESA spectrogram  (segment=$(SEG_LEN) samples, $(Int(OVERLAP*100)) % overlap)\n" *
            "AR(2) block 1: a1=$(COEFF_A[1])  →  block 2: a1=$(COEFF_B[1])",
    size  = (960, 500),
    dpi   = 150,
)

# Mark the transition between the two blocks with a vertical dashed line.
vline!(plt, [half * DT];
       lw      = 2,
       ls      = :dash,
       color   = :white,
       label   = "Regime change",
       legend  = :topright,
)

out_path = joinpath(@__DIR__, "toy_spectrogram.png")
savefig(plt, out_path)
println("\nSpectrogram saved to $out_path")
