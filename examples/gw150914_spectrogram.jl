"""
Time–frequency Memgram of GW150914 using public LIGO data.

The script:
1. Downloads a 32-second H1 strain segment centred on GW150914
   (GPS 1126259446–1126259478) from the LOSC tutorial repository on GitHub.
2. Applies a Butterworth bandpass filter (20–400 Hz) to suppress low-frequency
   seismic noise and high-frequency roll-off.
3. Computes the Memgram (MESA spectrogram) on overlapping short segments.
4. Saves the Memgram image to `examples/gw150914_spectrogram.png`.

Run from the repository root:

    julia --project=. examples/gw150914_spectrogram.jl

All parameters can be supplied via command-line flags or a TOML config file:

    julia --project=. examples/gw150914_spectrogram.jl \\
        --config examples/configs/gw150914_spectrogram.toml

    julia --project=. examples/gw150914_spectrogram.jl --seg_len 1024

Data source:
  LIGO Open Science Center (LOSC) tutorial data hosted on GitHub:
  https://github.com/losc-tutorial/LOSC_Event_tutorial
  The full catalogue is available at https://gwosc.org.

References:
  Abbott et al. (2016), PRL 116, 061102  (GW150914 detection paper)
  https://doi.org/10.1103/PhysRevLett.116.061102
"""

# Make the src directory visible so we can load Memspectrum without installing it.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "Memspectrum.jl"))
using .Memspectrum

using ArgParse
using TOML
using Downloads
using HDF5
using DSP
using Plots

# ---------------------------------------------------------------------------
# Argument parsing (command-line + TOML config file)
# ---------------------------------------------------------------------------

function parse_commandline()
    s = ArgParseSettings(description = "Time-frequency Memgram of GW150914")
    @add_arg_table! s begin
        "--config"
            help     = "Path to a TOML configuration file (optional)"
            arg_type = String
            default  = nothing
        "--h1_url"
            help     = "URL of the LOSC H1 HDF5 strain file"
            arg_type = String
            default  = nothing
        "--bp_low"
            help     = "Bandpass lower cut-off frequency (Hz)"
            arg_type = Float64
            default  = nothing
        "--bp_high"
            help     = "Bandpass upper cut-off frequency (Hz)"
            arg_type = Float64
            default  = nothing
        "--bp_order"
            help     = "Butterworth filter order"
            arg_type = Int
            default  = nothing
        "--seg_len"
            help     = "Samples per spectrogram segment"
            arg_type = Int
            default  = nothing
        "--overlap"
            help     = "Fractional overlap between segments (0–1)"
            arg_type = Float64
            default  = nothing
        "--f_min"
            help     = "Minimum frequency shown in spectrogram (Hz)"
            arg_type = Float64
            default  = nothing
        "--f_max"
            help     = "Maximum frequency shown in spectrogram (Hz)"
            arg_type = Float64
            default  = nothing
        "--optimisation_method"
            help     = "AR order-selection criterion"
            arg_type = String
            default  = nothing
        "--method"
            help     = "Burg algorithm variant (Fast or Standard)"
            arg_type = String
            default  = nothing
    end
    return parse_args(s)
end

args = parse_commandline()
cfg  = Dict{String,Any}()
if args["config"] !== nothing
    cfg = TOML.parsefile(args["config"])
end

_get(key, default) = args[key] !== nothing ? args[key] :
                     haskey(cfg, key)       ? cfg[key]  : default

# LOSC tutorial data hosted on GitHub (H1 strain, 4096 Hz, 32 s around GW150914)
const H1_URL = _get("h1_url",
    "https://raw.githubusercontent.com/losc-tutorial/LOSC_Event_tutorial/" *
    "master/H-H1_LOSC_4_V2-1126259446-32.hdf5")
const BP_LOW  = _get("bp_low",   20.0)
const BP_HIGH = _get("bp_high",  400.0)
const BP_ORDER = _get("bp_order", 4)
const SEG_LEN  = _get("seg_len",  512)
const OVERLAP  = _get("overlap",  0.95)
const F_MIN    = _get("f_min",    20.0)
const F_MAX    = _get("f_max",    400.0)
const OPT_METHOD = _get("optimisation_method", "FPE")
const METHOD     = _get("method",              "Standard")

# GW150914 merger GPS time (used to mark the event on the plot)
const GPS_MERGE = 1126259462.4

# ---------------------------------------------------------------------------
# 1.  Download LOSC H1 strain data
# ---------------------------------------------------------------------------

h1_path = joinpath(tempdir(), "H-H1_LOSC_4_V2-1126259446-32.hdf5")

if !isfile(h1_path)
    println("Downloading H1 strain data from LOSC tutorial repository …")
    Downloads.download(H1_URL, h1_path)
    println("  Saved to $h1_path  ($(round(filesize(h1_path)/1024, digits=1)) kB)")
else
    println("Using cached H1 data at $h1_path")
end

# ---------------------------------------------------------------------------
# 2.  Read strain and metadata from HDF5
# ---------------------------------------------------------------------------

strain, fs, gps_start = HDF5.h5open(h1_path, "r") do fid
    s   = Float64.(read(fid["strain/Strain"]))
    dur = Float64(read(fid["meta/Duration"]))
    gps = Float64(read(fid["meta/GPSstart"]))
    sr  = length(s) / dur
    s, sr, gps
end

dt = 1.0 / fs
N  = length(strain)
println("\nLoaded H1 strain:  $N samples,  fs=$(fs) Hz,  duration=$(N*dt) s")
println("  GPS start:  $gps_start  →  merger at t=$(GPS_MERGE - gps_start) s into segment")

# ---------------------------------------------------------------------------
# 3.  Bandpass filter (Butterworth, zero-phase via filtfilt)
# ---------------------------------------------------------------------------

println("\nBandpass filtering: $(BP_LOW)–$(BP_HIGH) Hz  (order $(BP_ORDER)) …")
# DSP.jl requires normalised cut-offs in (0, 1) where 1 = Nyquist = fs/2
f_ny  = fs / 2.0
resp  = Bandpass(BP_LOW / f_ny, BP_HIGH / f_ny)
filt_ = digitalfilter(resp, Butterworth(BP_ORDER))
strain_bp = filtfilt(filt_, strain)

# ---------------------------------------------------------------------------
# 4.  Compute MESA spectrogram (Memgram)
# ---------------------------------------------------------------------------

println("\nComputing Memgram  (segment_length=$SEG_LEN samples, overlap=$(round(Int, OVERLAP*100)) %) …")
t_centers_gps, f_grid, psd_mat = memgram(
    strain_bp, dt;
    segment_length      = SEG_LEN,
    overlap             = OVERLAP,
    optimisation_method = OPT_METHOD,
    method              = METHOD,
    verbose             = true,
)

# memgram returns times measured from the start of the input array (seconds).
# Convert to time relative to the GW150914 merger by adding the GPS start offset.
t_centers_rel = (gps_start .+ t_centers_gps) .- GPS_MERGE

# Restrict frequency axis to [F_MIN, F_MAX] for display
freq_mask = (f_grid .>= F_MIN) .& (f_grid .<= F_MAX)
f_show    = f_grid[freq_mask]
psd_show  = psd_mat[freq_mask, :]

println("\nMemgram: $(size(psd_show, 2)) segments × $(length(f_show)) freq bins")
println("  Time range:  $(round(t_centers_rel[1],  digits=2)) – " *
        "$(round(t_centers_rel[end], digits=2)) s  (relative to merger)")
println("  Freq range:  $(round(f_show[1],   digits=1)) – " *
        "$(round(f_show[end], digits=1)) Hz")

# ---------------------------------------------------------------------------
# 5.  Plot and save
# ---------------------------------------------------------------------------

plt = plot_spectrogram(
    t_centers_rel, f_show, psd_show;
    title = "GW150914 — H1 Memgram  ($(BP_LOW)–$(BP_HIGH) Hz bandpass)\n" *
            "segment=$(SEG_LEN) samples, $(round(Int, OVERLAP*100)) % overlap",
    size  = (960, 500),
    dpi   = 150,
)

xlabel!(plt, "Time relative to merger (s)")
ylabel!(plt, "Frequency (Hz)")

# Mark the merger time with a vertical dashed line
vline!(plt, [0.0];
       lw      = 2,
       ls      = :dash,
       color   = :white,
       label   = "GW150914 merger",
       legend  = :topright,
)

out_path = joinpath(@__DIR__, "gw150914_spectrogram.png")
savefig(plt, out_path)
println("\nMemgram saved to $out_path")
