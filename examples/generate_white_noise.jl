"""
Script to generate LIGO-like noise that matches the O3 design PSD at a fixed
autoregressive order p.

To generate 32 s of data @ 4096 Hz with AR order 300 run:

    julia generate_white_noise.jl --p 300 --t 32 --srate 4096 --savefile fixed_p_noise.dat

This will save the noise in the file fixed_p_noise.dat
"""

using ArgParse
using Downloads
using DelimitedFiles
using Random
using Plots

# Add the parent directory so we can load the Memspectrum package
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Memspectrum

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

function parse_commandline()
    s = ArgParseSettings(description = "Generate LIGO-like noise via MESA AR model")

    @add_arg_table! s begin
        "--p"
            help     = "Autoregressive order of the final timeseries"
            arg_type = Int
            default  = 300
        "--srate"
            help     = "Sample rate of the final timeseries (Hz)"
            arg_type = Float64
            default  = 4096.0
        "--t"
            help     = "Length in seconds of the final timeseries"
            arg_type = Float64
            default  = 32.0
        "--savefile"
            help     = "File to save the timeseries to (optional)"
            arg_type = String
            default  = nothing
    end

    return parse_args(s)
end

args = parse_commandline()
p_order  = args["p"]
srate    = args["srate"]
T        = args["t"]
savefile = args["savefile"]

# ---------------------------------------------------------------------------
# Download LIGO O3 PSD if not already present
# ---------------------------------------------------------------------------

psd_file = "aligo_O3actual_H1.txt"
psd_url  = "https://dcc.ligo.org/public/0165/T2000012/002/aligo_O3actual_H1.txt"
if !isfile(psd_file)
    @info "Downloading $psd_file ..."
    Downloads.download(psd_url, psd_file)
end

data_matrix = readdlm(psd_file, ' ', Float64; skipblanks=true, comments=true)
f_input   = data_matrix[:, 1]
psd_input = data_matrix[:, 2]

# ---------------------------------------------------------------------------
# Generate coloured noise matching the template PSD
# ---------------------------------------------------------------------------

t_grid, time_series, f_grid, frequency_series, interpolated_psd =
    generate_data(f_input, psd_input, T;
                  sampling_rate=srate, seed=0)

# ---------------------------------------------------------------------------
# Fit MESA AR model at fixed order p
# ---------------------------------------------------------------------------

m = MESA()
solve!(m, time_series;
       method="Standard",
       optimisation_method="Fixed",
       m=p_order + 1)

@assert m.p == p_order "Expected AR order $p_order but got $(m.p)"

f_mesa, psd_mesa = spectrum(m, 1.0 / srate; onesided=true)

# ---------------------------------------------------------------------------
# Forecast (simulate) noise from the fitted AR process
# ---------------------------------------------------------------------------

burn_in   = 20_000
total_len = Int(T * srate) + burn_in
seed_data = randn(m.p) .* sqrt(m.P)

fake_data_raw = forecast(m, seed_data, total_len;
                         number_of_simulations=1,
                         P=nothing,
                         include_data=false,
                         seed=nothing,
                         verbose=true)

fake_data = fake_data_raw[1, burn_in+1:end]   # discard burn-in; shape is (1, total_len)

# ---------------------------------------------------------------------------
# Check that generated data reproduces the target PSD
# ---------------------------------------------------------------------------

m_check = MESA()
solve!(m_check, fake_data; method="Standard")
f_check, psd_check = spectrum(m_check, 1.0 / srate; onesided=true)

println("True data AR order:      ", m.p)
println("Generated data AR order: ", m_check.p)
if abs(m_check.p - m.p) > 1
    @warn "The PSD of the generated noise differs noticeably from the target."
end

# ---------------------------------------------------------------------------
# Optional: save generated noise
# ---------------------------------------------------------------------------

if savefile !== nothing
    header = "T = $T | srate = $srate | p = $p_order"
    writedlm(savefile, fake_data, '\n')
    println("Noise saved to $savefile")
end

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

plt = plot(f_input, psd_input;
           xscale=:log10, yscale=:log10,
           ls=:dash, label="True PSD",
           xlabel="f (Hz)", ylabel="PSD (1/Hz)",
           title="Estimated PSD (vs true PSD)")
plot!(plt, f_mesa,  psd_mesa;  label="Estimated PSD")
plot!(plt, f_check, psd_check; label="Generated noise PSD")

display(plt)
