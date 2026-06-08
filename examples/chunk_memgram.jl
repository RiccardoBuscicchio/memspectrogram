"""
Memgram from MoLi pickle data.

Loads the requested pickle file and extracts:
- chunk_dict["samples"]
- chunk_dict["timestamps"]

Then computes a memgram with 10^4-second segments and 95% overlap, and
renders it with a magma colormap. It also computes a standard FFT spectrogram
for side-by-side comparison.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "Memspectrum.jl"))
using .Memspectrum

using FFTW
using Plots
using Statistics
using JSON

#const PICKLE_PATH = "../data/MoLi/mojito_source2_chunk_dict_-1days_to_tc.pickle"
const SEGMENT_SECONDS = 5000.0
const OVERLAP = 0.60
const ANIMATION_COLUMN_STEP = 2
const ANIMATION_FPS = 24

function _get_first(d, keys::Vector{String})
    for k in keys
        if haskey(d, k)
            return d[k]
        end
        ks = Symbol(k)
        if haskey(d, ks)
            return d[ks]
        end
    end
    error("None of keys $(keys) found in pickle dictionary")
end

function load_chunk_timeseries_from_pickle(path::AbstractString)
    if !isfile(path)
        error("Pickle file not found: $path")
    end

    chunk_dict = Pickle.load(path, proto=5)
    # Print chunk_dict keys for debugging
    println("Loaded pickle with keys: ", keys(chunk_dict))
    timestamps = vec(Float64.(_get_first(chunk_dict, ["t"])))
    samples = vec(Float64.(_get_first(chunk_dict, ["tdi_E"])))

    if length(timestamps) != length(samples)
        error("timestamps and samples have different lengths")
    end

    return timestamps, samples
end

function fft_spectrogram(
    samples::AbstractVector{<:Real},
    dt::Real;
    segment_length::Int,
    overlap::Float64,
)
    x = Float64.(samples)
    n = length(x)
    if n < segment_length
        error("Signal too short for segment_length=$segment_length")
    end

    hop = max(1, round(Int, segment_length * (1 - overlap)))
    n_seg = 1 + fld(n - segment_length, hop)
    n_freq = segment_length ÷ 2 + 1

    # Hann window for lower spectral leakage.
    window = 0.5 .- 0.5 .* cos.(2π .* (0:segment_length-1) ./ (segment_length - 1))
    window_power = sum(abs2, window)

    t_centers = Vector{Float64}(undef, n_seg)
    f_grid = collect(0:n_freq-1) ./ (segment_length * dt)
    psd = Matrix{Float64}(undef, n_freq, n_seg)

    for j in 1:n_seg
        i0 = 1 + (j - 1) * hop
        i1 = i0 + segment_length - 1
        seg = @view x[i0:i1]
        segw = seg .* window
        spec = rfft(segw)
        psd[:, j] = abs2.(spec) ./ (window_power / dt)
        t_centers[j] = ((i0 - 1) + 0.5 * (segment_length - 1)) * dt
    end

    return t_centers, f_grid, psd
end

function _interp_column_linear(x_src::AbstractVector{<:Real},
                               y_src::AbstractVector{<:Real},
                               x_dst::AbstractVector{<:Real})
    y_dst = similar(Float64.(x_dst))
    @inbounds for i in eachindex(x_dst)
        x = x_dst[i]
        if x <= x_src[1]
            y_dst[i] = y_src[1]
        elseif x >= x_src[end]
            y_dst[i] = y_src[end]
        else
            j = searchsortedlast(x_src, x)
            x0, x1 = x_src[j], x_src[j + 1]
            y0, y1 = y_src[j], y_src[j + 1]
            w = (x - x0) / (x1 - x0)
            y_dst[i] = (1 - w) * y0 + w * y1
        end
    end
    return y_dst
end

function memgram_logspace(
    samples::AbstractVector{<:Real},
    dt::Float64;
    segment_length::Int,
    overlap::Float64,
    frequencies::AbstractVector{<:Real},
    optimisation_method::String="FPE",
    method::String="Standard",
    verbose::Bool=true,
)
    0.0 <= overlap < 1.0 || error("overlap must be in [0, 1).")
    x = Float64.(samples)
    stride = max(1, round(Int, segment_length * (1.0 - overlap)))
    starts = collect(1:stride:length(x) - segment_length + 1)
    n_seg = length(starts)
    n_seg >= 1 || error("Time series too short for the requested segment_length.")

    f_grid = Float64.(frequencies)
    psd_matrix = Matrix{Float64}(undef, length(f_grid), n_seg)
    t_centers = Vector{Float64}(undef, n_seg)

    for (j, s) in enumerate(starts)
        seg = x[s:s + segment_length - 1]
        t_centers[j] = (s - 1 + 0.5 * segment_length) * dt

        mj = MESA()
        solve!(mj, seg; method=method, optimisation_method=optimisation_method, verbose=false)
        psd_matrix[:, j] = spectrum(mj, dt; frequencies=f_grid)

        if verbose
            print("\r  Segment $j / $n_seg")
        end
    end
    verbose && println()

    return t_centers, f_grid, psd_matrix
end


data = JSON.parsefile("../data/chunk_tdi_E.json")
timestamps = data["t"]
samples = data["tdi_E"]

# Turn the two lists into vectors of Float64, just in case they were read as arrays of Any.
timestamps = vec(Float64.(timestamps))
samples = vec(Float64.(samples))

#timestamps, samples = load_chunk_timeseries_from_pickle(PICKLE_PATH)
dt = 2.5
segment_length = round(Int, SEGMENT_SECONDS / dt)
f_min = 1.0 / (segment_length * dt)
f_max = 0.5 / dt
f_grid_log = exp10.(range(log10(f_min), log10(f_max), length=segment_length ÷ 2))

#println("Loaded $(length(samples)) samples from ")
println("Median dt: $dt s")
println("Segment length: $segment_length samples (~$(SEGMENT_SECONDS) s)")
println("Overlap: $(round(Int, OVERLAP * 100))%")

t_centers, f_grid, psd_mat = memgram_logspace(
    samples,
    dt;
    segment_length = segment_length,
    overlap = OVERLAP,
    frequencies = f_grid_log,
    optimisation_method = "FPE",
    method = "Standard",
    verbose = true,
)

time_days = (t_centers .+ timestamps[1]) ./ 86400.0
log_psd = log10.(max.(psd_mat, 1e-300))

t_fft, f_fft, psd_fft = fft_spectrogram(
    samples,
    dt;
    segment_length = segment_length,
    overlap = OVERLAP,
)

time_days_fft = (t_fft .+ timestamps[1]) ./ 86400.0
psd_fft_loggrid = Matrix{Float64}(undef, length(f_grid_log), size(psd_fft, 2))
for j in axes(psd_fft, 2)
    psd_fft_loggrid[:, j] = _interp_column_linear(f_fft, psd_fft[:, j], f_grid_log)
end
log_psd_fft = log10.(max.(psd_fft_loggrid, 1e-300))

# Use a shared color scale so both plots are directly comparable.
global_cmin = min(minimum(log_psd), minimum(log_psd_fft))
global_cmax = max(maximum(log_psd), maximum(log_psd_fft))
shared_clims = (global_cmin, global_cmax)

plt = heatmap(
    time_days,
    f_grid,
    log_psd;
    c = :magma,
    clims = shared_clims,
    xlabel = "",
    ylabel = "",
    xticks = false,
    yticks = false,
    yscale = :log10,
    title = "",
    colorbar_title = "",
    size = (1200, 720),
    dpi = 150,
)

out_path = joinpath(@__DIR__, "chunk_memgram_moli_minus1day.png")
savefig(plt, out_path)
println("Memgram saved to $out_path")

# Create a synthetic 10-hour data gap in the middle of the memgram and render
# it as a white strip by masking those columns to NaN.
gap_hours = 10.0
gap_days = gap_hours / 24.0
mid_day = 0.5 * (minimum(time_days) + maximum(time_days))
gap_mask = abs.(time_days .- mid_day) .<= (0.5 * gap_days)

log_psd_with_gap = copy(log_psd)
log_psd_with_gap[:, gap_mask] .= NaN

plt_gap = heatmap(
    time_days,
    f_grid,
    log_psd_with_gap;
    c = :magma,
    clims = shared_clims,
    xlabel = "",
    ylabel = "",
    xticks = false,
    yticks = false,
    yscale = :log10,
    title = "",
    colorbar_title = "",
    background_color = :white,
    background_color_inside = :white,
    size = (1200, 720),
    dpi = 150,
)

out_path_gap = joinpath(@__DIR__, "chunk_memgram_moli_minus1day_gap_10h.png")
savefig(plt_gap, out_path_gap)
println("Memgram with 10-hour gap saved to $out_path_gap")

plt_zoom = heatmap(
    time_days,
    f_grid,
    log_psd;
    c = :magma,
    clims = shared_clims,
    xlabel = "",
    ylabel = "",
    xticks = false,
    yticks = false,
    yscale = :log10,
    xlims = (109, 111),
    title = "",
    colorbar_title = "",
    size = (1200, 720),
    dpi = 150,
)

out_path_zoom = joinpath(@__DIR__, "chunk_memgram_moli_minus1day_zoom_109_111.png")
savefig(plt_zoom, out_path_zoom)
println("Memgram zoom saved to $out_path_zoom")

plt_fft = heatmap(
    time_days_fft,
    f_grid_log,
    log_psd_fft;
    c = :magma,
    clims = shared_clims,
    xlabel = "",
    ylabel = "",
    xticks = false,
    yticks = false,
    yscale = :log10,
    title = "",
    colorbar_title = "",
    size = (1200, 720),
    dpi = 150,
)

out_path_fft = joinpath(@__DIR__, "chunk_fft_spectrogram_moli_minus1day.png")
savefig(plt_fft, out_path_fft)
println("FFT spectrogram saved to $out_path_fft")

plt_fft_zoom = heatmap(
    time_days_fft,
    f_grid_log,
    log_psd_fft;
    c = :magma,
    clims = shared_clims,
    xlabel = "",
    ylabel = "",
    xticks = false,
    yticks = false,
    yscale = :log10,
    xlims = (109, 111),
    title = "",
    colorbar_title = "",
    size = (1200, 720),
    dpi = 150,
)

out_path_fft_zoom = joinpath(@__DIR__, "chunk_fft_spectrogram_moli_minus1day_zoom_109_111.png")
savefig(plt_fft_zoom, out_path_fft_zoom)
println("FFT spectrogram zoom saved to $out_path_fft_zoom")

function animate_memgram_reveal(
    time_days::AbstractVector{<:Real},
    f_grid::AbstractVector{<:Real},
    log_psd::AbstractMatrix{<:Real},
    clims::Tuple{<:Real,<:Real};
    out_dir::AbstractString,
    gif_path::AbstractString,
    column_step::Int=1,
    fps::Int=20,
)
    n_cols = size(log_psd, 2)
    mkpath(out_dir)

    anim = Animation()
    frame_idx = 0

    for col in 1:column_step:n_cols
        frame_idx += 1

        reveal = fill(NaN, size(log_psd))
        reveal[:, 1:col] = log_psd[:, 1:col]

        p = heatmap(
            time_days,
            f_grid,
            reveal;
            c = :magma,
            clims = clims,
            xlabel = "",
            ylabel = "",
            xticks = false,
            yticks = false,
            yscale = :log10,
            title = "",
            colorbar_title = "",
            size = (1200, 720),
            dpi = 150,
        )

        frame_path = joinpath(out_dir, "frame_" * lpad(string(frame_idx), 4, '0') * ".png")
        savefig(p, frame_path)
        frame(anim, p)
    end

    gif(anim, gif_path; fps=fps)
    return frame_idx
end

frames_dir = joinpath(@__DIR__, "chunk_memgram_frames")
gif_out_path = joinpath(@__DIR__, "chunk_memgram_reveal.gif")
n_frames = animate_memgram_reveal(
    time_days,
    f_grid,
    log_psd,
    shared_clims;
    out_dir = frames_dir,
    gif_path = gif_out_path,
    column_step = ANIMATION_COLUMN_STEP,
    fps = ANIMATION_FPS,
)

println("Memgram reveal animation saved: $gif_out_path")
println("Saved $n_frames frame PNGs to: $frames_dir")
