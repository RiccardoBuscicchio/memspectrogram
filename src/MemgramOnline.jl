"""
    MemgramOnline

Sub-module of Memspectrum providing an asynchronous, `Channel`-based online
spectrogram processor — the Julia analogue of Python's `asyncio` event-driven
pattern.

New data chunks are pushed into a `Channel{Vector{Float64}}`; a background
`Task` (Julia coroutine) drains the channel, maintains a sliding-window
buffer, and calls a user-supplied callback every time a new segment is ready.
The calling code is never blocked: `push_chunk!` returns immediately once the
chunk is enqueued.

## Typical usage

```julia
# Run from the repository root so that relative paths resolve correctly.
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
include(joinpath(@__DIR__, "src", "Memspectrum.jl"))
using .Memspectrum
using .Memspectrum.MemgramOnline

results = Dict(:t => Float64[], :psd => Vector{Float64}[], :f => Float64[])

proc = start_processor(;
    segment_length      = 512,
    overlap             = 0.75,
    dt                  = 1.0 / 512.0,
    optimisation_method = "FPE",
    method              = "Fast",
    on_update = (t_c, f_g, psd_c) -> begin
        push!(results[:t],   t_c)
        push!(results[:psd], copy(psd_c))
        isempty(results[:f]) && append!(results[:f], f_g)
    end,
)

for chunk in data_chunks
    push_chunk!(proc, chunk)
end

close_processor!(proc)   # wait for all pending segments to finish
```
"""
module MemgramOnline

import ..MESA, ..solve!, ..spectrum

export OnlineMemgramProcessor, start_processor, push_chunk!, close_processor!

# ---------------------------------------------------------------------------
# Public struct
# ---------------------------------------------------------------------------

"""
    OnlineMemgramProcessor

Holds the state for an active online Memgram computation.  All fields are
internal; use [`push_chunk!`](@ref) and [`close_processor!`](@ref) to
interact with the processor.

Fields:
- `channel`        : `Channel{Vector{Float64}}` – data ingestion queue
- `task`           : background `Task` that drains the channel
- `segment_length` : number of samples per AR fit
- `overlap`        : fractional overlap between consecutive segments
- `dt`             : sampling interval (seconds)
- `f_grid`         : one-sided frequency grid shared with every callback
"""
struct OnlineMemgramProcessor
    channel        :: Channel{Vector{Float64}}
    task           :: Task
    segment_length :: Int
    overlap        :: Float64
    dt             :: Float64
    f_grid         :: Vector{Float64}
end

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

"""
    start_processor(; segment_length, overlap=0.5, dt=1.0,
                      optimisation_method="FPE", method="Fast",
                      channel_size=64, on_update) -> OnlineMemgramProcessor

Create and immediately start an asynchronous Memgram processor.

A background `Task` is spawned that:
1. Waits for data chunks to appear on an internal `Channel`.
2. Appends them to a sliding-window buffer.
3. Whenever the buffer holds at least `segment_length` samples, fits an
   AR model via Burg's algorithm on the oldest `segment_length` samples,
   computes the one-sided PSD, and calls `on_update`.
4. Advances the buffer by one *stride* (`= round(segment_length * (1 - overlap))`
   samples) and repeats from step 3.
5. Exits cleanly when the channel is closed.

# Keyword arguments
- `segment_length`      : samples per AR segment (**required**)
- `overlap`             : fractional overlap in [0, 1) (default `0.5`)
- `dt`                  : sampling interval in seconds (default `1.0`)
- `optimisation_method` : AR order-selection criterion (default `"FPE"`)
- `method`              : Burg variant – `"Fast"` or `"Standard"` (default `"Fast"`)
- `channel_size`        : internal channel buffer depth (default `64`)
- `on_update`           : **required** callback with signature
                          `(t_center::Float64, f_grid::Vector{Float64},
                            psd_col::Vector{Float64}) -> nothing`

# Returns
An [`OnlineMemgramProcessor`](@ref).  Feed it with [`push_chunk!`](@ref) and
call [`close_processor!`](@ref) when the stream ends.
"""
function start_processor(;
        segment_length      :: Int,
        overlap             :: Float64 = 0.5,
        dt                  :: Float64 = 1.0,
        optimisation_method :: String  = "FPE",
        method              :: String  = "Fast",
        channel_size        :: Int     = 64,
        on_update)

    0.0 <= overlap < 1.0 ||
        error("overlap must be in [0, 1).")
    segment_length >= 4 ||
        error("segment_length must be at least 4.")

    stride = max(1, round(Int, segment_length * (1.0 - overlap)))

    # One-sided frequency grid – identical to what mesa_spectrogram uses.
    n_freq = segment_length ÷ 2
    f_grid = collect(0:n_freq-1) ./ (segment_length * dt)

    ch = Channel{Vector{Float64}}(channel_size)

    # -----------------------------------------------------------------------
    # Background Task – Julia coroutine, analogous to an asyncio consumer
    # -----------------------------------------------------------------------
    tsk = @async begin
        buffer        = Float64[]          # sliding window
        global_offset = 0                  # 0-based sample index of buffer[1]

        try
            for chunk in ch                    # blocks until a chunk arrives
                append!(buffer, chunk)

                # Process every full segment that fits in the current buffer.
                while length(buffer) >= segment_length
                    seg      = buffer[1:segment_length]
                    t_center = (global_offset + 0.5 * segment_length) * dt

                    m = MESA()
                    solve!(m, seg;
                           method              = method,
                           optimisation_method = optimisation_method,
                           verbose             = false)
                    _, psd_col = spectrum(m, dt; onesided = true)

                    on_update(t_center, f_grid, psd_col)

                    # Advance the window by one stride.
                    advance = min(stride, length(buffer))
                    deleteat!(buffer, 1:advance)
                    global_offset += advance
                end
            end
        catch e
            @error "MemgramOnline background task error" exception=(e, catch_backtrace())
        end
    end

    return OnlineMemgramProcessor(ch, tsk, segment_length, overlap, dt, f_grid)
end

"""
    push_chunk!(proc::OnlineMemgramProcessor, chunk::AbstractVector)

Enqueue a new data chunk for asynchronous processing.

The call returns immediately once the chunk is placed on the internal
`Channel`.  The background task will process it (possibly together with
previously buffered samples) as soon as it is scheduled.

# Arguments
- `proc`  : the processor returned by [`start_processor`](@ref)
- `chunk` : a `Vector` (or any `AbstractVector`) of new samples
"""
function push_chunk!(proc::OnlineMemgramProcessor, chunk::AbstractVector)
    put!(proc.channel, Float64.(vec(chunk)))
    return nothing
end

"""
    close_processor!(proc::OnlineMemgramProcessor)

Signal end-of-stream, then block until all pending segments have been
processed and the background task has exited cleanly.

After this call `proc` must not be used.
"""
function close_processor!(proc::OnlineMemgramProcessor)
    close(proc.channel)
    wait(proc.task)
    return nothing
end

end # module MemgramOnline
