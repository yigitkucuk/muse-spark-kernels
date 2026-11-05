def _make_scratch(dev):
    """Per-call scratch, sized for CHUNK_TOKENS tokens. Layout is fixed by
    the CUDA launcher (okc_run), which indexes scratch[0]/[1]/[2] as:
    [0] int32: 8 counters + 1024 doneA slots + the second ss copy
    (ss1, EMAX floats viewed as int32 -- the launcher takes its address
    as ib + 8 + 1024);
    [1] fp32: ss (EMAX) + ybuf (EMAX x HIDDEN) + WGS stream-K fixups;
    [2] bf16: inter (EMAX x HIDDEN) + WGS xA / xB (2 x WGS_XROWS x HIDDEN).
    Zeroed here so the first call starts from a clean grid barrier state;
    the kernel re-zeroes what it needs between launches.
    """
    return [torch.zeros(8 + 1024 + EMAX + 4096, dtype=torch.int32, device=dev),
            torch.zeros(EMAX + EMAX * HIDDEN + WGS_FIXF, dtype=torch.float32, device=dev),
            _inter_sentinel(torch.zeros((EMAX + 2 * WGS_XROWS) * HIDDEN, dtype=torch.bfloat16, device=dev))]


# ------------------------------------------------------------------ timed path
def kernel_fn(data, state):
    """Run one MoE layer. Fresh output every call; no host syncs; the only
    per-call work is picking the compiled variant by token count and
    launching it in CHUNK_TOKENS-token slices."""
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    assert cfg["local_start"] == state["lo"] and cfg["local_count"] == state["cnt"]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    if ids.data_ptr() & 15:  # the WGS routing loads topk_ids 16 B at a time (an offset view gets an aligned copy)
        ids = ids.clone()
    tw = data["topk_weights"].to(torch.float32).contiguous()
    out = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=x.device)
    sfx = state.get("sfx", ())
    sfxm, sfxs = state.get("sfxm", sfx), state.get("sfxs", sfx)
    if TINY_DEFINES and M <= TINY_MAX_M:
        ext = _ext(tuple(TINY_DEFINES) + sfxs)
    elif TINY2_DEFINES and M <= TINY2_MAX_M and (sfx or "-DOKC_WGS=1" not in tuple(TINY2_DEFINES)):
        ext = _ext(tuple(TINY2_DEFINES) + sfxs)  # (the WGS build needs the PRMT scale decode: else MID/MAIN)
    elif MID_DEFINES and M <= MID_MAX_M:
        ext = _ext(tuple(MID_DEFINES) + sfxs)
    else:
        ext = _ext(tuple(_main_defines(sfx)) + sfxm)
    for m0 in range(0, M, CHUNK_TOKENS):
        m1 = min(M, m0 + CHUNK_TOKENS)
        ext.okc_run(state["w"], state["scratch"], x[m0:m1], ids[m0:m1], tw[m0:m1], out[m0:m1],
                    state["lo"], state["cnt"], state["grid"])
    return out


