# ---------------------------------------------------------------------- setup
def prepare(weights):
    import time as _time
    _t0 = _time.time()
    """Repack weights (bit/byte permutations only, never requantized) into
    the layouts the kernel reads, allocate scratch, and compile the
    variants kernel_fn may pick. Called once; not on the timed path."""
    dev = weights["w13"].device
    lo, cnt = LOCAL_START, LOCAL_EXPERTS
    I = weights["w13"].shape[1] // 2   # gate+up rows: w13 stacks [gate; up]
    H = weights["w2"].shape[1]         # down rows
    assert I == HIDDEN and H == HIDDEN
    assert weights["post_norm_w"].dtype == torch.bfloat16  # kept as given (TASK.md: BF16 weights stay BF16)
    assert weights["w13_gs"].dtype == torch.float32 and weights["w2_gs"].dtype == torch.float32  # stored as given
    # rowsA: gate rows and up rows interleaved in 8-row groups (the mma
    # row-group size); rowsB: down rows in order.
    rb = torch.arange(I // 8, device=dev)
    g = torch.arange(8, device=dev)
    rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    rowsB = torch.arange(H, device=dev)
    prmt = PRMT_SCALES and _scales_prmt_exact(weights["w13_sf"][lo:lo + cnt], weights["w2_sf"][lo:lo + cnt])
    sc6 = bool(prmt) and _sc6_ok(weights["w13_sf"][lo:lo + cnt], weights["w2_sf"][lo:lo + cnt])
    sc6w, sc6s = sc6 and SC6_WGS and _wgs_on(), sc6 and SC6_SKINNY
    wA, sA, wB, sB = [], [], [], []
    for e in range(lo, lo + cnt):
        wA.append(_pack_codes(weights["w13"][e], rowsA))
        sA.append(_pack_scales(weights["w13_sf"][e], rowsA))
        wB.append(_pack_codes(weights["w2"][e], rowsB))
        sB.append(_pack_scales(weights["w2_sf"][e], rowsB))
    # LUT: fp8 e4m3 byte -> bf16x2(value * 2^64), packed into one int32.
    lut = (torch.arange(256, dtype=torch.int32, device=dev).to(torch.uint8).view(torch.float8_e4m3fn).float()
           * 2.0**64).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    lut = (lut | (lut << 16)).to(torch.int32)
    state = {
        # w: [packed gate/up codes, scales, group scales,
        #     packed down codes, scales, group scales,
        #     e4m3 LUT, post_norm_w (bf16 as given; kernel forms 1 + w in FP32),
        #     fat-order gate/up, fat-order down,
        #     unit-order gate/up, unit-order down,
        #     (+ WGS copies appended below, w[12] / w[13], when any variant uses WGS)]
        "w": [torch.cat(wA), torch.cat(sA), weights["w13_gs"][lo:lo + cnt].contiguous(),
              torch.cat(wB), torch.cat(sB), weights["w2_gs"][lo:lo + cnt].contiguous(),
              lut.contiguous(), weights["post_norm_w"].contiguous(),
              torch.cat([_fat_order(a, b) for a, b in zip(wA, sA)]), torch.cat([_fat_order(a, b) for a, b in zip(wB, sB)]),
              torch.cat([_unit_order(a, b, sc6s) for a, b in zip(wA, sA)]), torch.cat([_unit_order(a, b, sc6s) for a, b in zip(wB, sB)])],
        "lo": lo, "cnt": cnt, "scratch": _make_scratch(dev),
        "grid": torch.cuda.get_device_properties(dev).multi_processor_count,
    }
    if _wgs_on():  # w[12], w[13]: the WGS copies (one more copy of the weights, laid out per 64-row unit)
        state["w"].append(torch.cat([_wgs_copy(weights["w13"][e], weights["w13_sf"][e], _kperm_rows(rowsA), sc6w) for e in range(lo, lo + cnt)]))
        state["w"].append(torch.cat([_wgs_copy(weights["w2"][e], weights["w2_sf"][e], rowsB, sc6w) for e in range(lo, lo + cnt)]))
    if MFAB and locals().get("sc6w", False):   # the plain-scale WGS copy for the twin without SC6
        state["w_ns"] = state["w"][:12] + [torch.cat([_wgs_copy(weights["w13"][e], weights["w13_sf"][e], _kperm_rows(rowsA), False) for e in range(lo, lo + cnt)]),
                                          torch.cat([_wgs_copy(weights["w2"][e], weights["w2_sf"][e], rowsB, False) for e in range(lo, lo + cnt)])]
    state["sfx"] = ("-DOKC_PRMTSC=1",) if prmt else ()
    state["sfxm"] = state["sfx"] + (("-DOKC_SC6W=1",) if sc6w else ())
    state["sfxs"] = state["sfx"] + (("-DOKC_SC6S=1",) if sc6s else ())
    mods = [tuple(_main_defines(state["sfx"])) + state["sfxm"]]
    for d in (TINY_DEFINES, TINY2_DEFINES, MID_DEFINES):
        if d and (state["sfx"] or "-DOKC_WGS=1" not in tuple(d)):  # the WGS build needs the PRMT scale decode
            mods.append(tuple(d) + state["sfxs"])
    import concurrent.futures
    _todo = []
    for d in mods + (list(_mfab_defs(state).values()) + list(_mfab_defs(state, True).values()) + ([tuple(TINY_DEFINES) + ("-DOKC_L2H=0",) + tuple(state.get("sfxs", state["sfx"]))] if MFAB_L2H else []) if MFAB else []):
        if tuple(d) not in _todo:
            _todo.append(tuple(d))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as _pool:
        list(_pool.map(_ext, _todo))
    torch.cuda.synchronize()
    print(f"[mfab] modules built at {_time.time() - _t0:.0f}s", flush=True)
    for d in mods[:1] + (list(_mfab_defs(state).values()) + list(_mfab_defs(state, True).values()) if MFAB else []):
        print("[mfab] regs/local/smem (<false>, <true>)", [x for x in d if x.startswith(("-DOKC_WGS", "-DOKC_MFIN", "-DOKC_B4L", "-DOKC_MFP", "-DOKC_MFQ"))], _ext(d).okc_attrs(), flush=True)
    if SC6_ATTRS:
        print(f"[sc6] scales ok={sc6} wgs={sc6w} skinny={sc6s}", flush=True)
        for d in mods:
            print("[sc6] regs/local/smem (moe_fused<false>, <true>)", [x for x in d if x.startswith(('-DOKC_WGS', '-DOKC_NW', '-DOKC_TINY', '-DOKC_SPLIT', '-DOKC_SC6'))], _ext(d).okc_attrs(), flush=True)
    if DIAG:
        _diag(state, weights)
    if MFAB:
        _mfab(state, weights, _t0)
    return state


MFAB = False     # prepare() prints a paired timing of the MAIN-build variants (graph replays, cold L2, interleaved)
MFAB_VARIANTS = {"base": (("-DOKC_MFIN=0", "-DOKC_B4L=0"), True), "mf": (("-DOKC_MFIN=1", "-DOKC_B4L=0"), True), "prod": ((), False)}
# (extra defines, True: the twin without SC6: -DOKC_SC6W off and the plain WGS weight copy state["w_ns"])
MFATTR = {}   # extra builds compiled for their registers / local bytes only (never run)
MFAB_SHAPES = (32, 128, 256)
MFAB_L2H = False
MFAB_SEEDS = 6
MFAB_REPS = 24
MFAB_BUDGET_S = 200.0   # skip the extra builds when prepare() is already this old


def _mfab_defs(state, attr=False):
    main = tuple(_main_defines(state["sfx"]))
    out = {}
    for n, (extra, ns) in MFAB_VARIANTS.items():
        sfx = tuple(state["sfx"]) if ns else tuple(state.get("sfxm", state["sfx"]))
        out[n] = tuple(d for d in main if not d.startswith(("-DOKC_MFIN", "-DOKC_B4L", "-DOKC_MFS"))) + extra + sfx
    return out


def _mfab(state, weights, t0):
    import time
    dev = weights["w13"].device
    exts = {}
    defs_of = _mfab_defs(state)
    if time.time() - t0 > MFAB_BUDGET_S:
        print(f"[mfab] skipped: prepare is {time.time() - t0:.0f}s old", flush=True)
        return
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = {n: pool.submit(_ext, d) for n, d in defs_of.items()}
        for n, f in futs.items():
            try:
                exts[n] = f.result()
                print(f"[mfab] built {n} {MFAB_VARIANTS[n][0]} regs/local/smem (<false>, <true>) {exts[n].okc_attrs()} at {time.time() - t0:.0f}s", flush=True)
            except Exception as ex:  # diagnostics must never break the run
                print(f"[mfab] build {n} failed: {str(ex)[-1500:]}", flush=True)
    names = list(exts)
    if "base" not in exts:
        return
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
    for bs in MFAB_SHAPES:
        _mfab_time(bs, {n: (exts[n], (state.get("w_ns", state["w"]) if MFAB_VARIANTS[n][1] else state["w"])) for n in names}, state, dev, flush)
    if MFAB_L2H:   # the bs <= 4 build with and without L2 hints
        try:
            sfxs = tuple(state.get("sfxs", state["sfx"]))
            tv = {"base": (_ext(tuple(TINY_DEFINES) + ("-DOKC_L2H=0",) + sfxs), state["w"]), "prod": (_ext(tuple(TINY_DEFINES) + sfxs), state["w"])}
            for bs in (1, 4):
                _mfab_time(bs, tv, state, dev, flush)
        except Exception as ex:
            print(f"[mfab] l2h A/B failed: {str(ex)[-400:]}", flush=True)


def _mfab_time(bs, variants, state, dev, flush):
    names = list(variants)
    res = {n: [] for n in names}
    ratio = {n: [] for n in names}
    maxd = {n: 0.0 for n in names}
    for seed in range(MFAB_SEEDS):
        d = _gen_input(bs, 777 + 13 * seed + bs, dev)
        x, idt, tw = d["hidden_states"], d["topk_ids"].to(torch.int32).contiguous(), d["topk_weights"].float().contiguous()
        outs = {n: torch.empty((bs, HIDDEN), dtype=torch.bfloat16, device=dev) for n in names}
        graphs = {}
        for n in names:
            def fn(n=n):
                variants[n][0].okc_run(variants[n][1], state["scratch"], x, idt, tw, outs[n], LOCAL_START, LOCAL_EXPERTS, state["grid"])
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            graphs[n] = g
        torch.cuda.synchronize()
        for n in names:
            maxd[n] = max(maxd[n], float((outs[n].float() - outs["base"].float()).abs().max()))
        ts = {n: [] for n in names}
        for r in range(MFAB_REPS):
            order = names if r % 2 == 0 else names[::-1]
            for n in order:
                flush.add_(1)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[n].replay()
                e1.record()
                torch.cuda.synchronize()
                ts[n].append(e0.elapsed_time(e1) * 1000.0)
        for n in names:
            res[n].append(sorted(ts[n])[len(ts[n]) // 2])
            pr = sorted(a / b for a, b in zip(ts["base"], ts[n]))
            ratio[n].append(pr[len(pr) // 2])
    for n in names:
        rs = sorted(ratio[n])
        print(f"[mfab] bs {bs:3d} {n:5s} median {sorted(res[n])[len(res[n]) // 2]:7.1f} us  speedup vs base {rs[len(rs) // 2]:.4f} "
              f"(per seed {' '.join(f'{v:.3f}' for v in ratio[n])})  max|d| vs base {maxd[n]:.4g}", flush=True)


def _inter_sentinel(inter):
    """DF: the first INTER_SENT_ROWS rows of inter start (and, after every launch, end) as bf16 0xFFFF = not yet written."""
    inter[128 * HIDDEN:256 * HIDDEN].view(torch.int16).fill_(-1)
    return inter


