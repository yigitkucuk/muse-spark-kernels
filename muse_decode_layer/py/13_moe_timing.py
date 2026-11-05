def _mf_gen_input(bs, seed, dev):
    """Same distribution as the task's generate_input (busiest-GPU relabel); local experts are 0..31."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=dev)
    z = torch.randn((256,), generator=gen, device=dev)
    u = torch.rand((256,), generator=gen, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return x, ids.to(torch.int32).contiguous(), (routed.values / routed.values.sum(-1, keepdim=True)).float().contiguous()


def _mf_cases(sfx):
    main = tuple(_main_defines(sfx))
    return [(32, tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)), (128, main), (256, main)]


def _mf_start_base_builds(weights):
    """Compile the barrier twins (-DOKC_MFIN=0) of the builds the A/B times, on a CPU thread, while the layer prepares."""
    def run():
        try:
            sfx = _MA_SFX_IF_PRMT if (_MA_PRMT_SCALES and _ma_scales_prmt_exact(weights["w13_sf"], weights["w2_sf"])) else ()
            if not sfx:
                return
            off = ("-DOKC_SC6OFF=1",) if ("_sc6_ok" in globals() and SC6_WGS and _wgs_on() and _sc6_ok(weights["w13_sf"], weights["w2_sf"])) else ()
            for _, defs in _mf_cases(sfx):
                _ma_ext(tuple(defs) + ("-DOKC_MFIN=0",) + off + tuple(sfx))
        except Exception as ex:
            print(f"[mfab] base build failed: {str(ex)[-400:]}", flush=True)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def _mf_ns_weights(st, weights):
    """The plain-scale WGS weight copies (state weights are SC6-packed when SC6 applies): the barrier twin reads these."""
    I = H = 4096
    rb = torch.arange(I // 8, device=weights["w13"].device)
    g = torch.arange(8, device=weights["w13"].device)
    rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    rowsB = torch.arange(H, device=weights["w13"].device)
    return st["ma_w"][:12] + [torch.cat([_wgs_copy(weights["w13"][e].view(torch.uint8), weights["w13_sf"][e], _kperm_rows(rowsA), False) for e in range(32)]),
                              torch.cat([_wgs_copy(weights["w2"][e].view(torch.uint8), weights["w2_sf"][e], rowsB, False) for e in range(32)])]


def _mf_ab(st, t0, weights):
    dev = torch.device("cuda")
    sfx = tuple(st["ma_sfx"])
    sc6 = bool(globals().get("_MA_SC6W", False))
    w_ns = _mf_ns_weights(st, weights) if sc6 else st["ma_w"]
    if not sfx:
        print("[mfab] no PRMT scales: the MAIN build is the non-WGS fallback, nothing to compare", flush=True)
        return
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
    for bs, defs in _mf_cases(sfx):
        prod = _ma_ext(tuple(defs) + sfx)
        base = _ma_ext(tuple(defs) + ("-DOKC_MFIN=0",) + (("-DOKC_SC6OFF=1",) if sc6 else ()) + sfx)
        ratios, meds, maxd = [], [], 0.0
        for seed in range(MFAB_LAYER_SEEDS):
            x, ids, tw = _mf_gen_input(bs, 777 + 13 * seed + bs, dev)
            outs = {n: torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev) for n in ("base", "prod")}
            graphs = {}
            for n, ext in (("base", base), ("prod", prod)):
                def fn(n=n, ext=ext):
                    ext.okc_run(w_ns if n == "base" else st["ma_w"], st["ma_scratch"], x, ids, tw, outs[n], 0, 32, st["nsm"])
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
            maxd = max(maxd, float((outs["prod"].float() - outs["base"].float()).abs().max()))
            ts = {"base": [], "prod": []}
            for r in range(MFAB_LAYER_REPS):
                for n in (("base", "prod") if r % 2 == 0 else ("prod", "base")):
                    flush.add_(1)
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[n].replay()
                    e1.record()
                    torch.cuda.synchronize()
                    ts[n].append(e0.elapsed_time(e1) * 1000.0)
            pr = sorted(a / b for a, b in zip(ts["base"], ts["prod"]))
            ratios.append(pr[len(pr) // 2])
            meds.append(sorted(ts["prod"])[len(ts["prod"]) // 2])
        rs = sorted(ratios)
        print(f"[mfab] moe T={bs:3d} prod median {sorted(meds)[len(meds) // 2]:7.1f} us  speedup vs barrier twin ({'twin without SC6' if sc6 else 'same weights'}) {rs[len(rs) // 2]:.4f} "
              f"(per seed {' '.join(f'{v:.3f}' for v in ratios)})  max|d| {maxd:.4g}", flush=True)


_mfin_prior_prepare = prepare


def prepare(weights):
    t0 = _rq_time.time()
    th = _mf_start_base_builds(weights) if MFAB_LAYER else None
    st = _mfin_prior_prepare(weights)
    print(f"[mfab] layer prepare done at {_rq_time.time() - t0:.0f}s", flush=True)
    if MFAB_LAYER and th is not None:
        th.join(timeout=max(1.0, 195.0 - (_rq_time.time() - t0)))
        if _rq_time.time() - t0 < 205.0 and not th.is_alive():
            try:
                _mf_ab(st, t0, weights)
            except Exception as ex:  # diagnostics must never break the run
                print(f"[mfab] A/B failed: {str(ex)[-400:]}", flush=True)
        else:
            print(f"[mfab] skipped at {_rq_time.time() - t0:.0f}s", flush=True)
        torch.cuda.synchronize()
    return st


# ==============================================================================================
# Prepared-state hygiene. Everything chosen from a weight set during prepare() lives in the state that prepare()
# returns, and kernel_fn binds the module-level switches the builders read from that state on every call, so a second
# prepare() with other weights (a held-out rerun in the same process) can never leave a call on stale picks:
#   * the expert decoder choice (_MA_SFX: PRMT scale decode, _MA_SC6W: 6-bit block scales) is stored as
#     state["ma_sfx"] / state["ma_sc6w"] and re-bound at kernel_fn entry;
#   * the module-level MoE build handles are cleared at the start of every prepare(), so the builds of the new weight set
#     are derived from the new switches (the compiled modules themselves are cached by their full define lists);
#   * _calib_bias restores the router bias on every exit path (the tuner calls it, then restores at its end).
# ==============================================================================================
_rb_prior_calib_bias = _calib_bias


