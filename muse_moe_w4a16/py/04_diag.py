# ---------------------------------------------------------------------------------------------- diagnostics
# Everything below runs only when DIAG is set (prepare) or when called
# directly. It is development tooling, not part of the submission's timed
# path, and it is the only place host synchronisation appears besides
# prepare()'s one-time compile wait.
def _gen_input(bs, seed, dev):
    """Same distribution as the task's generate_input (normal dist, busiest-GPU relabel), written here."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, N_EXPERTS_TOTAL), generator=gen, device=dev)
    z = torch.randn((N_EXPERTS_TOTAL,), generator=gen, device=dev)
    u = torch.rand((N_EXPERTS_TOTAL,), generator=gen, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(TOP_K, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // LOCAL_EXPERTS).flatten(), minlength=TOP_K)
    ids = (ids - LOCAL_EXPERTS * pairs.argmax()) % N_EXPERTS_TOTAL
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": bs, "d_hidden": HIDDEN, "d_expert": HIDDEN, "n_experts": N_EXPERTS_TOTAL, "local_start": LOCAL_START,
                       "local_count": LOCAL_EXPERTS, "top_k": TOP_K, "sharding": SHARDING}}


def _time_graph(fn, reps=40):
    """Median / min CUDA-graph replay time of fn in microseconds."""
    dev = torch.device("cuda")
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.add_(1)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    ts.sort()
    del g
    return ts[len(ts) // 2], ts[0]


def _diag_bw(state):
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major == 9 else "")
    ext = load_inline(name=f"okc_bw_v216x_sm{arch}", cpp_sources="#include <torch/extension.h>\nvoid bw_run(torch::Tensor src, torch::Tensor out, int64_t mode, int64_t grid);",
                      cuda_sources=BW_SRC, functions=["bw_run"],
                      extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"], verbose=False)
    src = state["w"][0]
    nbytes = src.numel() * src.element_size()
    out = torch.zeros(4, dtype=torch.int32, device=src.device)
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=src.device)
    names = {5: "tma 6x16KB", 8: "tma 4x32KB", 12: "tma 4x16KB 2cta/sm", 16: "tma 5x36KB",
             17: "tma 4x36KB", 18: "tma 4x36KB as 2x18KB", 19: "tma 3x36KB", 20: "tma 5x32KB", 21: "tma2 8x16KB",
             22: "tma2 4x16KB", 23: "tma2 6x16KB", 24: "tma 2x36KB 2cta/sm"}
    for mode in DIAG_BW_MODES:
        if mode >= 5 and mode != 7 and major < 9:
            continue
        ts = []
        for _ in range(5):
            flush.add_(1)
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            ext.bw_run(src, out, mode, state["grid"])
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000.0)
        ts.sort()
        print(f"[diag-bw] {names[mode]:18s}: {ts[2]:8.1f} us  {nbytes / ts[2] / 1e6:6.2f} TB/s", flush=True)


def _diag(state, weights):
    import time
    dev = weights["w13"].device
    t0 = time.time()
    if DIAG_BW:
        try:
            _diag_bw(state)
        except Exception as ex:  # diagnostics must never break the run
            print(f"[diag-bw] failed: {ex}", flush=True)
    shapes = [1, 16, 32, 8, 128, 256]
    variants = {"main": tuple(MAIN_DEFINES)}
    variants.update(DIAG_VARIANTS)
    # compile the variants concurrently (separate extension dirs; nvcc runs in subprocesses)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_ext, [d for d in variants.values() if tuple(d) not in _EXTS]))
    print(f"[diag] compiled {len(variants)} variants in {time.time() - t0:.0f}s", flush=True)
    lines = []
    for bs in shapes:
        d = _gen_input(bs, 1234 + bs, dev)
        ids = d["topk_ids"]
        touched = int(torch.unique(ids[(ids >= 0) & (ids < LOCAL_EXPERTS)]).numel())
        row = [f"bs={bs:3d} touched={touched:2d}"]
        for name, defs in variants.items():
            ext = _ext(defs)
            x = d["hidden_states"]
            idt = ids.to(torch.int32).contiguous()
            tw = d["topk_weights"].float().contiguous()
            out = torch.empty((bs, HIDDEN), dtype=torch.bfloat16, device=dev)

            def fn():
                ext.okc_run(state["w"], state["scratch"], x, idt, tw, out, LOCAL_START, LOCAL_EXPERTS, state["grid"])

            med, mn = _time_graph(fn)
            # 28.31 MB: approx. weight bytes streamed per touched expert
            # (NVFP4 codes + scales for gate/up + down), diagnostic only.
            row.append(f"{name}: {med:7.1f}us ({touched * 28.31 / med:4.2f} TB/s)")
        lines.append("  ".join(row))
        print("[diag] " + lines[-1], flush=True)
    # timeline of the main variant with debug timestamps
    for dbgname in [k for k, v in variants.items() if "-DOKC_DEBUG=1" in v]:
        ext = _ext(variants[dbgname])
        G = state["grid"]
        dbg = torch.zeros(G * 8 + G * 8 * 4, dtype=torch.int64, device=dev)
        sc = list(state["scratch"]) + [dbg]
        names = ["start", "routed", "A done", "B done", "barrier", "final"]
        for bs in shapes:
            d = _gen_input(bs, 1234 + bs, dev)
            x = d["hidden_states"]
            idt = d["topk_ids"].to(torch.int32).contiguous()
            tw = d["topk_weights"].float().contiguous()
            out = torch.empty((bs, HIDDEN), dtype=torch.bfloat16, device=dev)
            flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
            for _ in range(3):
                flush.add_(1)
                dbg.zero_()
                torch.cuda.synchronize()
                ext.okc_run(state["w"], sc, x, idt, tw, out, LOCAL_START, LOCAL_EXPERTS, G)
                torch.cuda.synchronize()
            tt = dbg[:G * 8].view(G, 8).cpu().double()
            t00 = tt[:, 0][tt[:, 0] > 0].min()
            parts = []
            for k, n in enumerate(names):
                col = tt[:, k][tt[:, k] > 0]
                if col.numel():
                    parts.append(f"{n} {(col.min() - t00) / 1000:.1f}-{(col.max() - t00) / 1000:.1f}")
            print(f"[diag] timeline {dbgname} bs={bs}: " + "  ".join(parts), flush=True)
    print(f"[diag] total {time.time() - t0:.0f}s", flush=True)

