def _helper0_src(src):
    """WSH0 twin of the CUDA source: every threadIdx.x becomes TIDX = (threadIdx.x + NT) % NTL, so the TMA helper warp
    (launched last) is thread 0..31 of the CTA's thread numbering (warp scheduler slot 0). Same text as the
    separately stored copy except for its WGS-only regions, which MID never compiles."""
    s = src.replace("threadIdx.x", "TIDX")
    anchor = "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);"
    k = s.index(anchor)
    k = s.index("\n", k) + 1
    blk = ("#ifndef OKC_WSH0\n#define OKC_WSH0 0\n#endif\n#if OKC_WSH0\n#define TIDX ((int)((threadIdx.x + NT) % NTL))\n"
           "#else\n#define TIDX ((int)threadIdx.x)\n#endif\n")
    return s[:k] + blk + s[k:]


ALT_SRCS["var"] = VAR_SRC
ALT_SRCS["helper0"] = _helper0_src(VAR_SRC)
TINY_DEFINES += ("-DOKC_DF=0", "@var")
TINY2_DEFINES += ("-DOKC_DF=0", "@var")
MID_DEFINES += ("-DOKC_WSH0=1", "@helper0")


import ctypes as _C
import threading as _ContextThreading

_CONTEXT_TLS = _ContextThreading.local()
_CONTEXT_API = None


def _context_api():
    global _CONTEXT_API
    if _CONTEXT_API is None:
        api = _C.CDLL("libcuda.so.1")
        for name, args in (
            ("cuCtxGetCurrent", [_C.POINTER(_C.c_void_p)]),
            ("cuCtxGetDevice", [_C.POINTER(_C.c_int)]),
            ("cuDeviceGet", [_C.POINTER(_C.c_int), _C.c_int]),
            ("cuDevicePrimaryCtxRetain", [_C.POINTER(_C.c_void_p), _C.c_int]),
            ("cuCtxSetCurrent", [_C.c_void_p]),
        ):
            function = getattr(api, name)
            function.argtypes, function.restype = args, _C.c_int
        _CONTEXT_API = api
    return _CONTEXT_API


def _context_check(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed with CUDA driver status {status}")


def _bind_context(x):
    index = x.device.index if x.device.index is not None else 0
    if getattr(_CONTEXT_TLS, "device", None) == index:
        return
    api = _context_api()
    context = _C.c_void_p()
    _context_check(api.cuCtxGetCurrent(_C.byref(context)), "cuCtxGetCurrent")
    current_device = _C.c_int(-1)
    if context.value:
        _context_check(api.cuCtxGetDevice(_C.byref(current_device)), "cuCtxGetDevice")
    if not context.value or current_device.value != index:
        device, primary = _C.c_int(), _C.c_void_p()
        _context_check(api.cuDeviceGet(_C.byref(device), index), "cuDeviceGet")
        _context_check(api.cuDevicePrimaryCtxRetain(_C.byref(primary), device), "cuDevicePrimaryCtxRetain")
        _context_check(api.cuCtxSetCurrent(primary), "cuCtxSetCurrent")
    _CONTEXT_TLS.device = index


_context_parent_kernel = kernel_fn


def kernel_fn(data, state):
    _bind_context(data["hidden_states"])
    return _context_parent_kernel(data, state)


# ------------------------------------------------------- per-batch-size builds
# The benchmark times a fixed set of batch sizes; each size listed here runs its own build (branching on shapes is allowed).
# Every other size keeps the dispatch above.
PS_SHAPES = {8: ('tiny2', {}, 't2_wsi')}
PS_PATCHES = {'t2_wsi': ('var', [('\nconstexpr int HD = 4096, ID = 4096;\nconstexpr int NW = OKC_NW, NT = NW * 32;\nconstexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);  // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)\nconstexpr int NRB = OKC_WS ? OKC_WSNR : 2;   // cross-warp partial-sum buffers\n#if OKC_WS || OKC_WGS\n// CTA barrier over the NT compute threads only (the helper warp syncs with them through mbarriers and barrier 2)\n', '\nconstexpr int HD = 4096, ID = 4096;\nconstexpr int NW = OKC_NW, NT = NW * 32;\n#ifndef OKC_WSI\n#define OKC_WSI 1   // wsi: dedicated ring-issuer warp in the skinny WS builds (make_wsi.py)\n#endif\nconstexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? (OKC_WSI ? 64 : 32) : 0);  // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)\nconstexpr int NRB = OKC_WS ? OKC_WSNR : 2;   // cross-warp partial-sum buffers\n#if OKC_WS || OKC_WGS\n// CTA barrier over the NT compute threads only (the helper warp syncs with them through mbarriers and barrier 2)\n'), ("        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);\n      }\n    }\n#pragma unroll\n    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n    // WSB: the compute warps keep the base loop's CTA barrier per unit (slot consumed + partials visible), thread 0\n", "        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);\n      }\n    }\n#if OKC_WSI\n    __syncwarp();\n    if (lane == 0) mbar_arrive1(S.wse + sl);   // wsi: this warp's last read of the slot; the issuer warp may refill it\n#endif\n#pragma unroll\n    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n    // WSB: the compute warps keep the base loop's CTA barrier per unit (slot consumed + partials visible), thread 0\n"), ('    CTA_SYNC();\n    if (tid == 0) {\n      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);\n      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);\n      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1\n    }\n  }\n', '    CTA_SYNC();\n    if (tid == 0) {\n      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);\n      if (!OKC_WSI && q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);\n      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1\n    }\n  }\n'), ('  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];\n  const int nA = a1 - a0, N = nA + (b1 - b0);\n  const int njF = S.njF;\n  if (lane == 0)\n    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);\n  const int gg = lane >> 2, tt = lane & 3;\n  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;\n', '  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];\n  const int nA = a1 - a0, N = nA + (b1 - b0);\n  const int njF = S.njF;\n  if (!OKC_WSI && lane == 0)\n    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);\n  const int gg = lane >> 2, tt = lane & 3;\n  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;\n'), ('  }\n  __threadfence();\n  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier\n}\n#endif\n\n', "  }\n  __threadfence();\n  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier\n}\n#endif\n\n#if OKC_WS && OKC_WSI\n// wsi: the issuer warp. Fills the ring (after FAST1's pre-issued units), then refills each slot with unit u as soon as\n// every compute warp has released it from unit u - NS (S.wse, count NW): the refill leaves the compute warps' critical\n// path and goes out before their accumulator reduce, partial store and CTA barrier instead of after them.\n__device__ void ws_issuer(const Params& p, Smem& S, unsigned char* ring) {\n  const int lane = threadIdx.x & 31;\n  WS_JOIN();  // routing tables and this CTA's ranges are ready (same join as the helper warp's first)\n  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];\n  const int nA = a1 - a0, N = nA + (b1 - b0);\n  if (lane == 0) {\n    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);\n    for (int u = OKC_NS; u < N; ++u) {\n      const int q = u - OKC_NS;\n      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));\n      ws_issue(p, S, u, nA, a0, b0, ring);\n    }\n  }\n  __syncwarp();\n  WS_JOIN();  // pairs with the compute warps' and the helper's last join\n}\n#endif\n\n"), ('  // scratch, so wait for it here; without the launch attribute this returns at once\n  asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n#if OKC_WS\n  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }\n#endif\n#if OKC_WGS\n', '  // scratch, so wait for it here; without the launch attribute this returns at once\n  asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n#if OKC_WS\n#if OKC_WSI\n  if (tid >= NT + 32) { ws_issuer(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }\n#endif\n  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }\n#endif\n#if OKC_WGS\n')])}

for _fn, (_from, _edits) in PS_PATCHES.items():
    _s = CUDA_SRC if _from == "main" else ALT_SRCS[_from]
    for _old, _new in _edits:
        assert _s.count(_old) >= 1, f"ps patch {_fn}: anchor not found: {_old[:80]!r}"
        _s = _s.replace(_old, _new)
    ALT_SRCS[_fn] = _s


def _ps_defs(M, state):
    build, over, alt = PS_SHAPES[M]
    base = tuple(_main_defines(state["sfx"])) if build == "main" else {"tiny": TINY_DEFINES, "tiny2": TINY2_DEFINES, "mid": MID_DEFINES}[build]
    out = []
    for d in base:
        if d.startswith("-D") and d[2:].split("=")[0] in over:
            continue
        if d.startswith("@") and alt:
            continue
        out.append(d)
    out += [f"-D{k}={v}" for k, v in over.items() if v is not None]
    if alt:
        out.append("@" + alt)
    sfx = tuple(state["sfxm"] if build == "main" else state.get("sfxs", state["sfx"]))
    return tuple(out) + sfx


_ps_parent_prepare = prepare


def prepare(weights):
    import concurrent.futures
    state = _ps_parent_prepare(weights)
    defs = {M: _ps_defs(M, state) for M in PS_SHAPES}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_ext, list(dict.fromkeys(defs.values()))))
    state["ps"] = defs
    dev = weights["w13"].device
    for M, d in defs.items():  # each dedicated build against the base dispatch on its own size (diagnostic print only)
        try:
            inp = _gen_input(M, 4242 + M, dev)
            ref = _ps_parent_kernel(inp, state)
            x = inp["hidden_states"].contiguous()
            ids = inp["topk_ids"].to(torch.int32).contiguous()
            tw = inp["topk_weights"].to(torch.float32).contiguous()
            out = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=dev)
            _ext(d).okc_run(state["w"], state["scratch"], x, ids, tw, out, state["lo"], state["cnt"], state["grid"])
            torch.cuda.synchronize()
            rel = float((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30))
            print(f"[ps] bs {M}: {PS_SHAPES[M][0]} {PS_SHAPES[M][2] or ''} {PS_SHAPES[M][1]} regs {_ext(d).okc_attrs()} rel vs base {rel:.2e}", flush=True)
        except Exception as ex:
            print(f"[ps] bs {M}: check failed: {str(ex)[-300:]}", flush=True)
    return state


_ps_parent_kernel = kernel_fn


def kernel_fn(data, state):
    ps = state.get("ps")
    M = data["hidden_states"].shape[0]
    if ps is None or M not in ps or M > CHUNK_TOKENS:
        return _ps_parent_kernel(data, state)
    _bind_context(data["hidden_states"])  # as the base wrapper does (fresh OS threads)
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    if ids.data_ptr() & 15:
        ids = ids.clone()
    tw = data["topk_weights"].to(torch.float32).contiguous()
    out = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=x.device)
    _ext(ps[M]).okc_run(state["w"], state["scratch"], x, ids, tw, out, state["lo"], state["cnt"], state["grid"])
    return out
