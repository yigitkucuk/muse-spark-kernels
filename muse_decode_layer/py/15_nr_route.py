def _nr_patch(src):
    """Drop the release-atomic handshake and the last arriver's partial sum from a pooler/router kernel source."""
    out, n = _nr_re.subn(r"(    __shared__ int s_rlast;\n)    if \(tid == 0\) \{\n      int old;.*?\n    if \(s_rlast\) \{.*?\n    \}\n",
                         r"\1", src, count=1, flags=_nr_re.S)
    assert n == 1 and out.count("s_rlast") == 1, "handshake block not found"
    return out


def _nr_bfp_ext():
    sfx = f"_dly{bfp__SK2_DELAY}" if bfp__SK2_DELAY else ""
    common = bfp_CUDA_SRC.replace("@@MBAR@@", bfp_MBAR_HELPERS).replace("@@BODY@@", bfp_WG_HELPERS)
    src = common + bfp_wgmma_defs([8, 16, 32, 64, 128]) + bfp__tma_srcs(True) + bfp_MAP_SRC + _nr_patch(bfp__sk2_src(True))
    cpp = ("#include <torch/extension.h>\n"
           "void sk2f(torch::Tensor Ws, torch::Tensor Wsr, torch::Tensor x, torch::Tensor out, torch::Tensor outr, torch::Tensor ptab,"
           " int64_t pmax, int64_t base4, torch::Tensor part, torch::Tensor rcnt, int64_t M, int64_t tokr);\n"
           "void sk2_set_pdl(int64_t n);\n")
    return load_inline(name="dl_okc_bf16proj_v86y4_90a_norh" + sfx, cpp_sources=cpp, cuda_sources=src,
                       functions=["sk2f", "sk2_set_pdl"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"]
                       + ([f"-DSK2_DELAY_ESC={bfp__SK2_DELAY}"] if bfp__SK2_DELAY else []),
                       no_implicit_headers=True, verbose=False)


def _nr_gle_ext():
    source = _nr_patch(_gle_source())
    tag = _hashlib.md5(source.encode()).hexdigest()[:12]
    source = (source.replace("sk2_kernel", "sk2_kernel_gle" + tag).replace("launch_sk2", "launch_sk2_gle" + tag)
              .replace("void sk2(", "void sk2_gle" + tag + "(").replace("  sk2(Ws,", "  sk2_gle" + tag + "(Ws,")
              .replace("void sk2f(", "void sk2f_gle" + tag + "(").replace("  sk2f(Ws,", "  sk2f_gle" + tag + "(Ws,"))
    cpp = ("void gle_o(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,"
           " int64_t M, torch::Tensor s, torch::Tensor a, torch::Tensor b, torch::Tensor a2, torch::Tensor p,"
           " torch::Tensor uv, torch::Tensor part, torch::Tensor uvf);\n"
           "void gle_pr(torch::Tensor Ws, torch::Tensor Wsr, torch::Tensor x, torch::Tensor out, torch::Tensor outr,"
           " torch::Tensor ptab, int64_t pmax, int64_t base4, torch::Tensor part, torch::Tensor rcnt, int64_t M,"
           " torch::Tensor x32, torch::Tensor w32, int64_t wpf);\nvoid sk2_set_pdl(int64_t n);\n")
    return load_inline("gle_o_" + tag, "#include <torch/extension.h>\n" + cpp, cuda_sources=source,
                       functions=["gle_o", "gle_pr", "sk2_set_pdl"], no_implicit_headers=True,
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"], verbose=False)


def _nr_build(key, fn):
    t0 = _nr_time.time()
    try:
        _NR[key] = fn()
        print(f"[nr] built {key} in {_nr_time.time() - t0:.0f}s", flush=True)
    except BaseException as ex:
        _NR_ERR[key] = ex
        print(f"[nr] build {key} failed: {type(ex).__name__}: {str(ex)[-300:]}", flush=True)


def _nr_start():
    if not _NR_THREADS:
        for key, fn in (("bfp", _nr_bfp_ext), ("gle", _nr_gle_ext)):
            th = _nr_threading.Thread(target=_nr_build, args=(key, fn), daemon=True)
            th.start()
            _NR_THREADS.append(th)


@triton.jit
def _nr_logit(LP, row, e):
    # router logit of expert e for token row: the four quarter-K partials added in quarter order, as the handshake's last arriver did
    pb = LP + (e >> 3) * 1024 + row * 8 + (e & 7)
    return ((tl.load(pb) + tl.load(pb + 256)) + tl.load(pb + 512)) + tl.load(pb + 768)


@triton.jit
def _nr_gle_route(POOLED, LOGITS, PART, T, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, RT,
               PO, S, ALPHA, BETA, ALPHA2, S1,
               MODE: tl.constexpr, PARTS: tl.constexpr, NCH: tl.constexpr, TT: tl.constexpr, PSUM: tl.constexpr = False):
    # _sl_tile_route on the two-row (u, v) pooler / router outputs; r1 / r2 come from the o epilogue's partial sums,
    # complete before this launch (o completes before the pooler wait returns; the FP32 router is a plain launch)
    t = tl.program_id(0)
    kind = tl.program_id(1)
    k8 = tl.arange(0, 8)
    cta = tl.arange(0, 256)
    q = tl.sum(tl.load(PART + (cta[:, None] * T + t) * 8 + k8[None, :]), 0)
    q0 = tl.sum(tl.where(k8 == 0, q, 0.0), 0)
    q1 = tl.sum(tl.where(k8 == 1, q, 0.0), 0)
    q2 = tl.sum(tl.where(k8 == 2, q, 0.0), 0)
    q3 = tl.sum(tl.where(k8 == 3, q, 0.0), 0)
    r1 = tl.rsqrt(q0 / 8192.0 + 1e-8)
    r2 = tl.rsqrt((q1 + 2.0 * r1 * q2 + r1 * r1 * q3) / 8192.0 + 1e-5)
    if kind > PARTS:
        # carry programs (no wait: they read o's outputs and the inputs only): column chunk (kind - PARTS - 1) * T + t
        # of 8192 // (NCH * T) columns, for every token
        cc = ((kind - PARTS - 1) * TT + t) * (8192 // (NCH * TT)) + tl.arange(0, 8192 // (NCH * TT))
        for tok in tl.static_range(TT):
            r1t = tl.rsqrt(tl.sum(tl.load(PART + (cta * TT + tok) * 8), 0) / 8192.0 + 1e-8)
            tl.store(S1 + tok * 8192 + cc, _gle_carry(PO, S, ALPHA, BETA, ALPHA2, tok, cc, r1t, TT == 1))
    elif kind < PARTS:
        c = tl.arange(0, 4096)
        oc = kind * (4096 // PARTS) + tl.arange(0, 4096 // PARTS)
        gpre = 1.0 + tl.load(GPRE + oc).to(tl.float32)
        _gdc_wait()
        _gdc_launch()
        pooled = (tl.load(POOLED + t * 4096 + c).to(tl.float32) + tl.load(POOLED + (T + t) * 4096 + c).to(tl.float32) * r1) * r2
        inv = tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5)
        if PARTS == 1:
            vals = pooled
        else:
            vals = (tl.load(POOLED + t * 4096 + oc).to(tl.float32) + tl.load(POOLED + (T + t) * 4096 + oc).to(tl.float32) * r1) * r2
        tl.store(XOUT + t * 4096 + oc, (vals * inv * gpre).to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        bias = tl.load(BIAS + e)
        ls = tl.load(LS)
        q4 = tl.sum(tl.where(k8 == 4, q, 0.0), 0)
        q5 = tl.sum(tl.where(k8 == 5, q, 0.0), 0)
        q6 = tl.sum(tl.where(k8 == 6, q, 0.0), 0)
        tail = q4 + 2.0 * r1 * q5 + r1 * r1 * q6
        _gdc_wait()
        _gdc_launch()
        tl.store(RT + t, r1)
        tl.store(RT + T + t, tail)
        if PSUM:
            logit = (_nr_logit(LOGITS, t, e) + _nr_logit(LOGITS, T + t, e) * r1) * r2
        else:
            logit = (tl.load(LOGITS + t * 256 + e) + tl.load(LOGITS + (T + t) * 256 + e) * r1) * r2
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + bias
        if MODE != 2:
            bits = sel.to(tl.uint32, bitcast=True)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
            packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
            if MODE == 0:
                picked = tl.topk(packed, 8)
            else:
                sorted_values = tl.sort(packed, descending=True)
                picked = tl.gather(sorted_values, k8, 0)
            ids = (255 - (picked & 255)).to(tl.int32)
            wv = tl.gather(sc, ids, 0)
        else:
            ids = tl.zeros([8], dtype=tl.int32)
            wv = tl.zeros([8], dtype=tl.float32)
            for k in tl.static_range(8):
                idx = tl.argmax(sel, 0)
                val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
                ids = tl.where(k8 == k, idx, ids)
                wv = tl.where(k8 == k, val, wv)
                sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - ls + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)


@triton.jit
def _nr_sl_tile_route(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, SPLIT: tl.constexpr, EARLY: tl.constexpr, MODE: tl.constexpr, PARTS: tl.constexpr, PRELOAD: tl.constexpr, PSUM: tl.constexpr = False):
    # _pool_route_finish structure: program (t, 0) = pre-expert norm of the pooled row, (t, 1) = top-8 pick;
    # one warp each, both after the wait. Arithmetic is the base's _route/_route_split (fp32 pooled, 1 + w in fp32).
    t = tl.program_id(0)
    kind = tl.program_id(1)
    if kind<PARTS:
        c=tl.arange(0,4096)
        oc=kind*(4096//PARTS)+tl.arange(0,4096//PARTS)
        if PRELOAD:
            gpre=1.0+tl.load(GPRE+oc).to(tl.float32)
        _gdc_launch()
        _gdc_wait()
        if SPLIT:
            pooled=tl.load(POOLED+t*4096+c).to(tl.float32)
        else:
            pooled=tl.load(ALL+t*4608+c)
        inv=tl.rsqrt(tl.sum(pooled*pooled,0)/4096.0+1e-5)
        if not PRELOAD:
            gpre=1.0+tl.load(GPRE+oc).to(tl.float32)
        if PARTS==1:
            vals=pooled
        elif SPLIT:
            vals=tl.load(POOLED+t*4096+oc).to(tl.float32)
        else:
            vals=tl.load(ALL+t*4608+oc)
        tl.store(XOUT+t*4096+oc,(vals*inv*gpre).to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        if PRELOAD:
            bias=tl.load(BIAS+e)
            ls=tl.load(LS)
        _gdc_launch()
        _gdc_wait()
        if not PRELOAD:
            bias=tl.load(BIAS+e)
            ls=tl.load(LS)
        if PSUM:
            logit = _nr_logit(LOGITS, t, e)
        elif SPLIT:
            logit = tl.load(LOGITS + t * 256 + e)
        else:
            logit = tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + bias
        k8 = tl.arange(0, 8)
        if MODE != 2:
            bits = sel.to(tl.uint32, bitcast=True)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
            packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
            if MODE == 0:
                picked = tl.topk(packed, 8)
            else:
                sorted_values = tl.sort(packed, descending=True)
                picked = tl.gather(sorted_values, k8, 0)
            ids = (255 - (picked & 255)).to(tl.int32)
            wv = tl.gather(sc, ids, 0)
        else:
            ids = tl.zeros([8], dtype=tl.int32)
            wv = tl.zeros([8], dtype=tl.float32)
            for k in tl.static_range(8):
                idx = tl.argmax(sel, 0)
                val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
                ids = tl.where(k8 == k, idx, ids)
                wv = tl.where(k8 == k, val, wv)
                sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - ls + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)

import contextlib as _nr_ctxlib

_NR_T = (1, 8, 16)     # shapes that use the no-handshake pooler/router (measured gain at d1, d16, v1; d32 neutral)


@_nr_ctxlib.contextmanager
def _nr_rebind(**fns):
    g = globals()
    old = {k: g[k] for k in fns}
    g.update(fns)
    try:
        yield
    finally:
        g.update(old)


class _NrRoute:
    """Stand-in for a route kernel: a launch whose LOGITS argument is a placeholder from the no-handshake pooler is redirected to
    the PSUM variant with the partial buffer; any other launch goes to the original kernel."""

    def __init__(self, orig, new):
        self.orig, self.new = orig, new

    def __getitem__(self, grid):
        o, n = self.orig[grid], self.new[grid]

        def call(*args, **kw):
            part = _NR_PART.pop(args[1].data_ptr(), None)
            if part is None:
                return o(*args, **kw)
            return n(args[0], part, *args[2:], PSUM=True, **kw)
        return call


_NR_SLT = _NrRoute(_sl_tile_route, _nr_sl_tile_route)
_NR_GLE = _NrRoute(_gle_route, _nr_gle_route)


def _nr_gle_frt(xb, x32, state):
    """_gle_frt with the no-handshake kernel: logits is a placeholder, the partials go to the route kernel."""
    bst = state["bfp"]
    m = xb.shape[0]
    dev = xb.device
    if "fused_part" not in bst:
        bst["fused_part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=dev)
        bst["fused_cnt"] = torch.zeros(32, dtype=torch.int32, device=dev)
    pci, pstream, ptab_, ppmax, pb4, _ = bst["sk2"]["pooler"]
    pooled = torch.empty((m, 4096), device=dev, dtype=torch.bfloat16)
    logits = torch.empty((m, 256), device=dev, dtype=torch.float32)
    _bfp_pdl(m, "pool")
    ext = _NR["gle"]
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.gle_pr(pstream, bst["sk2"]["router"][1], xb, pooled, logits, ptab_, ppmax, pb4, bst["fused_part"], bst["fused_cnt"],
               m, x32, state["router_fp32"], int(GLE_ILV))
    _NR_PART[logits.data_ptr()] = bst["fused_part"]
    return pooled, logits


def _nr_bfp_fused(x, state):
    """bfp_fused_pooler_router with the no-handshake kernel (m <= 32)."""
    m = x.shape[0]
    assert m <= 32 and state.get("sm90") and "pooler" in state["sk2"] and "router" in state["sk2"]
    ci, stream, tab, pmax, b4, n_rows = state["sk2"]["pooler"]
    rstream = state["sk2"]["router"][1]
    if "fused_part" not in state:
        state["fused_part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=x.device)
        state["fused_cnt"] = torch.zeros(32, dtype=torch.int32, device=x.device)
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    outr = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    ext = _NR["bfp"]
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.sk2f(stream, rstream, x, out, outr, tab, pmax, b4, state["fused_part"], state["fused_cnt"], m, bfp__sk2_tokr(m))
    _NR_PART[outr.data_ptr()] = state["fused_part"]
    return out, outr


def _nr_logits_from(part, m):
    p = part.view(32, 4, 32, 8)
    return (((p[:, 0] + p[:, 1]) + p[:, 2]) + p[:, 3]).permute(1, 0, 2).reshape(32, 256)[:m]


def _nr_selftest(st, dev):
    """Kernel-level bit identity of the no-handshake kernels (partial-sum logits and pooled output) against the handshake kernels."""
    gen = torch.Generator(device=dev).manual_seed(11)
    bst = st["bfp"]
    ok = True
    for m in (2, 8):       # the T = 1 (u, v rows) and T = 8 FP32-router launches
        xb = torch.randn((m, 8192), generator=gen, device=dev).to(torch.bfloat16)
        x32 = torch.randn((m, 8192), generator=gen, device=dev)
        _NR_PART.clear()
        pb, lb = _gle_frt(xb, x32, st)
        pn, ln = _nr_gle_frt(xb, x32, st)
        torch.cuda.synchronize()
        ok &= bool(torch.equal(_nr_logits_from(_NR_PART.pop(ln.data_ptr()), m), lb)) and bool(torch.equal(pn, pb))
    for m in (16,):        # the T = 16 BF16 hi|lo launch
        x = torch.randn((m, 8192), generator=gen, device=dev).to(torch.bfloat16)
        pb, lb = bfp_fused_pooler_router(x, bst)
        pn, ln = _nr_bfp_fused(x, bst)
        torch.cuda.synchronize()
        ok &= bool(torch.equal(_nr_logits_from(_NR_PART.pop(ln.data_ptr()), m), lb)) and bool(torch.equal(pn, pb))
    _NR_PART.clear()
    return ok


_nr_start()
_nr_prior_prepare = prepare


def prepare(weights):
    st = _nr_prior_prepare(weights)
    st["nr_on"] = False
    try:
        for th in _NR_THREADS:
            th.join(timeout=120)
        if not _NR_ERR and "bfp" in _NR and "gle" in _NR and st.get("bfp") is not None:
            if _nr_selftest(st, st["qkv"].device):
                # the link chain consumes the placeholders; pin it (and the fused pooler/router) at T = 16
                st.setdefault("compose_link_enabled", {})[16] = True
                st.setdefault("mm_pick", {})[(16, "pr")] = "bfpf"
                st["nr_on"] = True
        print(f"[nr] no-handshake pooler/router: {'on' if st['nr_on'] else 'off'} (errors {sorted(_NR_ERR)})", flush=True)
    except BaseException as ex:
        st["nr_on"] = False
        print(f"[nr] off: {type(ex).__name__}: {str(ex)[-300:]}", flush=True)
    return st


_nr_prior_kernel_fn = kernel_fn


def kernel_fn(data, state):
    if state.get("nr_on") and data["config"]["tokens"] in _NR_T:
        _NR_PART.clear()
        try:
            with _nr_rebind(_gle_frt=_nr_gle_frt, bfp_fused_pooler_router=_nr_bfp_fused, _sl_tile_route=_NR_SLT, _gle_route=_NR_GLE):
                res = _nr_prior_kernel_fn(data, state)
            if _NR_PART:     # a placeholder nobody consumed would mean a route kernel read unwritten logits
                raise RuntimeError("norh: unconsumed router partials")
        finally:
            _NR_PART.clear()
        return res
    return _nr_prior_kernel_fn(data, state)


_RP_ORIG_BOUNDARY = _link_boundary


def _rp_link_boundary(P, S, T, stream, st, post, s_out, h_out, nu=HIDDEN, router_out=None):
    if T == 16 and S == 1 and nu == HIDDEN and not (not post and router_out is not None):
        br = P.view(T, -1)
        if br.shape[1] == HIDDEN and br.dtype == torch.bfloat16:
            g = st["link"]
            pfx = "2" if post else "1"
            _link_al(br, stream, s_out, h_out)
            _link_row_boundary_pre[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx], g["beta" + pfx],
                                          h_out, s_out, HIDDEN, HIDDEN, post, 4, 32, num_warps=32, launch_pdl=True)
            return True
    return _RP_ORIG_BOUNDARY(P, S, T, stream, st, post, s_out, h_out, nu=nu, router_out=router_out)


_link_boundary = _rp_link_boundary
_rp_prior_prepare = prepare


def prepare(weights):
    st = _rp_prior_prepare(weights)
    # the row kernels are called by the link chain; pin it (with the fused pooler/router) at T = 16 (measured neutral by itself)
    st.setdefault("compose_link_enabled", {})[16] = True
    st.setdefault("mm_pick", {})[(16, "pr")] = "bfpf"
    return st


# ---- data-flag (DF) sentinel rows: moved out of the way of the other MoE builds.
# The DF builds (T=9..16 today, T=5..8 with df8) read the A output ("inter") with a per-lane sentinel check (bf16 0xFFFF = not yet written) and give the
# sentinel back at the end of every launch. They used inter rows [128, 256), which the T=17..32 and T=128/256 builds overwrite with real data (inter rows
# 0 .. M*8-1), so the first DF launch after such a launch could accept last launch's stale values as "written". Rows 2560.. are never written by any build at T <= 256
# (inter rows stay below 2056), so they keep the sentinel across every other launch. No runtime cost: the same kernel, one constant changed.
_DFT_ROW0 = 2560
_DFT_OLD = "p.inter += (size_t)128 * 4096;"
_DFT_NEW = "p.inter += (size_t)%d * 4096;" % _DFT_ROW0
_dft_prev_patch = _df_patch


