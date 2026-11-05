FMHA_CUDA_SRC = __cuda__("fmha.cu")
FMHA_CPP = r"""
void fmha_ng(torch::Tensor q, torch::Tensor kvt, torch::Tensor out, torch::Tensor po, torch::Tensor pml,
                torch::Tensor lens, torch::Tensor k_scale, torch::Tensor v_scale, double smqk_log2e,
                int64_t B, int64_t L, int64_t capblks, int64_t S, int64_t win, int64_t nc,
                torch::Tensor gate, int64_t gstride, double ng_eps);
void gatherp(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor kvt, torch::Tensor page_table,
             torch::Tensor lens, torch::Tensor seq, torch::Tensor lens_out, int64_t pt_stride, int64_t pagelog,
             int64_t capblks, int64_t B, int64_t L, int64_t win);
"""
_FMHA = {}


def _fmha_build():
    try:
        major = torch.cuda.get_device_capability()[0]
        if major < 9:
            return
        _FMHA["ext"] = load_inline(
            name="pfmha_n72_exact_v1" if FMHA_EXACT else "pfmha_n72_v1",  # N72: own build (A / B in one process)
            cpp_sources=[FMHA_CPP], cuda_sources=[FMHA_CUDA_SRC], functions=["fmha_ng", "gatherp"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a"]
            + (["-DPFMHA_EXACT=1"] if FMHA_EXACT else []), verbose=False)
    except Exception as exc:  # noqa: BLE001
        _FMHA["err"] = repr(exc)[-600:]


import threading as _fmha_threading
_FMHA_THREAD = _fmha_threading.Thread(target=_fmha_build, daemon=True)
_FMHA_THREAD.start()


def _fmha_pick(L, ntf, B, nsm=132):
    """(consumer warpgroups, split): his pick_cfg4 cost model, with B requests per grid."""
    rows = B * L * 16
    best = (float("inf"), 3, 1)
    for nc in (3, 4):
        units = ((L + 8 * nc - 1) // (8 * nc)) * 2 * B
        for s in range(1, 17):
            waves = -(-units * s // nsm)
            cost = waves * (ntf / s + 20.0) * 1.41 * 0.75 * (nc / 3) * (0.9 if nc == 4 else 1.0)
            if s > 1:
                cost += s * rows * 136 * 2 / 3e12 * 1e6 + 3.0
            if cost < best[0] - 1e-9:
                best = (cost, nc, s)
    return best[1], best[2]


def _fmha_attn(st, data, q8, qkv, cfg):
    """og [T, 1024] bf16 = o_norm(attention) * sigmoid(gate), by his gather_kp + fmha7 (+ combine_ng)."""
    ext = st["fmha"]
    B, L, window, page = cfg["batch"], cfg["q_len"], cfg["window"], cfg["page_size"]
    dev = q8.device
    T = B * L
    pt = data["page_table"]
    pagelog = page.bit_length() - 1
    max_len = pt.shape[1] * page
    lens_full = (data["seq_lens"] + L).to(torch.int32)
    cap = ((min(window - 1 + L, max_len) if window else max_len) + 255) & ~255
    capblks = cap // 128
    kvt = torch.empty((B * 2 * capblks, 16384), dtype=torch.uint8, device=dev)
    kc = data["k_cache"].view(torch.uint8)
    vc = data["v_cache"].view(torch.uint8)
    if window:
        lens = torch.empty_like(lens_full)
        ext.gatherp(kc, vc, kvt, pt, lens_full, data["seq_lens"].to(torch.int32), lens, pt.stride(0), pagelog, capblks,
                    B, L, window)
        ntf = min(capblks, (window + 23) // 128 + 2)
    else:
        lens = lens_full
        none = lens_full[:0]
        ext.gatherp(kc, vc, kvt, pt, lens_full, none, none, pt.stride(0), pagelog, capblks, B, L, 0)
        ntf = capblks
    nc, split = _fmha_pick(L, ntf, B, st.get("nsm", 132))
    out = torch.empty((T, Q_SIZE), dtype=torch.bfloat16, device=dev)
    if split > 1:
        po = torch.empty((split, T * HQ, HD), dtype=torch.float16, device=dev)
        pml = torch.empty((split, T * HQ, 2), dtype=torch.float32, device=dev)
    else:
        po = pml = out
    smqk = cfg["sm_scale"] * cfg["qk_scale_factor"] * 1.4426950408889634
    ext.fmha_ng(q8.view(torch.uint8), kvt, out, po, pml, lens, data["k_scale"], data["v_scale"], smqk, B, L, capblks,
                   split, window, nc, qkv[:, Q_SIZE + 2 * KV_SIZE:], QKV_SIZE, RMS_EPS)
    return out


def _fmha_ok(st, cfg):
    return st.get("fmha") is not None and FMHA_CLASSES.get(_attn_class(cfg), False) and (cfg["page_size"] & (cfg["page_size"] - 1)) == 0


def _fmha_prepare(st):
    """Join the build, then compare against our own kernel on the scored classes; on any mismatch, stay off."""
    _FMHA_THREAD.join()
    st["fmha"] = None
    if "ext" not in _FMHA:
        print("FMHA unavailable:", _FMHA.get("err", "no sm_90"), flush=True)
        return
    st["fmha"] = _FMHA["ext"]
    worst = 0.0
    try:
        for B, L, ctx, window, page in ((1, 753, 3000, 2048, 1), (2, 300, 2500, 0, 1), (1, 200, 5000, 0, 64),
                                        (1, 640, 4096, 2048, 64)):
            data, cfg, q8, qkv = _synthetic_attn_case(B, L, ctx, window, page, 17)
            ref = _attn_ref(data, cfg, q8, qkv)
            got = _fmha_attn(st, data, q8, qkv, cfg)
            torch.cuda.synchronize()
            err = float((got.float() - ref.float()).norm() / ref.float().norm())
            worst = max(worst, err)
            print(f"FMHA check B{B} L{L} ctx{ctx} win{window} page{page}: rel err vs ref {err:.2e}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("FMHA check failed:", repr(exc)[:400], flush=True)
        st["fmha"] = None
        return
    if not worst < 6e-2:
        print(f"FMHA off: mismatch {worst:.2e}", flush=True)
        st["fmha"] = None


import ctypes as _ct

_CTX_TLS = _fmha_threading.local()
_CTX_API = []


