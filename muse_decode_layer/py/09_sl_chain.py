_SL_GLUE_CUDA = __cuda__("sl_glue.cu")

@_lru_cache(None)
def _sl_cluster_module():
    from tvm_ffi.cpp import load_inline as _tvm_load_inline
    return _tvm_load_inline(name='sl_early_tail_' + _hashlib.md5(_SL_GLUE_CUDA.encode()).hexdigest()[:12],
                            cuda_sources=_SL_GLUE_CUDA, functions=['b1'],
                            extra_cuda_cflags=['-O3', '-std=c++17'])

@triton.jit
def _sl_cluster_finish(P, T, CARRY, TAIL, BETA, POST, NEXT, SOUT, HOUT,
                       BC: tl.constexpr, USE_PDL: tl.constexpr = True):
    t = tl.program_id(0)
    tile = tl.program_id(1)
    c = tl.arange(0,1024)
    beta = tl.load(BETA+c)
    gp = tl.load(POST+c).to(tl.float32)
    carry = tl.load(CARRY+t*8192+c)
    tail = tl.load(TAIL+t)
    o = tile*BC+tl.arange(0,BC)
    outcarry = tl.load(CARRY+t*8192+o)
    gn = tl.load(NEXT+o).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    f = (tl.load(P+t*1024+c).to(tl.float32)*gp).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(f*f,0)/8192.0+1e-8)
    n = (f*inv).to(tl.bfloat16).to(tl.float32)
    head = carry+beta*n
    invnext = tl.rsqrt((tail+tl.sum(head*head,0))/8192.0+1e-5)
    result = tl.where(o<1024,tl.gather(head,o%1024,0),outcarry)
    tl.store(SOUT+t*8192+o,result)
    tl.store(HOUT+t*8192+o,(result*invnext*gn).to(tl.bfloat16))

SL_CLUSTER_CONFIG={1: (8, 256), 8: (16, 1024)}
def _sl_cluster_after_attention(data, state, o_in, gate, T):
    """Copy of the base _after_attention (one-program-per-token path) with the linked boundary/route kernels.
    Every kernel the link does not cover falls back to the base kernel with the base arguments."""
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    tail = torch.empty((T,), **f32)
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    hr = torch.empty((T, HIDDEN), **f32) if T <= 8 else None
    if gate is None:
        P, S = dense(o_in, state["o"], T, state, "o")
    else:
        P, S = dense_ng(o_in, gate, state["o"], T, state)
    if S != 1 or P.dtype != torch.bfloat16:
        return _link_after_attention(data,state,o_in,gate,T)
    g=state['link']
    _sl_cluster_module().b1(P.view(T,8192),data['stream'],g['pre1'],g['alpha1'],g['beta1'],h,s1,g['alpha2'],tail,hr if T <= 8 else s1,T <= 8)
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    P, S = _pool_router_fp32(h, hr, T, state, ppick)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "native")),
                               marlin_available=state["bank"] is not None, native_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "native"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the native NVFP4 kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:
        pooled, logits = P
        _link_al(pooled, logits, xm, ids, wts, lids)
        _sl_link_route[(T, 2)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=True, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], num_warps=SL_ROUTE_CONFIG[T][0], launch_pdl=True)
    elif S == 1 and P.shape[-1] == 4608 and P.dtype == torch.float32:
        allp = P.view(T, 4608)
        _link_al(allp, xm, ids, wts, lids)
        _sl_link_route[(T, 2)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=False, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], num_warps=SL_ROUTE_CONFIG[T][0], launch_pdl=True)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=False, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if use_ma:
        m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ext = _moe(T, state)
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    P, S = dense(m, wu, T, state, "un")
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if S != 1 or P.dtype != torch.bfloat16:
        raise RuntimeError('small-load cluster requires BF16 sliced unpooler')
    nw,bc=SL_CLUSTER_CONFIG[T]
    _sl_cluster_finish[(T,8192//bc)](P,T,s1,tail,g['beta2'],g['post2'],g['pre2'],s2,out,BC=bc,num_warps=nw,launch_pdl=True)
    return out, s2, ids


_sl_previous_after_attention=_after_attention

def _after_attention(data,state,o_in,gate,T):
    if T in SL_CLUSTER_CONFIG and 'link' in state and state.get('compose_link_enabled', {}).get(T, True):
        return _sl_cluster_after_attention(data,state,o_in,gate,T)
    return _sl_previous_after_attention(data,state,o_in,gate,T)

_sl_previous_prepare=prepare

def prepare(weights):
    st=_sl_previous_prepare(weights)
    _sl_cluster_module()
    return st
@triton.jit
def _sl_link_route(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, SPLIT: tl.constexpr, EARLY: tl.constexpr, MODE: tl.constexpr):
    # _pool_route_finish structure: program (t, 0) = pre-expert norm of the pooled row, (t, 1) = top-8 pick;
    # one warp each, both after the wait. Arithmetic is the base's _route/_route_split (fp32 pooled, 1 + w in fp32).
    t = tl.program_id(0)
    kind = tl.program_id(1)
    if EARLY:
        _gdc_launch()
    _gdc_wait()
    if not EARLY:
        _gdc_launch()
    if kind == 0:
        c = tl.arange(0, 4096)
        if SPLIT:
            pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
        else:
            pooled = tl.load(ALL + t * 4608 + c)
        gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
        x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        if SPLIT:
            logit = tl.load(LOGITS + t * 256 + e)
        else:
            logit = tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + tl.load(BIAS + e)
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
        lid = (ids - tl.load(LS) + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)

SL_ROUTE_CONFIG={1: (8, 0, 0), 8: (4, 0, 0)}
@triton.jit
def _sl_tile_route(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, SPLIT: tl.constexpr, EARLY: tl.constexpr, MODE: tl.constexpr, PARTS: tl.constexpr, PRELOAD: tl.constexpr):
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
        if SPLIT:
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

SL_TILE_CONFIG={1:(4,1,True),8:(4,4,True)}
def _sl_tile_after_attention(data, state, o_in, gate, T):
    """Copy of the base _after_attention (one-program-per-token path) with the linked boundary/route kernels.
    Every kernel the link does not cover falls back to the base kernel with the base arguments."""
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    tail = torch.empty((T,), **f32)
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    hr = torch.empty((T, HIDDEN), **f32) if T <= 8 else None
    if gate is None:
        P, S = dense(o_in, state["o"], T, state, "o")
    else:
        P, S = dense_ng(o_in, gate, state["o"], T, state)
    if S != 1 or P.dtype != torch.bfloat16:
        return _link_after_attention(data,state,o_in,gate,T)
    g=state['link']
    _sl_cluster_module().b1(P.view(T,8192),data['stream'],g['pre1'],g['alpha1'],g['beta1'],h,s1,g['alpha2'],tail,hr if T <= 8 else s1,T <= 8)
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    P, S = _pool_router_fp32(h, hr, T, state, ppick)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "native")),
                               marlin_available=state["bank"] is not None, native_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "native"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the native NVFP4 kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:
        pooled, logits = P
        _link_al(pooled, logits, xm, ids, wts, lids)
        _sl_tile_route[(T,SL_TILE_CONFIG[T][1]+1)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=True, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], PARTS=SL_TILE_CONFIG[T][1], PRELOAD=SL_TILE_CONFIG[T][2], num_warps=SL_TILE_CONFIG[T][0], launch_pdl=True)
    elif S == 1 and P.shape[-1] == 4608 and P.dtype == torch.float32:
        allp = P.view(T, 4608)
        _link_al(allp, xm, ids, wts, lids)
        _sl_tile_route[(T,SL_TILE_CONFIG[T][1]+1)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=False, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], PARTS=SL_TILE_CONFIG[T][1], PRELOAD=SL_TILE_CONFIG[T][2], num_warps=SL_TILE_CONFIG[T][0], launch_pdl=True)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=False, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if use_ma:
        m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ext = _moe(T, state)
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    P, S = dense(m, wu, T, state, "un")
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if S != 1 or P.dtype != torch.bfloat16:
        raise RuntimeError('small-load cluster requires BF16 sliced unpooler')
    nw,bc=SL_CLUSTER_CONFIG[T]
    _sl_cluster_finish[(T,8192//bc)](P,T,s1,tail,g['beta2'],g['post2'],g['pre2'],s2,out,BC=bc,num_warps=nw,launch_pdl=True)
    return out, s2, ids


_sl_tile_prior_after=_after_attention
def _after_attention(data,state,o_in,gate,T):
    if T in SL_TILE_CONFIG and 'link' in state and state.get('compose_link_enabled', {}).get(T, True):
        return _sl_tile_after_attention(data,state,o_in,gate,T)
    return _sl_tile_prior_after(data,state,o_in,gate,T)

# Physical helper warp zero in the small-token native builds.
MOE_TINY2_DEFINES += ("@moe_helper0",)
MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1")


# QKV <= 8 direct escape reads, preserving decode's PDL wait and launchers.
_DE_EXT=None
_DE_RUN=bfp__run_sk2
_DE_PREPARE=prepare

def _de_source():
 cfg=('qkv8n_r144','qkv_gate',2304,24,3,1,4,2,2,1)
 _,nstage=bfp__sk2_geom(cfg,8)
 dispatch=f'launch_sk2<1, 8, 3, 1, 4, 2, {nstage}, true, false, 2, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return;'
 source=bfp_SK2_SRC.replace('@@SK2_DISPATCH@@',dispatch).replace('@@SK2F_DISPATCH@@','')
 source=source.replace('(NSTAGE >= 4 ? 4 : NSTAGE)','4').replace('NW + (COMP ? NESC : 0)','NW')
 source=source.replace('  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");','')
 a=source.index('      int pe = 1;');b=source.index('\n    }\n    mbar_arrive(cbar);',a)
 source=source[:a]+r"""
      for(int pe=1+ew;pe<=cnt;pe+=NESC) {
        const int2 e=ptab_s[pe];const int kk=e.x&65535;
        if(lane<M && kk>=gs0*KST && kk<(gs0+nst)*KST)
          cr[lane*CP+(e.x>>16)] += __int_as_float(e.y)*__bfloat162float(x[(size_t)lane*K+kk]);
      }
"""+source[b:]
 shared=bfp_CUDA_SRC.replace('@@MBAR@@',bfp_MBAR_HELPERS).replace('@@BODY@@','')
 return shared+bfp_MAP_SRC+source

def _de_ext():
 global _DE_EXT
 if _DE_EXT is None:
  import hashlib
  source=_de_source();tag=hashlib.md5(source.encode()).hexdigest()[:12];source=source.replace('sk2_kernel','sk2_kernel_narrow_'+tag).replace('launch_sk2','launch_sk2_narrow_'+tag);cpp=bfp_CPP_SRC[bfp_CPP_SRC.index('void sk2('):bfp_CPP_SRC.index('void sk2f(')]+'void sk2_set_pdl(int64_t n);'
  _DE_EXT=load_inline('proj_dl_escape_'+hashlib.md5(source.encode()).hexdigest()[:12],'#include <torch/extension.h>\n'+cpp,
    cuda_sources=source,functions=['sk2','sk2_set_pdl'],no_implicit_headers=True,extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],verbose=False)
 return _DE_EXT

def prepare(weights):
 _de_ext()
 return _DE_PREPARE(weights)

def bfp__run_sk2(x,layer,m,n,k,state,dg=0,dbg=None):
 if layer!='qkv_gate' or not 2<=m<=8:return _DE_RUN(x,layer,m,n,k,state,dg,dbg)
 name=bfp__sk2_name(layer,m);ci,stream,tab,pmax,b4,n_rows=state['sk2'][name]
 out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
 ext=_de_ext();ext.sk2_set_pdl(bfp_SK2_PDL)
 ext.sk2(stream,x,out,tab,pmax,b4,m,n,k,ci,bfp__sk2_tokr(m),dg,state['dummy'] if dbg is None else dbg)
 return out

import threading as _GlueContextThreading
_GLUE_CONTEXT_TLS = _GlueContextThreading.local()
_GLUE_CONTEXT_API = None

