def _compose_times(fns, reps=16):
    # All candidates see the same synthetic inputs and read-only cold-cache sweep.
    graphs, outputs = [], []
    for fn in fns:
        fn()
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out=fn()
        graphs.append(g);outputs.append(out)
    flush=torch.ones((65536,1024),device='cuda',dtype=torch.float32)
    sink=torch.empty(1024,device='cuda',dtype=torch.float32)
    events=[]
    for r in range(reps):
        order=range(len(graphs)) if r%2==0 else reversed(range(len(graphs)))
        for i in order:
            torch.sum(flush,dim=0,out=sink)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();graphs[i].replay();b.record()
            events.append((i,a,b))
    torch.cuda.synchronize()
    samples=[[] for _ in graphs]
    for i,a,b in events:samples[i].append(a.elapsed_time(b)*1000)
    return [sorted(s)[len(s)//2] for s in samples]


_unpinned_prepare = prepare

def prepare(weights):
    st = _unpinned_prepare(weights)
    bfp = st.get("bfp")
    if bfp is not None:
        for T in (1, 8, 16, 32):
            for name, layer in _BFP_LAYER.items():
                bfp["choice"][(layer, T)] = bfp__static_choice(layer, T, bfp)
            for name in ("qkv", "o", "un"):
                st["mm_pick"][(T, name)] = "bfp"
            can_fuse = (bfp["choice"][("pooler", T)][0] == "sk2"
                        and bfp["choice"][("router", T)][0] == "sk2")
            st["mm_pick"][(T, "pr")] = "bfpf" if T <= 32 and can_fuse else "bfp"
    # The link's fixed Lt plans handle these shapes. Explicit pr=bfp would
    # bypass the plan and run two separate projections (measured +10..12us).
    # qkv must be True for the link's direct Lt/attention path.
    for T in (128, 256):
        for name in ("qkv", "o", "pr", "un"):
            st["mm_pick"][(T, name)] = True
    return st

_policy_parent_prepare=prepare


def _policy_variant(st,T,mode):
    ss=dict(st)
    ss['mm_pick']=dict(st['mm_pick'])
    ss['link_lt']=dict(st.get('link_lt',{}))
    ss['compose_link_enabled']=dict(st.get('compose_link_enabled',{}))
    ss['compose_boundary_warps']=dict(st.get('compose_boundary_warps',{}))
    link,bw,drops=mode
    ss['compose_link_enabled'][T]=link
    ss['compose_boundary_warps'][T]=bw
    for name in drops:ss['link_lt'].pop((name,T),None)
    return ss


def prepare(weights):
    st=_policy_parent_prepare(weights)
    st['compose_base_mm']=dict(st['mm_pick'])
    st['compose_base_link_lt']=dict(st['link_lt'])
    st['compose_active']=set()
    st['compose_selection']=[]
    # Exactly preserve head behavior until a complete-chain measurement picks another path.
    st['compose_link_enabled']={T:T in (1,8,128,256) for T in (1,8,16,32,128,256)}
    st['compose_boundary_warps']={}
    gen=torch.Generator(device=st['qkv'].device).manual_seed(1618033)
    for T in (1,8,16,32,128,256):
        original=st['compose_link_enabled'][T]
        modes=[(original,16,()),(True,16,()),(False,16,()),(False,8,()),(False,4,())]
        # Keep measured large-T Lt plans; only compare complete glue paths.
        modes=list(dict.fromkeys(modes))
        data={'stream':torch.randn(T,HIDDEN,device=st['qkv'].device,generator=gen),
              'local_start':torch.zeros(1,device=st['qkv'].device,dtype=torch.int32)}
        o_in=(.1*torch.randn(T,1024,device=st['qkv'].device,generator=gen)).to(torch.bfloat16)
        # Synthetic only: route to its busiest EP range to avoid benchmarking idle experts.
        ids=_after_attention(data,st,o_in,None,T)[2]
        counts=torch.bincount(ids.flatten().long()//32,minlength=8)
        data['local_start'].fill_(32*int(counts.argmax()))
        variants=[_policy_variant(st,T,mode) for mode in modes]
        ts=_compose_times([lambda ss=ss:_after_attention(data,ss,o_in,None,T) for ss in variants],reps=32)
        best=min(range(len(ts)),key=ts.__getitem__)
        if ts[best]>=ts[0]-.4:best=0
        picked=variants[best]
        st['compose_link_enabled'][T]=picked['compose_link_enabled'][T]
        st['compose_boundary_warps'][T]=picked['compose_boundary_warps'][T]
        st['link_lt']=picked['link_lt']
        st['compose_selection'].append({'T':T,'region':'chain_policy','us':ts,'modes':modes,'choice':modes[best]})
    return st


