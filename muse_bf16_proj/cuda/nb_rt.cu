
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#define RD __device__ __forceinline__
RD uint32_t routertail_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
RD void routertail_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void routertail_expect(uint32_t b, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void routertail_wait(uint32_t b, int ph) {
  asm volatile("{ .reg .pred p; wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra wait; }" ::"r"(b),"r"(ph):"memory");
}
RD void routertail_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" :::"memory"); }
RD uint32_t routertail_map(uint32_t a, int r) { uint32_t b; asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(b):"r"(a),"r"(r));return b; }
RD void routertail_push(uint32_t d, float4 v, uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];" ::"r"(d),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");
}
RD float4 routertail_pull(uint32_t d) {
  float4 v; asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(d):"memory");return v;
}
RD void routertail_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }
RD uint64_t routertail_desc(uint32_t a) { return (uint64_t)((a&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62); }
RD void routertail_tma(uint32_t dst, const void* tm, int k,int r,uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::"r"(dst),"l"(tm),"r"(k),"r"(r),"r"(bar):"memory");
}

template<int MT,int S,int WK>
__global__ __launch_bounds__(WK*32) void routertail_small_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m) {
  constexpr int KU=128/S/WK, NI=MT*32, P=4, OWN=NI/P, PARTS=S*1;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;
  const int gw=(rank*16+row/16)*WK+warp;
  extern __shared__ __align__(16) unsigned char sm[];
  float4* recv=(float4*)sm;
  float4* part=recv+PARTS*OWN;
  const uint32_t rb=routertail_smem(sm+(PARTS*OWN+WK*NI)*16);
  int4 w[KU][8];uint4 xx[KU][MT][2];
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int j=0;j<8;j++) {
      const int4* p=wp+((k*(16*S*WK)+gw)*8+j)*32+lane;
      asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];":"=r"(w[k][j].x),"=r"(w[k][j].y),"=r"(w[k][j].z),"=r"(w[k][j].w):"l"(p));
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;
#pragma unroll
      for(int h=0;h<2;h++) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(xx[k][n][h].x),"=r"(xx[k][n][h].y),"=r"(xx[k][n][h].z),"=r"(xx[k][n][h].w):"l"(p+h*8));
    }
  }
  if(rank<P && tid==0) {routertail_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");}
  __syncthreads();
  if(rank<P && tid==0) {routertail_expect(rb,PARTS*OWN*16);asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");}
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
  float c[2][MT][4]={};
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int n=0;n<MT;n++) {
      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};
#pragma unroll
      for(int q=0;q<4;q++) {
#pragma unroll
        for(int h=0;h<2;h++) {
          const int4 a=w[k][q+h*4];
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       :"+f"(c[h][n][0]),"+f"(c[h][n][1]),"+f"(c[h][n][2]),"+f"(c[h][n][3]):"r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b[2*q]),"r"(b[2*q+1]));
        }
      }
    }
  }
#pragma unroll
  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
  if(tid<NI) {
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<WK;k++) routertail_add(sum,part[k*NI+tid]);
    const int dst=tid/OWN,local=tid%OWN;
    routertail_push(routertail_map(routertail_smem(recv+rank*OWN+local),dst),sum,routertail_map(rb,dst));
  }

  if(rank<P && tid<OWN) {
    routertail_wait(rb,0);
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<PARTS;k++) routertail_add(sum,recv[k*OWN+tid]);
    const int v=rank*OWN+tid,ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);
    if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}
    if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}
  }

}

template<int MT,int S,int WK>
int routertail_small_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto fn=routertail_small_kernel<MT,S,WK>;
  constexpr int bytes=(S*1/4+WK)*MT*512+16;
  static bool configured=false;
  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;
  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m);
}
int routertail_small(int64_t w,int64_t x,int64_t out,int m,int s,int wk,int64_t stream,int query) {
#define RSM(S,W) if(s==S && wk==W) {if(m<=8)return routertail_small_t<1,S,W>(w,x,out,m,stream,query);if(m<=16)return routertail_small_t<2,S,W>(w,x,out,m,stream,query);if(m<=32)return routertail_small_t<4,S,W>(w,x,out,m,stream,query);}
RSM(8,8) RSM(16,8) RSM(16,4)
return -99;
}

// ---- c12: the same kernel on lossless 12-bit planes (hi and lo BF16 of the FP32 router): per (16 rows x 64 k) warp tile and
// plane the lane reads 3 x 16 B (2 low-byte words groups + 1 nibble group) instead of 4 x 16 B; exact escape corrections in an
// extra warp. Plane layout [ku][S][16][WK][h][part 0..2][lane] int4; nibble = sign | (exp7 - base) (3 bits), 4 per byte pair.
RD void rt12_decode8(uint32_t l0, uint32_t l1, uint32_t nw, uint32_t b4, uint32_t* r) {
  const uint32_t ha = ((nw & 0x07070707u) + b4) | ((nw << 4) & 0x80808080u);
  const uint32_t hb = (((nw >> 4) & 0x07070707u) + b4) | (nw & 0x80808080u);
  r[0] = __byte_perm(l0, ha, 0x5140); r[1] = __byte_perm(l0, ha, 0x7362);
  r[2] = __byte_perm(l1, hb, 0x5140); r[3] = __byte_perm(l1, hb, 0x7362);
}
template<int MT,int S,int WK>
__global__ __launch_bounds__(WK*32) void routertail_c12_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m,const unsigned* etab,uint32_t bqh,uint32_t bql) {
  constexpr int KU=128/S/WK, NI=MT*32, P=4, OWN=NI/P, PARTS=S*1;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;
  const int gw=(rank*16+row/16)*WK+warp;
  extern __shared__ __align__(16) unsigned char sm[];
  float4* recv=(float4*)sm;
  float4* part=recv+PARTS*OWN;
  const uint32_t rb=routertail_smem(sm+(PARTS*OWN+WK*NI)*16);
  int4 w[KU][2][3];uint4 xx[KU][MT][2];
  const unsigned eword=lane<16?__ldg(etab+(size_t)(blockIdx.x*WK+warp)*16+lane):0u;   // [cta][warp][16]: word 0 = count, then entries
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int j=0;j<6;j++) {
      const int4* p=wp+((k*(16*S*WK)+gw)*6+j)*32+lane;
      asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];":"=r"(w[k][j/3][j%3].x),"=r"(w[k][j/3][j%3].y),"=r"(w[k][j/3][j%3].z),"=r"(w[k][j/3][j%3].w):"l"(p));
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;
#pragma unroll
      for(int h=0;h<2;h++) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(xx[k][n][h].x),"=r"(xx[k][n][h].y),"=r"(xx[k][n][h].z),"=r"(xx[k][n][h].w):"l"(p+h*8));
    }
  }
  if(rank<P && tid==0) {routertail_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");}
  __syncthreads();
  if(rank<P && tid==0) {routertail_expect(rb,PARTS*OWN*16);asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");}
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
  float c[2][MT][4]={};
  const int ecnt=__shfl_sync(0xffffffffu,(int)eword,0);
  const uint32_t b4h=bqh*0x01010101u,b4l=bql*0x01010101u;
#pragma unroll
  for(int k=0;k<KU;k++) {
    uint32_t a[2][4][4];
#pragma unroll
    for(int h=0;h<2;h++) {
      const uint32_t b4=h?b4l:b4h;
      rt12_decode8(w[k][h][0].x,w[k][h][0].y,w[k][h][2].x,b4,a[h][0]);
      rt12_decode8(w[k][h][0].z,w[k][h][0].w,w[k][h][2].y,b4,a[h][1]);
      rt12_decode8(w[k][h][1].x,w[k][h][1].y,w[k][h][2].z,b4,a[h][2]);
      rt12_decode8(w[k][h][1].z,w[k][h][1].w,w[k][h][2].w,b4,a[h][3]);
    }
    // exact escapes: overwrite the mis-decoded element with the true BF16 (entry: lane 0-4, q 5-6, reg 7-8, half 9, h 10, k 11, bits 16-31)
    for(int e=0;e<ecnt;e++) {
      const unsigned en=__shfl_sync(0xffffffffu,(int)eword,e+1);
      if(((en>>11)&1)!=(unsigned)k && KU>1) continue;
      if((int)(en&31)!=lane) continue;
      const int q=(en>>5)&3,rg=(en>>7)&3,hf=(en>>9)&1,hh=(en>>10)&1;
      const uint32_t sh=hf*16,msk=0xFFFFu<<sh,val=(en>>16)<<sh;
#pragma unroll
      for(int hq=0;hq<2;hq++)
#pragma unroll
        for(int qq=0;qq<4;qq++)
#pragma unroll
          for(int rr=0;rr<4;rr++)
            if(hq==hh && qq==q && rr==rg) a[hq][qq][rr]=(a[hq][qq][rr]&~msk)|val;
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};
#pragma unroll
      for(int q=0;q<4;q++) {
#pragma unroll
        for(int h=0;h<2;h++) {
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       :"+f"(c[h][n][0]),"+f"(c[h][n][1]),"+f"(c[h][n][2]),"+f"(c[h][n][3]):"r"(a[h][q][0]),"r"(a[h][q][1]),"r"(a[h][q][2]),"r"(a[h][q][3]),"r"(b[2*q]),"r"(b[2*q+1]));
        }
      }
    }
  }
#pragma unroll
  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
  if(tid<NI) {
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<WK;k++) routertail_add(sum,part[k*NI+tid]);
    const int dst=tid/OWN,local=tid%OWN;
    routertail_push(routertail_map(routertail_smem(recv+rank*OWN+local),dst),sum,routertail_map(rb,dst));
  }

  if(rank<P && tid<OWN) {
    routertail_wait(rb,0);
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<PARTS;k++) routertail_add(sum,recv[k*OWN+tid]);
    const int v=rank*OWN+tid,ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);
    if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}
    if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}
  }
}
template<int MT,int S,int WK>
int routertail_c12_t(int64_t w,int64_t x,int64_t out,int m,int64_t etab,int bqh,int bql,int64_t stream) {
  auto fn=routertail_c12_kernel<MT,S,WK>;
  constexpr int bytes=(S*1/4+WK)*MT*512+16;
  static bool configured=false;
  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;
  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m,(const unsigned*)etab,(uint32_t)bqh,(uint32_t)bql);
}
int routertail_c12(int64_t w,int64_t x,int64_t out,int m,int s,int wk,int64_t etab,int bqh,int bql,int64_t stream) {
#define RSC(S,W) if(s==S && wk==W) {if(m<=8)return routertail_c12_t<1,S,W>(w,x,out,m,etab,bqh,bql,stream);if(m<=16)return routertail_c12_t<2,S,W>(w,x,out,m,etab,bqh,bql,stream);if(m<=32)return routertail_c12_t<4,S,W>(w,x,out,m,etab,bqh,bql,stream);}
RSC(8,8) RSC(16,8) RSC(16,4)
return -99;
}
